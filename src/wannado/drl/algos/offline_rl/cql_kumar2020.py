"""CQL: Conservative Q-Learning for Offline Reinforcement Learning (Kumar et al., NeurIPS 2020).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : offline_rl / S (Foundational Offline RL Benchmark)
- Parent        : `sac` (Haarnoja et al., 2018)
- Key Diff      : When training purely from static logged datasets with zero online
                  environment interaction, naive off-policy critics hallucinate wildly
                  inflated Q-values on unseen actions. CQL adds a log-sum-exp conservative
                  penalty pushing down Q(s, a) everywhere except dataset-supported actions.
- Extension Hook [AI-Prompt Hook]:
    1. Avoid OOD action queries altogether via expectile regression -> `iql_offline`.
    2. Tune `cql_alpha` (conservative penalty strength) against held-out Q-gap.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import (
    finalize_offline_run,
    maybe_offline_periodic_hooks,
    prepare_offline_run,
    setup_offline_run,
)
from wannado.drl.core.networks import QNetwork, SquashedGaussianActor, TwinQCritic
from wannado.drl.core.obs_spec import obs_to_tensor, slice_obs
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import OfflineTransitionDataset

META = AlgoMeta(
    id="cql",
    name="Conservative Q-Learning (CQL)",
    family="offline_rl",
    paradigm="offline",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("discrete", "continuous"),
    env_modalities=("offline_rl", "discrete_control", "continuous_control"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="sac",
    diff_from_parent="Adds conservative log-sum-exp Q-penalty so static offline datasets never over-estimate unseen OOD actions.",
    summary_plain="Learns safely from logged historical datasets without live interaction by penalizing untested actions.",
    when_to_use="Zero online exploration allowed (only static trajectories/logs exist) and dataset quality is mixed.",
    when_not_to_use="Online interaction is cheap and safe (`sac`/`ppo`) or you want simpler hyperparameter tuning (`iql_offline`).",
    hyperparam_intuition={
        "cql_alpha": "Strength of penalty pushing down Q-values on unseen out-of-distribution actions.",
        "num_grad_steps": "Number of offline minibatch gradient updates over the static dataset.",
        "gamma": "Discount horizon.",
    },
    extension_hooks=(
        "Add automatic Lagrange dual alpha_cql targeting fixed Q-gap",
        "Replace log-sum-exp penalty with in-sample expectile regression -> `iql_offline`",
    ),
    module_path="wannado.drl.algos.offline_rl.cql_kumar2020",
    citation=AlgoCitation(
        key="kumar2020conservative",
        title="Conservative Q-Learning for Offline Reinforcement Learning",
        authors="Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine",
        venue="Advances in Neural Information Processing Systems (NeurIPS)",
        year=2020,
        url="https://arxiv.org/abs/2006.04779",
        bibtex=(
            "@inproceedings{kumar2020conservative,\n"
            "  title={Conservative Q-Learning for Offline Reinforcement Learning},\n"
            "  author={Kumar, Aviral and Zhou, Aurick and Tucker, George and Levine, Sergey},\n"
            "  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},\n"
            "  year={2020},\n"
            "  url={https://arxiv.org/abs/2006.04779}\n"
            "}"
        ),
        ieee='A. Kumar, A. Zhou, G. Tucker, and S. Levine, "Conservative Q-learning for offline reinforcement learning," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.',
        apa="Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-learning for offline reinforcement learning. NeurIPS.",
    ),
)


@dataclass
class CQLConfig:
    num_grad_steps: int = 2_500
    batch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    cql_alpha: float = 1.0
    num_random_actions: int = 8
    hidden_dims: tuple[int, ...] = (128, 128)
    extractor_type: str = "auto"
    extractor_kwargs: dict[str, Any] | None = None
    normalize_obs: bool = True
    eval_episodes: int = 10
    checkpoint_dir: str | None = None
    checkpoint_freq: int = 0
    eval_freq: int = 0
    save_best_model: bool = True
    keep_last_k: int = 3
    resume_from: str | None = None
    tb_log_dir: str | None = None
    device: str = "auto"
    seed: int = 42


def _repeat_batch_obs(b_obs: Any, n_rep: int) -> Any:
    if isinstance(b_obs, dict):
        return {
            k: v.unsqueeze(1).expand(-1, n_rep, *v.shape[1:]).reshape(-1, *v.shape[1:])
            for k, v in b_obs.items()
        }
    return b_obs.unsqueeze(1).expand(-1, n_rep, *b_obs.shape[1:]).reshape(-1, *b_obs.shape[1:])


def train_cql(
    dataset: OfflineTransitionDataset,
    config: CQLConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Conservative Q-Learning (CQL) purely from a static OfflineTransitionDataset."""
    cfg = config or CQLConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    n_data = len(dataset)
    if n_data < 2:
        raise ValueError("Offline dataset must contain at least 2 transitions.")

    ctx = setup_offline_run(
        dataset,
        extractor_type=cfg.extractor_type,
        normalize_obs=cfg.normalize_obs,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_freq=cfg.checkpoint_freq,
        eval_freq=cfg.eval_freq,
        save_best_model=cfg.save_best_model,
        keep_last_k=cfg.keep_last_k,
        tb_log_dir=cfg.tb_log_dir,
        eval_env_or_factory=eval_env_or_factory,
        device=dev,
    )
    obs_spec = ctx.obs_spec
    cfg_dict = asdict(cfg)
    t0 = time.perf_counter()
    q_hist, cql_gap_hist = [], []

    if dataset.is_discrete:
        q_net = QNetwork(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_net = QNetwork(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        hard_update(q_net, target_net)
        opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

        policy_kind = "q_network"
        primary_key = "q_net"
        policy_spec = {
            "net_class": "QNetwork",
            "init_kwargs": {
                "action_dim": int(dataset.action_dim),
                "hidden_dims": list(cfg.hidden_dims),
                "extractor_type": cfg.extractor_type,
                "extractor_kwargs": cfg.extractor_kwargs,
            },
        }
        models_map: dict[str, Any] = {"q_net": q_net, "target_net": target_net}
        opts_map: dict[str, torch.optim.Optimizer] = {"q_net": opt}
        obs_all, next_obs_all, start_step, learning_curve = prepare_offline_run(
            ctx, dataset, resume_from=cfg.resume_from, models=models_map, optimizers=opts_map, rng=rng
        )

        masks_raw = dataset.extra.get("action_masks")
        next_masks_raw = dataset.extra.get("next_action_masks", masks_raw)
        masks_all = np.asarray(masks_raw, dtype=np.bool_) if masks_raw is not None else None
        next_masks_all = np.asarray(next_masks_raw, dtype=np.bool_) if next_masks_raw is not None else None

        for step in range(start_step + 1, cfg.num_grad_steps + 1):
            idx = rng.integers(0, n_data, size=cfg.batch_size)
            b_obs = obs_to_tensor(slice_obs(obs_all, idx), obs_spec, dev, add_batch_dim=False)
            b_act = torch.as_tensor(dataset.actions[idx], device=dev, dtype=torch.long)
            b_rew = torch.as_tensor(dataset.rewards[idx], device=dev, dtype=torch.float32)
            b_next = obs_to_tensor(slice_obs(next_obs_all, idx), obs_spec, dev, add_batch_dim=False)
            b_term = torch.as_tensor(dataset.terminateds[idx], device=dev, dtype=torch.float32)

            with torch.no_grad():
                q_next_all = target_net(b_next)
                if next_masks_all is not None:
                    b_nmask = torch.as_tensor(next_masks_all[idx], device=dev, dtype=torch.bool)
                    has_valid = b_nmask.any(dim=1, keepdim=True)
                    safe_mask = torch.where(has_valid, b_nmask, torch.ones_like(b_nmask))
                    q_next_all = q_next_all.masked_fill(~safe_mask, -1e9)
                td_target = b_rew + cfg.gamma * (1.0 - b_term) * q_next_all.max(dim=1).values
            q_all = q_net(b_obs)
            q_data = q_all.gather(1, b_act.unsqueeze(1)).squeeze(1)
            td_loss = F.mse_loss(q_data, td_target)
            # [AI-Prompt Hook]: Discrete CQL log-sum-exp OOD penalty (respecting feasibility masks when supplied).
            q_for_lse = q_all
            if masks_all is not None:
                b_mask = torch.as_tensor(masks_all[idx], device=dev, dtype=torch.bool)
                b_mask = b_mask.scatter(1, b_act.unsqueeze(1), True)
                q_for_lse = q_for_lse.masked_fill(~b_mask, -1e9)
            cql_pen = (torch.logsumexp(q_for_lse, dim=1) - q_data).mean()
            loss = td_loss + cfg.cql_alpha * cql_pen

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
            opt.step()
            polyak_update(q_net, target_net, cfg.tau)

            q_hist.append(float(q_data.mean().item()))
            cql_gap_hist.append(float(cql_pen.item()))
            if step % max(1, cfg.num_grad_steps // 10) == 0:
                learning_curve.append({"step": float(step), "td_loss": float(td_loss.item()), "cql_gap": float(cql_pen.item())})
                if ctx.ckpt_mgr is not None:
                    ctx.ckpt_mgr.log_scalars(step, {"train/td_loss": float(td_loss.item()), "train/cql_gap": float(cql_pen.item())})

            maybe_offline_periodic_hooks(
                ctx,
                step,
                algo_id=META.id,
                policy_kind=policy_kind,
                policy_spec=policy_spec,
                primary_model_key=primary_key,
                models=models_map,
                optimizers=opts_map,
                config_dict=cfg_dict,
                eval_episodes=cfg.eval_episodes,
                seed=cfg.seed,
                learning_curve=learning_curve,
                rng=rng,
            )

        result_model: Any = q_net
    else:
        act_low = dataset.action_low if dataset.action_low is not None else -np.ones(dataset.action_dim, dtype=np.float32)
        act_high = dataset.action_high if dataset.action_high is not None else np.ones(dataset.action_dim, dtype=np.float32)
        actor = SquashedGaussianActor(
            obs_spec,
            dataset.action_dim,
            act_low,
            act_high,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        critic = TwinQCritic(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            action_low=act_low,
            action_high=act_high,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_critic = TwinQCritic(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            action_low=act_low,
            action_high=act_high,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        hard_update(critic, target_critic)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
        low_t = torch.as_tensor(act_low, device=dev, dtype=torch.float32)
        high_t = torch.as_tensor(act_high, device=dev, dtype=torch.float32)

        policy_kind = "squashed_actor"
        primary_key = "actor"
        policy_spec = {
            "net_class": "SquashedGaussianActor",
            "init_kwargs": {
                "action_dim": int(dataset.action_dim),
                "action_low": act_low.tolist(),
                "action_high": act_high.tolist(),
                "hidden_dims": list(cfg.hidden_dims),
                "extractor_type": cfg.extractor_type,
                "extractor_kwargs": cfg.extractor_kwargs,
            },
        }
        models_map = {"actor": actor, "critic": critic, "target_critic": target_critic}
        opts_map = {"actor": actor_opt, "critic": critic_opt}
        obs_all, next_obs_all, start_step, learning_curve = prepare_offline_run(
            ctx, dataset, resume_from=cfg.resume_from, models=models_map, optimizers=opts_map, rng=rng
        )

        for step in range(start_step + 1, cfg.num_grad_steps + 1):
            idx = rng.integers(0, n_data, size=cfg.batch_size)
            b_obs = obs_to_tensor(slice_obs(obs_all, idx), obs_spec, dev, add_batch_dim=False)
            b_act = torch.as_tensor(dataset.actions[idx], device=dev, dtype=torch.float32)
            b_rew = torch.as_tensor(dataset.rewards[idx], device=dev, dtype=torch.float32)
            b_next = obs_to_tensor(slice_obs(next_obs_all, idx), obs_spec, dev, add_batch_dim=False)
            b_term = torch.as_tensor(dataset.terminateds[idx], device=dev, dtype=torch.float32)

            with torch.no_grad():
                na, nlp = actor.get_action(b_next)
                tq1, tq2 = target_critic(b_next, na)
                td_target = b_rew.unsqueeze(-1) + cfg.gamma * (1.0 - b_term.unsqueeze(-1)) * (torch.min(tq1, tq2) - 0.1 * nlp)

            q1, q2 = critic(b_obs, b_act)
            td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

            rep_obs = _repeat_batch_obs(b_obs, cfg.num_random_actions)
            n_cand = cfg.batch_size * cfg.num_random_actions
            rand_a = low_t + torch.rand(n_cand, dataset.action_dim, device=dev) * (high_t - low_t)
            with torch.no_grad():
                pol_a, pol_lp = actor.get_action(rep_obs)
            # Evaluate random & policy candidates separately so (B, K) rows stay strictly aligned per state
            # and apply Kumar et al. (2020, Appendix F) importance-sampling density correction.
            cq1_r, cq2_r = critic(rep_obs, rand_a)
            cq1_p, cq2_p = critic(rep_obs, pol_a)
            cq1_r = cq1_r.view(cfg.batch_size, cfg.num_random_actions)
            cq2_r = cq2_r.view(cfg.batch_size, cfg.num_random_actions)
            cq1_p = cq1_p.view(cfg.batch_size, cfg.num_random_actions)
            cq2_p = cq2_p.view(cfg.batch_size, cfg.num_random_actions)
            pol_lp_mat = pol_lp.view(cfg.batch_size, cfg.num_random_actions)
            log_unif = -float(np.sum(np.log(np.maximum(act_high - act_low, 1e-6))))
            cat_q1 = torch.cat([cq1_r - log_unif, cq1_p - pol_lp_mat], dim=1)
            cat_q2 = torch.cat([cq2_r - log_unif, cq2_p - pol_lp_mat], dim=1)
            log_2k = math.log(2.0 * cfg.num_random_actions)
            cql_pen = (torch.logsumexp(cat_q1, dim=1) - log_2k - q1.squeeze(-1)).mean() + (
                torch.logsumexp(cat_q2, dim=1) - log_2k - q2.squeeze(-1)
            ).mean()

            critic_loss = td_loss + 0.5 * cfg.cql_alpha * cql_pen
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            pi_a, log_pi = actor.get_action(b_obs)
            q1_pi, q2_pi = critic(b_obs, pi_a)
            actor_loss = (0.1 * log_pi - torch.min(q1_pi, q2_pi)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            polyak_update(critic, target_critic, cfg.tau)

            q_hist.append(float(q1.mean().item()))
            cql_gap_hist.append(float(cql_pen.item()))
            if step % max(1, cfg.num_grad_steps // 10) == 0:
                learning_curve.append({"step": float(step), "td_loss": float(td_loss.item()), "cql_gap": float(cql_pen.item())})
                if ctx.ckpt_mgr is not None:
                    ctx.ckpt_mgr.log_scalars(step, {"train/td_loss": float(td_loss.item()), "train/cql_gap": float(cql_pen.item())})

            maybe_offline_periodic_hooks(
                ctx,
                step,
                algo_id=META.id,
                policy_kind=policy_kind,
                policy_spec=policy_spec,
                primary_model_key=primary_key,
                models=models_map,
                optimizers=opts_map,
                config_dict=cfg_dict,
                eval_episodes=cfg.eval_episodes,
                seed=cfg.seed,
                learning_curve=learning_curve,
                rng=rng,
            )

        result_model = {"actor": actor, "critic": critic}

    return finalize_offline_run(
        ctx,
        dataset,
        algo_id=META.id,
        policy_kind=policy_kind,
        policy_spec=policy_spec,
        primary_model_key=primary_key,
        models=models_map,
        optimizers=opts_map,
        result_model=result_model,
        config_dict=cfg_dict,
        num_grad_steps=cfg.num_grad_steps,
        gamma=cfg.gamma,
        t0=t0,
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
            "max_q_value": float(max(q_hist, key=abs)) if q_hist else 0.0,
            "mean_cql_ood_gap": float(np.mean(cql_gap_hist)) if cql_gap_hist else 0.0,
            "is_continuous": 0.0 if dataset.is_discrete else 1.0,
        },
        eval_episodes=cfg.eval_episodes,
        seed=cfg.seed,
        rng=rng,
        explicit_optimizations=[] if dataset.is_discrete else ["clipped_double_q"],
    )
