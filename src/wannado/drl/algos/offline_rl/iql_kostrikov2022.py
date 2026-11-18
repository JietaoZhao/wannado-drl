"""IQL: Implicit Q-Learning for Offline RL (Kostrikov, Nair, & Levine, ICLR 2022).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : offline_rl / S (Clean, OOD-Free Offline RL SOTA)
- Parent        : `cql` (Kumar et al., 2020)
- Key Diff      : Eliminates querying out-of-distribution actions altogether! Uses
                  asymmetric expectile regression (`expectile` tau in (0.5, 1.0)) on in-dataset
                  (s, a) pairs to estimate an upper state-value V(s), then trains the actor
                  via Advantage-Weighted Regression (AWR).
- Extension Hook [AI-Prompt Hook]:
    1. Tune `expectile` (0.7-0.9) and `awr_beta` temperature to trade off safety vs stitching.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from wannado.drl.core.algo_helpers import (
    finalize_offline_run,
    maybe_offline_periodic_hooks,
    prepare_offline_run,
    setup_offline_run,
)
from wannado.drl.core.networks import (
    ContinuousGaussianActor,
    DiscretePolicyNet,
    QNetwork,
    TwinQCritic,
    ValueNetwork,
)
from wannado.drl.core.obs_spec import obs_to_tensor, slice_obs
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import OfflineTransitionDataset

META = AlgoMeta(
    id="iql_offline",
    name="Implicit Q-Learning (IQL)",
    family="offline_rl",
    paradigm="offline",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("discrete", "continuous"),
    env_modalities=("offline_rl", "discrete_control", "continuous_control"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="cql",
    diff_from_parent="Replaces OOD action penalization with in-sample expectile V(s) regression + Advantage-Weighted Regression (AWR).",
    summary_plain="Learns purely from logged datasets without ever querying unseen actions; fast and very stable.",
    when_to_use="Preferred default for static offline datasets where CQL hyperparameter tuning feels brittle.",
    when_not_to_use="Online environments where live exploration is allowed (`ppo`/`sac`) or multi-agent systems (`mappo`).",
    hyperparam_intuition={
        "expectile": "Asymmetry parameter (0.7-0.8) steering V(s) toward the upper tail of dataset actions.",
        "awr_beta": "Inverse temperature controlling how sharply policy extraction favors high-advantage actions.",
        "num_grad_steps": "Offline gradient iterations over logged transitions.",
    },
    extension_hooks=(
        "Swap Gaussian AWR actor for diffusion/flow behavior policy",
        "Fine-tune online after offline pretraining using `sac`",
    ),
    module_path="wannado.drl.algos.offline_rl.iql_kostrikov2022",
    citation=AlgoCitation(
        key="kostrikov2022offline",
        title="Offline Reinforcement Learning with Implicit Q-Learning",
        authors="Ilya Kostrikov, Ashvin Nair, and Sergey Levine",
        venue="International Conference on Learning Representations (ICLR)",
        year=2022,
        url="https://arxiv.org/abs/2110.06169",
        bibtex=(
            "@inproceedings{kostrikov2022offline,\n"
            "  title={Offline Reinforcement Learning with Implicit Q-Learning},\n"
            "  author={Kostrikov, Ilya and Nair, Ashvin and Levine, Sergey},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2022},\n"
            "  url={https://arxiv.org/abs/2110.06169}\n"
            "}"
        ),
        ieee='I. Kostrikov, A. Nair, and S. Levine, "Offline reinforcement learning with implicit Q-learning," in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.',
        apa="Kostrikov, I., Nair, A., & Levine, S. (2022). Offline reinforcement learning with implicit Q-learning. ICLR.",
    ),
)


@dataclass
class IQLConfig:
    num_grad_steps: int = 2_500
    batch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.75
    awr_beta: float = 3.0
    max_awr_weight: float = 100.0
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


def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0.0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def train_iql_offline(
    dataset: OfflineTransitionDataset,
    config: IQLConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Implicit Q-Learning (IQL) purely from an OfflineTransitionDataset."""
    cfg = config or IQLConfig()
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

    v_net = ValueNetwork(
        obs_spec,
        cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=cfg.lr)

    if dataset.is_discrete:
        q_net = QNetwork(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_q = QNetwork(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        hard_update(q_net, target_q)
        actor_logits = DiscretePolicyNet(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        q_opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
        actor_opt = torch.optim.Adam(actor_logits.parameters(), lr=cfg.lr)

        policy_kind = "discrete_policy"
        primary_key = "actor"
        policy_spec = {
            "net_class": "DiscretePolicyNet",
            "init_kwargs": {
                "action_dim": int(dataset.action_dim),
                "hidden_dims": list(cfg.hidden_dims),
                "extractor_type": cfg.extractor_type,
                "extractor_kwargs": cfg.extractor_kwargs,
            },
        }
        models_map: dict[str, Any] = {"actor": actor_logits, "q": q_net, "target_q": target_q, "v": v_net}
        opts_map: dict[str, torch.optim.Optimizer] = {"actor": actor_opt, "q": q_opt, "v": v_opt}
        result_model: Any = {"actor": actor_logits, "q": q_net, "v": v_net}
    else:
        act_low = dataset.action_low if dataset.action_low is not None else -np.ones(dataset.action_dim, dtype=np.float32)
        act_high = dataset.action_high if dataset.action_high is not None else np.ones(dataset.action_dim, dtype=np.float32)
        q_net_twin = TwinQCritic(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_q_twin = TwinQCritic(
            obs_spec,
            dataset.action_dim,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        hard_update(q_net_twin, target_q_twin)
        actor_cont = ContinuousGaussianActor(
            obs_spec,
            dataset.action_dim,
            act_low,
            act_high,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        q_opt = torch.optim.Adam(q_net_twin.parameters(), lr=cfg.lr)
        actor_opt = torch.optim.Adam(actor_cont.parameters(), lr=cfg.lr)

        policy_kind = "gaussian_actor"
        primary_key = "actor"
        policy_spec = {
            "net_class": "ContinuousGaussianActor",
            "init_kwargs": {
                "action_dim": int(dataset.action_dim),
                "action_low": act_low.tolist(),
                "action_high": act_high.tolist(),
                "hidden_dims": list(cfg.hidden_dims),
                "extractor_type": cfg.extractor_type,
                "extractor_kwargs": cfg.extractor_kwargs,
            },
        }
        models_map = {"actor": actor_cont, "q": q_net_twin, "target_q": target_q_twin, "v": v_net}
        opts_map = {"actor": actor_opt, "q": q_opt, "v": v_opt}
        result_model = {"actor": actor_cont, "q": q_net_twin, "v": v_net}

    obs_all, next_obs_all, start_step, learning_curve = prepare_offline_run(
        ctx, dataset, resume_from=cfg.resume_from, models=models_map, optimizers=opts_map, rng=rng
    )

    t0 = time.perf_counter()
    v_losses, q_losses, q_hist = [], [], []
    masks_raw = dataset.extra.get("action_masks")
    masks_all = np.asarray(masks_raw, dtype=np.bool_) if (dataset.is_discrete and masks_raw is not None) else None

    for step in range(start_step + 1, cfg.num_grad_steps + 1):
        idx = rng.integers(0, n_data, size=cfg.batch_size)
        b_obs = obs_to_tensor(slice_obs(obs_all, idx), obs_spec, dev, add_batch_dim=False)
        b_rew = torch.as_tensor(dataset.rewards[idx], device=dev, dtype=torch.float32)
        b_next = obs_to_tensor(slice_obs(next_obs_all, idx), obs_spec, dev, add_batch_dim=False)
        b_term = torch.as_tensor(dataset.terminateds[idx], device=dev, dtype=torch.float32)

        if dataset.is_discrete:
            b_act = torch.as_tensor(dataset.actions[idx], device=dev, dtype=torch.long)
            with torch.no_grad():
                q_t = target_q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            v_pred = v_net(b_obs).squeeze(-1)
            v_loss = _expectile_loss(q_t - v_pred, cfg.expectile)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            with torch.no_grad():
                v_next = v_net(b_next).squeeze(-1)
                q_target = b_rew + cfg.gamma * (1.0 - b_term) * v_next
            q_pred = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(q_pred, q_target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()
            polyak_update(q_net, target_q, cfg.tau)

            with torch.no_grad():
                adv = target_q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1) - v_net(b_obs).squeeze(-1)
                weights = torch.exp(cfg.awr_beta * adv).clamp(max=cfg.max_awr_weight)
            logits_b = actor_logits(b_obs)
            if masks_all is not None:
                b_mask = torch.as_tensor(masks_all[idx], device=dev, dtype=torch.bool).scatter(1, b_act.unsqueeze(1), True)
                logits_b = logits_b.masked_fill(~b_mask, -1e9)
            log_probs = Categorical(logits=logits_b).log_prob(b_act)
            actor_loss = -(weights * log_probs).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            q_hist.append(float(q_pred.mean().item()))
        else:
            b_act = torch.as_tensor(dataset.actions[idx], device=dev, dtype=torch.float32)
            with torch.no_grad():
                tq1, tq2 = target_q_twin(b_obs, b_act)
                q_t = torch.min(tq1, tq2).squeeze(-1)
            v_pred = v_net(b_obs).squeeze(-1)
            v_loss = _expectile_loss(q_t - v_pred, cfg.expectile)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            with torch.no_grad():
                v_next = v_net(b_next).squeeze(-1)
                q_target = b_rew + cfg.gamma * (1.0 - b_term) * v_next
            q1, q2 = q_net_twin(b_obs, b_act)
            q_loss = F.mse_loss(q1.squeeze(-1), q_target) + F.mse_loss(q2.squeeze(-1), q_target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()
            polyak_update(q_net_twin, target_q_twin, cfg.tau)

            # [AI-Prompt Hook]: Advantage-Weighted Regression extracts actor strictly from dataset actions.
            with torch.no_grad():
                tq1_a, tq2_a = target_q_twin(b_obs, b_act)
                adv = torch.min(tq1_a, tq2_a).squeeze(-1) - v_net(b_obs).squeeze(-1)
                weights = torch.exp(cfg.awr_beta * adv).clamp(max=cfg.max_awr_weight)
            dist = actor_cont.get_dist(b_obs)
            log_probs = dist.log_prob(b_act).sum(dim=-1)
            actor_loss = -(weights * log_probs).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            q_hist.append(float(q1.mean().item()))

        v_losses.append(float(v_loss.item()))
        q_losses.append(float(q_loss.item()))
        if step % max(1, cfg.num_grad_steps // 10) == 0:
            learning_curve.append({"step": float(step), "v_loss": v_losses[-1], "q_loss": q_losses[-1]})
            if ctx.ckpt_mgr is not None:
                ctx.ckpt_mgr.log_scalars(step, {"train/v_loss": v_losses[-1], "train/q_loss": q_losses[-1]})

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
            "mean_expectile_v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "mean_q_loss": float(np.mean(q_losses)) if q_losses else 0.0,
            "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
            "max_q_value": float(max(q_hist, key=abs)) if q_hist else 0.0,
            "is_continuous": 0.0 if dataset.is_discrete else 1.0,
        },
        eval_episodes=cfg.eval_episodes,
        seed=cfg.seed,
        rng=rng,
    )
