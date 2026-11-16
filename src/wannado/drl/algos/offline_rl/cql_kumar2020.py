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

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.networks import QNetwork, SquashedGaussianActor, TwinQCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import OfflineTransitionDataset, RunningMeanStd
from wannado.drl.mdp.baselines import evaluate_policy_episodes

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
    supports_action_mask=False,
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
    normalize_obs: bool = True
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


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

    obs_rms = RunningMeanStd((dataset.obs_dim,)) if cfg.normalize_obs else None
    if obs_rms is not None:
        obs_rms.update(dataset.obs)
        obs_rms.freeze(True)
        obs_all = obs_rms.normalize(dataset.obs)
        next_obs_all = obs_rms.normalize(dataset.next_obs)
    else:
        obs_all = dataset.obs
        next_obs_all = dataset.next_obs

    t0 = time.perf_counter()
    learning_curve: list[dict[str, float]] = []
    q_hist, cql_gap_hist = [], []

    if dataset.is_discrete:
        q_net = QNetwork(dataset.obs_dim, dataset.action_dim, cfg.hidden_dims).to(dev)
        target_net = QNetwork(dataset.obs_dim, dataset.action_dim, cfg.hidden_dims).to(dev)
        hard_update(q_net, target_net)
        opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

        for step in range(1, cfg.num_grad_steps + 1):
            idx = rng.integers(0, n_data, size=cfg.batch_size)
            b_obs = torch.as_tensor(obs_all[idx], device=dev, dtype=torch.float32)
            b_act = torch.as_tensor(dataset.actions[idx], device=dev, dtype=torch.long)
            b_rew = torch.as_tensor(dataset.rewards[idx], device=dev, dtype=torch.float32)
            b_next = torch.as_tensor(next_obs_all[idx], device=dev, dtype=torch.float32)
            b_term = torch.as_tensor(dataset.terminateds[idx], device=dev, dtype=torch.float32)

            with torch.no_grad():
                td_target = b_rew + cfg.gamma * (1.0 - b_term) * target_net(b_next).max(dim=1).values
            q_all = q_net(b_obs)
            q_data = q_all.gather(1, b_act.unsqueeze(1)).squeeze(1)
            td_loss = F.mse_loss(q_data, td_target)
            # [AI-Prompt Hook]: Discrete CQL log-sum-exp OOD penalty.
            cql_pen = (torch.logsumexp(q_all, dim=1) - q_data).mean()
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

        model_obj: Any = q_net

        def policy_fn(obs_in: np.ndarray, _info: dict[str, Any] | None = None) -> int:
            o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
            with eval_mode(q_net), torch.no_grad():
                return int(torch.argmax(q_net(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0))).item())

    else:
        act_low = dataset.action_low if dataset.action_low is not None else -np.ones(dataset.action_dim, dtype=np.float32)
        act_high = dataset.action_high if dataset.action_high is not None else np.ones(dataset.action_dim, dtype=np.float32)
        actor = SquashedGaussianActor(dataset.obs_dim, dataset.action_dim, act_low, act_high, cfg.hidden_dims).to(dev)
        critic = TwinQCritic(dataset.obs_dim, dataset.action_dim, cfg.hidden_dims).to(dev)
        target_critic = TwinQCritic(dataset.obs_dim, dataset.action_dim, cfg.hidden_dims).to(dev)
        hard_update(critic, target_critic)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
        low_t = torch.as_tensor(act_low, device=dev, dtype=torch.float32)
        high_t = torch.as_tensor(act_high, device=dev, dtype=torch.float32)

        for step in range(1, cfg.num_grad_steps + 1):
            idx = rng.integers(0, n_data, size=cfg.batch_size)
            b_obs = torch.as_tensor(obs_all[idx], device=dev, dtype=torch.float32)
            b_act = torch.as_tensor(dataset.actions[idx], device=dev, dtype=torch.float32)
            b_rew = torch.as_tensor(dataset.rewards[idx], device=dev, dtype=torch.float32)
            b_next = torch.as_tensor(next_obs_all[idx], device=dev, dtype=torch.float32)
            b_term = torch.as_tensor(dataset.terminateds[idx], device=dev, dtype=torch.float32)

            with torch.no_grad():
                na, nlp = actor.get_action(b_next)
                tq1, tq2 = target_critic(b_next, na)
                td_target = b_rew.unsqueeze(-1) + cfg.gamma * (1.0 - b_term.unsqueeze(-1)) * (torch.min(tq1, tq2) - 0.1 * nlp)

            q1, q2 = critic(b_obs, b_act)
            td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

            # Sample random & current-policy candidate actions for continuous CQL logsumexp
            rep_obs = b_obs.unsqueeze(1).repeat(1, cfg.num_random_actions, 1).view(-1, dataset.obs_dim)
            rand_a = low_t + torch.rand(rep_obs.shape[0], dataset.action_dim, device=dev) * (high_t - low_t)
            pol_a, _ = actor.get_action(rep_obs)
            cand_obs = torch.cat([rep_obs, rep_obs], dim=0)
            cand_a = torch.cat([rand_a, pol_a.detach()], dim=0)
            cq1, cq2 = critic(cand_obs, cand_a)
            cq1 = cq1.view(cfg.batch_size, 2 * cfg.num_random_actions)
            cq2 = cq2.view(cfg.batch_size, 2 * cfg.num_random_actions)
            cql_pen = (torch.logsumexp(cq1, dim=1) - q1.squeeze(-1)).mean() + (
                torch.logsumexp(cq2, dim=1) - q2.squeeze(-1)
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

        model_obj = {"actor": actor, "critic": critic}

        def policy_fn(obs_in: np.ndarray, _info: dict[str, Any] | None = None) -> np.ndarray:
            o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
            with eval_mode(actor), torch.no_grad():
                a_t, _ = actor.get_action(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0), deterministic=True)
                return a_t.squeeze(0).cpu().numpy()

    eval_stats = (
        evaluate_policy_episodes(eval_env_or_factory, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)
        if eval_env_or_factory is not None
        else None
    )

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.num_grad_steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=model_obj,
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
            "mean_cql_ood_gap": float(np.mean(cql_gap_hist)) if cql_gap_hist else 0.0,
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True,
    )
