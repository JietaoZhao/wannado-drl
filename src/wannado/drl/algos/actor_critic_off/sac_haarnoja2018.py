"""SAC: Soft Actor-Critic with Automatic Entropy Tuning (Haarnoja et al., ICML 2018).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : actor_critic_off / S (Sample-Efficient Continuous Control Default)
- Parent        : `td3` (Fujimoto et al., 2018)
- Key Diff      : Replaces TD3's deterministic actor + manual Gaussian exploration noise
                  with a squashed stochastic Gaussian policy trained under maximum-entropy
                  RL, automatically tuning temperature alpha toward target entropy -dim(A).
- Extension Hook [AI-Prompt Hook]:
    1. Add LayerNorm Q-critics + 50/50 demo/intervention replay -> `hil_serl`.
    2. Penalize out-of-distribution actions in Q-loss for static datasets -> `cql`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.buffers import ReplayBuffer
from wannado.drl.core.networks import SquashedGaussianActor, TwinQCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="sac",
    name="Soft Actor-Critic (SAC)",
    family="actor_critic_off",
    paradigm="off_policy",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("continuous",),
    env_modalities=("continuous_control", "custom_sim"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="td3",
    diff_from_parent="Replaces deterministic actor + fixed exploration noise with maximum-entropy squashed Gaussian policy and auto-tuned alpha.",
    summary_plain="Go-to sample-efficient algorithm for continuous control; balances reward and automatic exploration curiosity.",
    when_to_use="Continuous Box actions when simulator steps are costly and you want strong sample efficiency + smooth exploration.",
    when_not_to_use="Discrete actions (use `ppo` or `dqn`) or zero-interaction static logs (use `cql` / `iql_offline`).",
    hyperparam_intuition={
        "init_alpha": "Starting entropy temperature; automatically tuned when autotune_alpha=True.",
        "tau": "Target Q-critic Polyak smoothing factor.",
        "learning_starts": "Initial random steps seeding replay buffer.",
    },
    extension_hooks=(
        "Add LayerNorm in critics + 50/50 demo/intervention replay -> `hil_serl`",
        "Add conservative log-sum-exp Q penalty on static data -> `cql`",
    ),
    module_path="wannado.drl.algos.actor_critic_off.sac_haarnoja2018",
    citation=AlgoCitation(
        key="haarnoja2018soft",
        title="Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
        authors="Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine",
        venue="Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80",
        year=2018,
        url="https://proceedings.mlr.press/v80/haarnoja18a.html",
        bibtex=(
            "@inproceedings{haarnoja2018soft,\n"
            "  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},\n"
            "  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={80},\n"
            "  year={2018},\n"
            "  url={https://proceedings.mlr.press/v80/haarnoja18a.html}\n"
            "}"
        ),
        ieee='T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018.',
        apa="Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. ICML (PMLR 80).",
    ),
)


@dataclass
class SACConfig:
    total_timesteps: int = 8_000
    learning_starts: int = 400
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    autotune_alpha: bool = True
    buffer_size: int = 50_000
    batch_size: int = 64
    hidden_dims: tuple[int, ...] = (128, 128)
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_sac(
    env_or_factory: Any,
    config: SACConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Soft Actor-Critic (SAC) on bounded continuous Box action spaces."""
    cfg = config or SACConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)
    if env.is_discrete or env.action_low is None or env.action_high is None:
        raise ValueError("sac requires bounded continuous Box action spaces.")

    actor = SquashedGaussianActor(env.obs_dim, env.action_dim, env.action_low, env.action_high, cfg.hidden_dims).to(dev)
    critic = TwinQCritic(env.obs_dim, env.action_dim, cfg.hidden_dims).to(dev)
    target_critic = TwinQCritic(env.obs_dim, env.action_dim, cfg.hidden_dims).to(dev)
    hard_update(critic, target_critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), device=dev, requires_grad=cfg.autotune_alpha)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None
    target_entropy = -float(env.action_dim)

    replay = ReplayBuffer(cfg.buffer_size, env.obs_dim, env.action_dim, discrete_action=False, device=dev)
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    t0 = time.perf_counter()
    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_idx = 0.0, 0
    learning_curve: list[dict[str, float]] = []
    q_hist, ent_hist = [], []

    for step in range(1, cfg.total_timesteps + 1):
        if obs_rms is not None:
            obs_rms.update(obs)
            obs_proc = obs_rms.normalize(obs)
        else:
            obs_proc = obs

        if step < cfg.learning_starts:
            act = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        else:
            with torch.no_grad():
                a_t, _ = actor.get_action(torch.as_tensor(obs_proc, device=dev, dtype=torch.float32).unsqueeze(0))
                act = a_t.squeeze(0).cpu().numpy()

        next_obs, reward, term, trunc, _ = env.step(act)
        ep_ret += float(reward)
        next_proc = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs
        replay.add(obs_proc, act, float(reward), next_proc, bool(term))

        if term or trunc:
            ep_idx += 1
            if ep_idx % 5 == 0 or step == cfg.total_timesteps:
                learning_curve.append({"step": float(step), "episode_return": float(ep_ret)})
            obs, _ = env.reset(seed=cfg.seed + ep_idx)
            ep_ret = 0.0
        else:
            obs = next_obs

        if step >= cfg.learning_starts and replay.size >= cfg.batch_size:
            b_obs, b_act, b_rew, b_next, b_term = replay.sample(cfg.batch_size, rng=rng)
            alpha = log_alpha.exp().detach()

            # [AI-Prompt Hook]: Soft Bellman target subtracting alpha * log pi(a'|s').
            with torch.no_grad():
                next_a, next_lp = actor.get_action(b_next)
                tq1, tq2 = target_critic(b_next, next_a)
                min_tq = torch.min(tq1, tq2) - alpha * next_lp
                td_target = b_rew.unsqueeze(-1) + cfg.gamma * (1.0 - b_term.unsqueeze(-1)) * min_tq

            q1, q2 = critic(b_obs, b_act)
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            q_hist.append(float(q1.mean().item()))

            pi_a, log_pi = actor.get_action(b_obs)
            q1_pi, q2_pi = critic(b_obs, pi_a)
            actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            ent_hist.append(float(-log_pi.mean().item()))

            if cfg.autotune_alpha and alpha_opt is not None:
                alpha_loss = (-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

            polyak_update(critic, target_critic, cfg.tau)

    if obs_rms is not None:
        obs_rms.freeze(True)

    def policy_fn(obs_in: np.ndarray, _info: dict[str, Any] | None = None) -> np.ndarray:
        o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
        with eval_mode(actor), torch.no_grad():
            a_t, _ = actor.get_action(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0), deterministic=True)
            return a_t.squeeze(0).cpu().numpy()

    eval_target = eval_env_or_factory if eval_env_or_factory is not None else env
    eval_stats = evaluate_policy_episodes(eval_target, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.total_timesteps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model={"actor": actor, "critic": critic},
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
            "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
            "final_alpha": float(log_alpha.exp().item()),
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
    )
