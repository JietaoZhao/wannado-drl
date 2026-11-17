"""TD3: Twin Delayed Deep Deterministic Policy Gradient (Fujimoto, van Hoof, & Meger, ICML 2018).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : actor_critic_off / S (Continuous Deterministic SOTA Benchmark)
- Parent        : `ddpg` (Lillicrap et al., 2016)
- Key Diff      : Fixes DDPG's brittle Q-value explosion via three stabilizing upgrades:
                  1. Clipped Double-Q Learning (min(Q1, Q2) critic target)
                  2. Delayed Policy Updates (update actor every `policy_delay` critic steps)
                  3. Target Policy Smoothing (jitter target actions inside `noise_clip`)
- Extension Hook [AI-Prompt Hook]:
    1. Replace deterministic actor + fixed noise with stochastic max-entropy policy -> `sac`.
    2. Use LayerNorm twin critics + 50/50 demo-online sampling -> `hil_serl`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.buffers import ReplayBuffer
from wannado.drl.core.networks import DeterministicActor, TwinQCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="td3",
    name="Twin Delayed DDPG (TD3)",
    family="actor_critic_off",
    paradigm="off_policy",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("continuous",),
    env_modalities=("continuous_control", "custom_sim"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="ddpg",
    diff_from_parent="Adds Twin Clipped Double-Q critics, delayed actor/target updates, and target action smoothing.",
    summary_plain="Gold-standard deterministic continuous control algorithm; far more stable than DDPG.",
    when_to_use="Sample-efficient continuous control benchmarks where deterministic smooth control is preferred.",
    when_not_to_use="Discrete actions (`dqn`/`ppo`) or multimodal exploration tasks needing entropy maximization (`sac`).",
    hyperparam_intuition={
        "policy_delay": "Number of critic updates per actor update (default 2 lets Q-values settle first).",
        "policy_noise": "Smoothing jitter added to target actions to prevent narrow Q-function spikes.",
        "tau": "Polyak soft update rate for target networks.",
    },
    extension_hooks=(
        "Swap deterministic actor for entropy-tuned squashed Gaussian -> `sac`",
        "Add TD3+BC behavior cloning regularizer for offline continuous control",
    ),
    module_path="wannado.drl.algos.actor_critic_off.td3_fujimoto2018",
    citation=AlgoCitation(
        key="fujimoto2018addressing",
        title="Addressing Function Approximation Error in Actor-Critic Methods",
        authors="Scott Fujimoto, Herke van Hoof, and David Meger",
        venue="Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80",
        year=2018,
        url="https://proceedings.mlr.press/v80/fujimoto18a.html",
        bibtex=(
            "@inproceedings{fujimoto2018addressing,\n"
            "  title={Addressing Function Approximation Error in Actor-Critic Methods},\n"
            "  author={Fujimoto, Scott and van Hoof, Herke and Meger, David},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={80},\n"
            "  year={2018},\n"
            "  url={https://proceedings.mlr.press/v80/fujimoto18a.html}\n"
            "}"
        ),
        ieee='S. Fujimoto, H. van Hoof, and D. Meger, "Addressing function approximation error in actor-critic methods," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018.',
        apa="Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. ICML (PMLR 80).",
    ),
)


@dataclass
class TD3Config:
    total_timesteps: int = 8_000
    learning_starts: int = 500
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.15
    buffer_size: int = 50_000
    batch_size: int = 64
    hidden_dims: tuple[int, ...] = (128, 128)
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_td3(
    env_or_factory: Any,
    config: TD3Config | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Twin Delayed DDPG (TD3) on bounded continuous Box action spaces."""
    cfg = config or TD3Config()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)
    if env.is_discrete or env.action_low is None or env.action_high is None:
        raise ValueError("td3 requires bounded continuous Box action spaces.")

    actor = DeterministicActor(env.obs_dim, env.action_dim, env.action_low, env.action_high, cfg.hidden_dims).to(dev)
    target_actor = DeterministicActor(env.obs_dim, env.action_dim, env.action_low, env.action_high, cfg.hidden_dims).to(dev)
    hard_update(actor, target_actor)

    critic = TwinQCritic(env.obs_dim, env.action_dim, cfg.hidden_dims).to(dev)
    target_critic = TwinQCritic(env.obs_dim, env.action_dim, cfg.hidden_dims).to(dev)
    hard_update(critic, target_critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    replay = ReplayBuffer(cfg.buffer_size, env.obs_dim, env.action_dim, discrete_action=False, device=dev)
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    act_low_t = torch.as_tensor(env.action_low, device=dev, dtype=torch.float32)
    act_high_t = torch.as_tensor(env.action_high, device=dev, dtype=torch.float32)
    act_scale_t = (act_high_t - act_low_t) / 2.0
    act_scale_np = (env.action_high - env.action_low) / 2.0

    t0 = time.perf_counter()
    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_idx = 0.0, 0
    learning_curve: list[dict[str, float]] = []
    q_hist: list[float] = []
    update_step = 0

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
                raw = actor(torch.as_tensor(obs_proc, device=dev, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()
            noise = rng.normal(0.0, cfg.exploration_noise * act_scale_np, size=env.action_dim).astype(np.float32)
            act = np.clip(raw + noise, env.action_low, env.action_high).astype(np.float32)

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
            update_step += 1
            b_obs, b_act, b_rew, b_next, b_term = replay.sample(cfg.batch_size, rng=rng)
            # [AI-Prompt Hook]: TD3 Clipped Double-Q + target policy smoothing.
            with torch.no_grad():
                raw_noise = torch.randn_like(b_act) * (cfg.policy_noise * act_scale_t)
                clipped_noise = torch.clamp(raw_noise, -cfg.noise_clip * act_scale_t, cfg.noise_clip * act_scale_t)
                next_a = torch.clamp(target_actor(b_next) + clipped_noise, act_low_t, act_high_t)
                tq1, tq2 = target_critic(b_next, next_a)
                min_tq = torch.min(tq1, tq2).squeeze(-1)
                td_target = b_rew + cfg.gamma * (1.0 - b_term) * min_tq

            q1, q2 = critic(b_obs, b_act)
            critic_loss = F.mse_loss(q1.squeeze(-1), td_target) + F.mse_loss(q2.squeeze(-1), td_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            q_hist.append(float(q1.mean().item()))

            if update_step % cfg.policy_delay == 0:
                actor_loss = -critic.q1_forward(b_obs, actor(b_obs)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                polyak_update(actor, target_actor, cfg.tau)
                polyak_update(critic, target_critic, cfg.tau)

    if obs_rms is not None:
        obs_rms.freeze(True)

    def policy_fn(obs_in: np.ndarray, _info: dict[str, Any] | None = None) -> np.ndarray:
        o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
        with eval_mode(actor), torch.no_grad():
            return actor(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()

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
            "max_q_value": float(max(q_hist, key=abs)) if q_hist else 0.0,
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
    )
