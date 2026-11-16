"""DDPG: Deep Deterministic Policy Gradient (Lillicrap et al., ICLR 2016).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : actor_critic_off / B (Foundational Continuous Off-Policy Actor-Critic)
- Parent        : `dqn` (Mnih et al., 2015)
- Key Diff      : Extends off-policy replay + target networks from discrete Q-learning
                  to continuous action vectors by learning a deterministic actor mu(s)
                  alongside a differentiable critic Q(s, a) with soft Polyak target updates.
- Extension Hook [AI-Prompt Hook]:
    1. Add Twin Critics, delayed actor update, and target action noise -> `td3`.
    2. Switch deterministic actor to maximum-entropy stochastic policy -> `sac`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.buffers import ReplayBuffer
from wannado.drl.core.networks import DeterministicActor, build_mlp
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="ddpg",
    name="Deep Deterministic Policy Gradient (DDPG)",
    family="actor_critic_off",
    paradigm="off_policy",
    tier="B",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("continuous",),
    env_modalities=("continuous_control", "custom_sim"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="dqn",
    diff_from_parent="Replaces discrete argmax Q(s, a) with a continuous deterministic actor mu(s) trained via dQ/da.",
    summary_plain="First deep off-policy actor-critic for continuous control; direct ancestor of TD3 and SAC.",
    when_to_use="Continuous control baselines and comparing improvements brought by TD3/SAC.",
    when_not_to_use="Sensitive continuous tasks prone to Q-overestimation (prefer `td3` or `sac`) or discrete actions (`dqn`/`ppo`).",
    hyperparam_intuition={
        "tau": "Polyak soft target blending speed (0.005 keeps target critics smooth).",
        "exploration_noise": "Gaussian action noise scale added during exploration.",
        "learning_starts": "Random warmup transitions collected before gradient updates begin.",
    },
    extension_hooks=(
        "Add twin critics + target noise + policy delay -> `td3`",
        "Replace deterministic policy with entropy-regularized Gaussian -> `sac`",
    ),
    module_path="wannado.drl.algos.actor_critic_off.ddpg_lillicrap2016",
    citation=AlgoCitation(
        key="lillicrap2016continuous",
        title="Continuous control with deep reinforcement learning",
        authors="Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra",
        venue="International Conference on Learning Representations (ICLR)",
        year=2016,
        url="https://arxiv.org/abs/1509.02971",
        bibtex=(
            "@inproceedings{lillicrap2016continuous,\n"
            "  title={Continuous control with deep reinforcement learning},\n"
            "  author={Lillicrap, Timothy P. and Hunt, Jonathan J. and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2016},\n"
            "  url={https://arxiv.org/abs/1509.02971}\n"
            "}"
        ),
        ieee='T. P. Lillicrap et al., "Continuous control with deep reinforcement learning," in Proc. Int. Conf. Learn. Represent. (ICLR), 2016.',
        apa="Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. ICLR.",
    ),
)


@dataclass
class DDPGConfig:
    total_timesteps: int = 8_000
    learning_starts: int = 500
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 50_000
    batch_size: int = 64
    exploration_noise: float = 0.15
    hidden_dims: tuple[int, ...] = (128, 128)
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_ddpg(
    env_or_factory: Any,
    config: DDPGConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train DDPG on continuous Box-action environments."""
    cfg = config or DDPGConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)
    if env.is_discrete or env.action_low is None or env.action_high is None:
        raise ValueError("ddpg requires bounded continuous Box action spaces.")

    actor = DeterministicActor(env.obs_dim, env.action_dim, env.action_low, env.action_high, cfg.hidden_dims).to(dev)
    target_actor = DeterministicActor(env.obs_dim, env.action_dim, env.action_low, env.action_high, cfg.hidden_dims).to(dev)
    hard_update(actor, target_actor)

    critic = build_mlp(env.obs_dim + env.action_dim, cfg.hidden_dims, 1, activation=torch.nn.ReLU).to(dev)
    target_critic = build_mlp(env.obs_dim + env.action_dim, cfg.hidden_dims, 1, activation=torch.nn.ReLU).to(dev)
    hard_update(critic, target_critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    replay = ReplayBuffer(cfg.buffer_size, env.obs_dim, env.action_dim, discrete_action=False, device=dev)
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    act_scale = (env.action_high - env.action_low) / 2.0
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_idx = 0.0, 0
    learning_curve: list[dict[str, float]] = []
    q_hist: list[float] = []

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
            noise = rng.normal(0.0, cfg.exploration_noise * act_scale, size=env.action_dim).astype(np.float32)
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
            b_obs, b_act, b_rew, b_next, b_term = replay.sample(cfg.batch_size, rng=rng)
            # [AI-Prompt Hook]: Single Q-target; replace with Twin Clipped Double-Q in `td3`.
            with torch.no_grad():
                next_a = target_actor(b_next)
                next_q = target_critic(torch.cat([b_next, next_a], dim=-1)).squeeze(-1)
                td_target = b_rew + cfg.gamma * (1.0 - b_term) * next_q

            q_pred = critic(torch.cat([b_obs, b_act], dim=-1)).squeeze(-1)
            critic_loss = F.mse_loss(q_pred, td_target)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            q_hist.append(float(q_pred.mean().item()))

            actor_loss = -critic(torch.cat([b_obs, actor(b_obs)], dim=-1)).mean()
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
        train_diagnostics={"mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0},
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
    )
