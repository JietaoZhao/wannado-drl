"""Dueling Double DQN: State-Value & Advantage Decomposition (Wang et al., ICML 2016).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : value_based / A (SOTA discrete architecture before Rainbow)
- Parent        : `double_dqn` (van Hasselt et al., 2016)
- Key Diff      : Splits the network head into a scalar state-value V(s) and action
                  advantages A(s, a), combined as Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)).
                  Useful when many actions yield nearly identical outcomes in a state.
- Extension Hook [AI-Prompt Hook]:
    1. Replace uniform sampling with Prioritized Experience Replay (PER, Schaul 2016).
    2. Combine with distributional return atoms -> Rainbow-lite.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.buffers import ReplayBuffer
from wannado.drl.core.networks import DuelingQNetwork
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="dueling_dqn",
    name="Dueling Double DQN",
    family="value_based",
    paradigm="off_policy",
    tier="A",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("discrete",),
    env_modalities=("discrete_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="double_dqn",
    diff_from_parent="Decomposes Q(s,a) into V(s) + (A(s,a) - mean A(s,.)) alongside Double-DQN targets.",
    summary_plain="Separates 'how good is this situation' from 'which action is better', speeding up multi-action discrete learning.",
    when_to_use="Discrete problems with many actions where individual action choices only matter in critical states.",
    when_not_to_use="Continuous action tasks (`sac`/`td3`) or tiny 2-action problems where standard Double DQN already suffices.",
    hyperparam_intuition={
        "hidden_dim": "Width of shared state trunk and V/A branches.",
        "target_update_freq": "Target sync interval stabilizing dueling advantage separation.",
        "lr": "Adam step size.",
    },
    extension_hooks=(
        "Add Prioritized Experience Replay weights w_i = (N * P(i))^{-beta}",
        "Replace MSE/Huber target with categorical distribution -> `c51`",
    ),
    module_path="wannado.drl.algos.value_based.dueling_dqn_wang2016",
    citation=AlgoCitation(
        key="wang2016dueling",
        title="Dueling Network Architectures for Deep Reinforcement Learning",
        authors="Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas",
        venue="Proceedings of the 33rd International Conference on Machine Learning (ICML), PMLR 48",
        year=2016,
        url="https://proceedings.mlr.press/v48/wangf16.html",
        bibtex=(
            "@inproceedings{wang2016dueling,\n"
            "  title={Dueling Network Architectures for Deep Reinforcement Learning},\n"
            "  author={Wang, Ziyu and Schaul, Tom and Hessel, Matteo and van Hasselt, Hado and Lanctot, Marc and de Freitas, Nando},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={48},\n"
            "  year={2016},\n"
            "  url={https://proceedings.mlr.press/v48/wangf16.html}\n"
            "}"
        ),
        ieee='Z. Wang et al., "Dueling network architectures for deep reinforcement learning," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 48, 2016.',
        apa="Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. ICML (PMLR 48).",
    ),
)


@dataclass
class DuelingDQNConfig:
    total_timesteps: int = 12_000
    learning_starts: int = 400
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 20_000
    batch_size: int = 64
    target_update_freq: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.5
    hidden_dim: int = 128
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_dueling_dqn(
    env_or_factory: Any,
    config: DuelingDQNConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Dueling Double DQN on any Discrete-action environment."""
    cfg = config or DuelingDQNConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)
    if not env.is_discrete:
        raise ValueError("dueling_dqn supports Discrete action spaces only.")

    q_net = DuelingQNetwork(env.obs_dim, env.action_dim, hidden_dim=cfg.hidden_dim).to(dev)
    target_net = DuelingQNetwork(env.obs_dim, env.action_dim, hidden_dim=cfg.hidden_dim).to(dev)
    hard_update(q_net, target_net)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.buffer_size, env.obs_dim, env.action_dim, discrete_action=True, device=dev)
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    t0 = time.perf_counter()
    obs, info = env.reset(seed=cfg.seed)
    ep_ret = 0.0
    ep_idx = 0
    learning_curve: list[dict[str, float]] = []
    td_losses: list[float] = []
    q_means: list[float] = []

    for step in range(1, cfg.total_timesteps + 1):
        decay_steps = max(1.0, cfg.exploration_fraction * cfg.total_timesteps)
        eps = max(cfg.eps_end, cfg.eps_start - (step / decay_steps) * (cfg.eps_start - cfg.eps_end))
        if obs_rms is not None:
            obs_rms.update(obs)
            obs_proc = obs_rms.normalize(obs)
        else:
            obs_proc = obs

        mask = info.get("action_mask")
        if float(rng.random()) < eps:
            valid = np.flatnonzero(mask) if mask is not None else np.arange(env.action_dim)
            act = int(rng.choice(valid if valid.size > 0 else np.arange(env.action_dim)))
        else:
            with torch.no_grad():
                q_vals = q_net(torch.as_tensor(obs_proc, device=dev, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                if mask is not None:
                    m = torch.as_tensor(mask, device=dev, dtype=torch.bool).view_as(q_vals)
                    if bool(m.any()):
                        q_vals = q_vals.masked_fill(~m, -1e9)
                act = int(torch.argmax(q_vals).item())

        next_obs, reward, term, trunc, next_info = env.step(act)
        ep_ret += float(reward)
        next_proc = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs
        replay.add(obs_proc, act, float(reward), next_proc, bool(term))

        if term or trunc:
            ep_idx += 1
            if ep_idx % 5 == 0 or step == cfg.total_timesteps:
                learning_curve.append({"step": float(step), "episode_return": float(ep_ret), "epsilon": float(eps)})
            obs, info = env.reset(seed=cfg.seed + ep_idx)
            ep_ret = 0.0
        else:
            obs, info = next_obs, next_info

        if step >= cfg.learning_starts and replay.size >= cfg.batch_size:
            b_obs, b_act, b_rew, b_next, b_term = replay.sample(cfg.batch_size, rng=rng)
            # [AI-Prompt Hook]: Combine Dueling V+A head with Double-DQN target decoupling.
            with torch.no_grad():
                best_next_act = q_net(b_next).argmax(dim=1, keepdim=True)
                next_q = target_net(b_next).gather(1, best_next_act).squeeze(1)
                td_target = b_rew + cfg.gamma * (1.0 - b_term) * next_q
            q_pred = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_pred, td_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()
            td_losses.append(float(loss.item()))
            q_means.append(float(q_pred.mean().item()))

        if step % cfg.target_update_freq == 0:
            hard_update(q_net, target_net)

    if obs_rms is not None:
        obs_rms.freeze(True)

    def policy_fn(obs_in: np.ndarray, info_in: dict[str, Any] | None = None) -> int:
        o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
        with eval_mode(q_net), torch.no_grad():
            q_vals = q_net(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        mask_in = (info_in or {}).get("action_mask")
        if mask_in is not None:
            m = torch.as_tensor(mask_in, device=dev, dtype=torch.bool).view_as(q_vals)
            if bool(m.any()):
                q_vals = q_vals.masked_fill(~m, -1e9)
        return int(torch.argmax(q_vals).item())

    eval_target = eval_env_or_factory if eval_env_or_factory is not None else env
    eval_stats = evaluate_policy_episodes(eval_target, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.total_timesteps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=q_net,
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_td_loss": float(np.mean(td_losses)) if td_losses else 0.0,
            "mean_q_value": float(np.mean(q_means)) if q_means else 0.0,
            "max_q_value": float(np.max(q_means)) if q_means else 0.0,
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
    )
