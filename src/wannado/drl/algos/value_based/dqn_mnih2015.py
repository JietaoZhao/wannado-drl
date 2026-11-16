"""DQN: Deep Q-Network (Mnih et al., Nature 2015).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : value_based / S (Landmark deep off-policy value baseline)
- Parent        : `q_learning` (Watkins & Dayan, 1992)
- Key Leap      : Replaces the discrete Q-table with a neural Q-function, stabilizes
                  correlated trajectories via Experience Replay, and freezes TD targets
                  periodically using a lagging Target Network.
- Extension Hook [AI-Prompt Hook]:
    1. Decouple argmax action choice (online net) from value estimate (target net) -> `double_dqn`.
    2. Split Q(s, a) into state-value V(s) and advantage A(s, a) -> `dueling_dqn`.
    3. Learn full return distribution atoms instead of scalar expectation -> `c51`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.buffers import ReplayBuffer
from wannado.drl.core.networks import QNetwork
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="dqn",
    name="Deep Q-Network (DQN)",
    family="value_based",
    paradigm="off_policy",
    tier="S",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("discrete",),
    env_modalities=("discrete_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="q_learning",
    diff_from_parent="Replaces Q-table with neural network, circular Experience Replay, and target Q-network.",
    summary_plain="Learns Q-values via neural network + replay memory; standard starting point for discrete deep RL.",
    when_to_use="Discrete action spaces with continuous/high-dimensional state vectors where replay efficiency helps.",
    when_not_to_use="Continuous Box actions (use `sac` or `td3`) or large action branching (consider `ppo`).",
    hyperparam_intuition={
        "lr": "Adam step size for Q-network updates.",
        "buffer_size": "How many past transitions stay in replay memory.",
        "target_update_freq": "How often target network weights sync to prevent moving-target oscillations.",
    },
    extension_hooks=(
        "Decouple online argmax from target evaluation -> `double_dqn`",
        "Decompose Q into V(s) + centered A(s, a) -> `dueling_dqn`",
        "Add Prioritized Experience Replay weights from |TD error|",
    ),
    module_path="wannado.drl.algos.value_based.dqn_mnih2015",
    citation=AlgoCitation(
        key="mnih2015human",
        title="Human-level control through deep reinforcement learning",
        authors="Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al.",
        venue="Nature, vol. 518, no. 7540, pp. 529-533",
        year=2015,
        url="https://doi.org/10.1038/nature14236",
        bibtex=(
            "@article{mnih2015human,\n"
            "  title={Human-level control through deep reinforcement learning},\n"
            "  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and others},\n"
            "  journal={Nature},\n"
            "  volume={518},\n"
            "  number={7540},\n"
            "  pages={529--533},\n"
            "  year={2015},\n"
            "  doi={10.1038/nature14236}\n"
            "}"
        ),
        ieee='V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, pp. 529-533, 2015.',
        apa="Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.",
    ),
)


@dataclass
class DQNConfig:
    total_timesteps: int = 12_000
    learning_starts: int = 400
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 20_000
    batch_size: int = 64
    target_update_freq: int = 200
    train_freq: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.5
    hidden_dims: tuple[int, ...] = (128, 128)
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def _masked_argmax(q_vals: torch.Tensor, mask: np.ndarray | None) -> int:
    if mask is not None:
        m = torch.as_tensor(mask, device=q_vals.device, dtype=torch.bool).view_as(q_vals)
        if bool(m.any()):
            q_vals = q_vals.masked_fill(~m, -1e9)
    return int(torch.argmax(q_vals).item())


def train_dqn(
    env_or_factory: Any,
    config: DQNConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Deep Q-Network (DQN) on any Discrete-action environment."""
    cfg = config or DQNConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)
    if not env.is_discrete:
        raise ValueError("dqn supports Discrete action spaces only.")

    q_net = QNetwork(env.obs_dim, env.action_dim, hidden_dims=cfg.hidden_dims).to(dev)
    target_net = QNetwork(env.obs_dim, env.action_dim, hidden_dims=cfg.hidden_dims).to(dev)
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
                act = _masked_argmax(q_vals, mask)

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

        if step >= cfg.learning_starts and step % cfg.train_freq == 0 and replay.size >= cfg.batch_size:
            b_obs, b_act, b_rew, b_next, b_term = replay.sample(cfg.batch_size, rng=rng)
            # [AI-Prompt Hook]: Standard Bellman target; swap max(next_q) for Double DQN or C51 projection.
            with torch.no_grad():
                next_q_max = target_net(b_next).max(dim=1).values
                td_target = b_rew + cfg.gamma * (1.0 - b_term) * next_q_max
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
        return _masked_argmax(q_vals, (info_in or {}).get("action_mask"))

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
