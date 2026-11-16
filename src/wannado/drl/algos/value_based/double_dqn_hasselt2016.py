"""Double DQN: Eliminating Overestimation Bias (van Hasselt, Guez, & Silver, AAAI 2016).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : value_based / A (Fixes DQN maximization bias)
- Parent        : `dqn` (Mnih et al., 2015)
- Key Diff      : Selects the greedy next action with the online Q-network, then
                  evaluates that chosen action using the frozen target Q-network:
                  target = r + gamma * Q_target(s', argmax_{a'} Q_online(s', a')).
- Extension Hook [AI-Prompt Hook]:
    1. Replace QNetwork architecture with V(s) + centered A(s, a) -> `dueling_dqn`.
    2. Use Polyak soft target updates (tau) instead of hard periodic copies.
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
from wannado.drl.core.provenance import resolve_run_optimizations
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="double_dqn",
    name="Double Deep Q-Network (Double DQN)",
    family="value_based",
    paradigm="off_policy",
    tier="A",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("discrete",),
    env_modalities=("discrete_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="dqn",
    diff_from_parent="Decouples next-action argmax selection (online net) from Q-value evaluation (target net).",
    summary_plain="Fixes DQN's over-optimistic Q-value inflation with a one-line target decoupling change.",
    when_to_use="Drop-in upgrade over vanilla DQN whenever discrete Q-values overshoot or fluctuate.",
    when_not_to_use="Continuous action tasks (use `td3` or `sac`) or tasks needing state-value separation (`dueling_dqn`).",
    hyperparam_intuition={
        "lr": "Optimizer learning rate for Q-network updates.",
        "target_update_freq": "Target sync interval; larger values stabilize Bellman targets.",
        "gamma": "Discount horizon weighting future vs immediate rewards.",
    },
    extension_hooks=(
        "Swap QNetwork to DuelingQNetwork -> `dueling_dqn`",
        "Add n-step TD returns inside replay buffer",
    ),
    module_path="wannado.drl.algos.value_based.double_dqn_hasselt2016",
    citation=AlgoCitation(
        key="vanhasselt2016deep",
        title="Deep Reinforcement Learning with Double Q-learning",
        authors="Hado van Hasselt, Arthur Guez, and David Silver",
        venue="Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)",
        year=2016,
        url="https://arxiv.org/abs/1509.06461",
        bibtex=(
            "@inproceedings{vanhasselt2016deep,\n"
            "  title={Deep Reinforcement Learning with Double Q-learning},\n"
            "  author={van Hasselt, Hado and Guez, Arthur and Silver, David},\n"
            "  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},\n"
            "  year={2016},\n"
            "  url={https://arxiv.org/abs/1509.06461}\n"
            "}"
        ),
        ieee='H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in Proc. AAAI Conf. Artif. Intell., 2016.',
        apa="van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. Proceedings of the AAAI Conference on Artificial Intelligence.",
    ),
)


@dataclass
class DoubleDQNConfig:
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
    hidden_dims: tuple[int, ...] = (128, 128)
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_double_dqn(
    env_or_factory: Any,
    config: DoubleDQNConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Double DQN on any Discrete-action environment."""
    cfg = config or DoubleDQNConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)
    if not env.is_discrete:
        raise ValueError("double_dqn supports Discrete action spaces only.")

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
    saw_action_mask = False
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
        if mask is not None:
            saw_action_mask = True
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
            # [AI-Prompt Hook]: Double Q-learning target decouples action choice (online) from valuation (target).
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
    cfg_dict = asdict(cfg)
    opts = resolve_run_optimizations(META.id, config=cfg_dict, saw_action_mask=saw_action_mask)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.total_timesteps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=q_net,
        config=cfg_dict,
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_td_loss": float(np.mean(td_losses)) if td_losses else 0.0,
            "mean_q_value": float(np.mean(q_means)) if q_means else 0.0,
            "max_q_value": float(np.max(q_means)) if q_means else 0.0,
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
        optimizations_used=[o.id for o in opts],
    )
