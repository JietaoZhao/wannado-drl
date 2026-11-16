"""Tabular Q-Learning: Off-Policy Temporal-Difference Control (Watkins & Dayan, 1992).

===============================================================================
[Academic Provenance & Lineage]
- Tier          : B (Foundational Off-Policy Value-Based Origin)
- Family        : tabular_classic -> value_based ancestor
- Reference     : Watkins, C. J. C. H., & Dayan, P. (1992).
                  "Q-learning." Machine Learning, 8(3-4), 279-292.
- URL           : https://doi.org/10.1007/BF00992698
- Plain Intuition:
    While exploring randomly (like wandering around a maze), Q-Learning updates its
    scorecard assuming that at the NEXT state it will play its absolute best move.
    That decoupling between "how you explore" and "what you evaluate" is what makes
    modern off-policy RL (DQN, SAC, Offline RL) possible!
- Diff vs. SARSA (1994):
    Swaps on-policy target Q(s', a') for greedy Bellman optimality target max_{a'} Q(s', a'),
    allowing off-policy learning directly toward the optimal policy.
- Extension Directions [AI-Prompt Hook]:
    1. Replace Q-table with Neural Net + ReplayBuffer + Target Net -> `dqn`.
    2. Use two independent Q-tables to decouple argmax & value eval -> Double Q-Learning.
===============================================================================
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import discretize_observation, set_global_seed
from wannado.drl.core.wrappers import ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="q_learning",
    name="Tabular Q-Learning",
    family="tabular_classic",
    paradigm="tabular_off_policy",
    tier="B",
    on_policy=False,
    sample_efficiency="medium",
    action_spaces=("discrete",),
    env_modalities=("tabular", "discrete_control"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="sarsa",
    diff_from_parent="Switches SARSA's on-policy target Q(s', a') to off-policy greedy Bellman max_{a'} Q(s', a').",
    summary_plain="Classic off-policy Q-table algorithm; foundational ancestor of DQN, Double DQN, QMIX, and CQL.",
    when_to_use="Small discrete gridworlds, teaching value-based RL, or rapid tabular sanity baselines.",
    when_not_to_use="Large/continuous state spaces (discretization suffers curse of dimensionality; switch to `dqn` or `ppo`).",
    hyperparam_intuition={
        "alpha": "Step size updating Q-table entries from each transition.",
        "gamma": "Discount factor determining planning horizon.",
        "eps_start": "Exploratory randomness before converging to greedy action selection.",
    },
    extension_hooks=(
        "Replace Q-table lookup with neural network + ReplayBuffer -> `dqn`",
        "Maintain Q_A / Q_B tables to avoid maximization bias -> Double Q-Learning",
    ),
    module_path="wannado.drl.algos.tabular_classic.q_learning_watkins1992",
    citation=AlgoCitation(
        key="watkins1992qlearning",
        title="Q-learning",
        authors="Christopher J. C. H. Watkins and Peter Dayan",
        venue="Machine Learning",
        year=1992,
        url="https://doi.org/10.1007/BF00992698",
        bibtex=(
            "@article{watkins1992qlearning,\n"
            "  title={Q-learning},\n"
            "  author={Watkins, Christopher J. C. H. and Dayan, Peter},\n"
            "  journal={Machine Learning},\n"
            "  volume={8},\n"
            "  number={3-4},\n"
            "  pages={279--292},\n"
            "  year={1992},\n"
            "  publisher={Springer},\n"
            "  doi={10.1007/BF00992698}\n"
            "}"
        ),
        ieee='C. J. C. H. Watkins and P. Dayan, "Q-learning," Machine Learning, vol. 8, no. 3-4, pp. 279-292, 1992.',
        apa="Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8(3-4), 279-292.",
    ),
)


@dataclass
class QLearningConfig:
    total_timesteps: int = 15_000
    alpha: float = 0.2
    gamma: float = 0.99
    eps_start: float = 0.5
    eps_end: float = 0.02
    bins: int = 8
    eval_episodes: int = 10
    seed: int = 42


def train_q_learning(
    env_or_factory: Any,
    config: QLearningConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train tabular off-policy Q-Learning with action-mask support."""
    cfg = config or QLearningConfig()
    rng = set_global_seed(cfg.seed)
    env = ensure_gym_env(env_or_factory)
    if not env.is_discrete:
        raise ValueError("q_learning supports Discrete action spaces only.")

    is_discrete_obs = isinstance(getattr(env.raw_env, "observation_space", None), gym.spaces.Discrete)
    q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(env.action_dim, dtype=np.float64))

    t0 = time.perf_counter()
    steps = 0
    ep_idx = 0
    learning_curve: list[dict[str, float]] = []
    td_errors: list[float] = []

    while steps < cfg.total_timesteps:
        obs, info = env.reset(seed=cfg.seed + ep_idx)
        ep_idx += 1
        done = False
        ep_ret = 0.0

        while not done and steps < cfg.total_timesteps:
            s_key = discretize_observation(obs, is_discrete_obs=is_discrete_obs, bins=cfg.bins)
            eps = cfg.eps_start + min(1.0, steps / max(1, cfg.total_timesteps)) * (cfg.eps_end - cfg.eps_start)
            mask = info.get("action_mask")
            valid = np.flatnonzero(mask) if mask is not None else np.arange(env.action_dim)
            if valid.size == 0:
                valid = np.arange(env.action_dim)

            if float(rng.random()) < eps:
                act = int(rng.choice(valid))
            else:
                best_q = np.max(q_table[s_key][valid])
                act = int(rng.choice(valid[np.isclose(q_table[s_key][valid], best_q)]))

            next_obs, reward, term, trunc, next_info = env.step(act)
            ep_ret += float(reward)
            steps += 1
            ns_key = discretize_observation(next_obs, is_discrete_obs=is_discrete_obs, bins=cfg.bins)

            # [AI-Prompt Hook]: Off-policy Bellman optimality target over valid next actions.
            next_mask = next_info.get("action_mask")
            if next_mask is not None and np.any(next_mask):
                max_next_q = float(np.max(q_table[ns_key][np.asarray(next_mask, dtype=np.bool_)]))
            else:
                max_next_q = float(np.max(q_table[ns_key]))

            target = float(reward) if term else float(reward) + cfg.gamma * max_next_q
            td_err = target - q_table[s_key][act]
            q_table[s_key][act] += cfg.alpha * td_err
            td_errors.append(abs(td_err))

            obs = next_obs
            info = next_info
            done = bool(term or trunc)

        if ep_idx % 10 == 0 or steps >= cfg.total_timesteps:
            learning_curve.append({"step": float(steps), "episode_return": float(ep_ret), "epsilon": float(eps)})

    def policy_fn(obs_in: np.ndarray, info_in: dict[str, Any] | None = None) -> int:
        s_k = discretize_observation(obs_in, is_discrete_obs=is_discrete_obs, bins=cfg.bins)
        q_row = q_table[s_k] if s_k in q_table else np.zeros(env.action_dim, dtype=np.float64)
        mask = (info_in or {}).get("action_mask")
        if mask is not None:
            valid = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
            if valid.size > 0:
                return int(valid[int(np.argmax(q_row[valid]))])
        return int(np.argmax(q_row))

    eval_target = eval_env_or_factory if eval_env_or_factory is not None else env
    eval_stats = evaluate_policy_episodes(eval_target, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=dict(q_table),
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "visited_states": len(q_table),
            "mean_abs_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
        },
        eval_stats=eval_stats,
    )
