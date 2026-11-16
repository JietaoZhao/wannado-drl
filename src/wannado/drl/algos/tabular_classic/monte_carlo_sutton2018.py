"""First-Visit Monte Carlo Control (Sutton & Barto, 2018, Chapter 5).

===============================================================================
[Academic Provenance & Lineage]
- Tier          : C (Foundational Pedagogical Tabular Baseline)
- Family        : tabular_classic (Monte Carlo / Episodic Return Learning)
- Reference     : Sutton, R. S., & Barto, A. G. (2018).
                  "Reinforcement Learning: An Introduction" (2nd ed., Chapter 5). MIT Press.
- URL           : https://incompleteideas.net/book/the-book-2nd.html
- Plain Intuition:
    Think of Monte Carlo Control like learning chess purely by playing entire games
    until checkmate/draw, then looking back at every move you made and updating its
    score with the actual final score from that moment onward—no guessing mid-game!
- Lineage Role:
    Origin node of episodic reinforcement learning. Unlike Temporal-Difference (SARSA /
    Q-Learning), Monte Carlo has zero bootstrapping bias because it waits for full
    returns G_t, at the cost of higher variance and requiring finite episodes.
- Extension Directions [AI-Prompt Hook]:
    1. Replace full-episode return G_t with 1-step TD bootstrapping -> `sarsa` / `q_learning`.
    2. Use importance sampling ratios for off-policy Monte Carlo evaluation.
    3. Parameterize policy directly instead of Q-table -> `reinforce`.
===============================================================================
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import discretize_observation, set_global_seed
from wannado.drl.core.wrappers import ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="monte_carlo",
    name="First-Visit Monte Carlo Control",
    family="tabular_classic",
    paradigm="tabular_on_policy",
    tier="C",
    on_policy=True,
    sample_efficiency="low",
    action_spaces=("discrete",),
    env_modalities=("tabular", "discrete_control"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id=None,
    diff_from_parent="Foundational tabular episodic baseline: averages complete episode returns G_t without bootstrapping.",
    summary_plain="Learns action values from complete finished episodes with zero bootstrapping; ideal for learning RL fundamentals.",
    when_to_use="Small discrete/discretized episodic tasks, teaching/debugging RL basics, or non-Markovian episodic benchmarks.",
    when_not_to_use="Continuous control, infinite-horizon streaming tasks, or high-dimensional observations (use DQN or PPO).",
    hyperparam_intuition={
        "gamma": "How much future rewards matter relative to immediate rewards (0.99 = far-sighted).",
        "eps_start": "Starting exploration probability so the agent tries unseen moves early on.",
        "alpha": "Step size (or 1/N visit count when None) controlling how fast Q-table values move.",
    },
    extension_hooks=(
        "Swap full return G_t for 1-step on-policy TD target -> `sarsa`",
        "Add off-policy weighted importance sampling ratio rho_{t:T-1}",
    ),
    module_path="wannado.drl.algos.tabular_classic.monte_carlo_sutton2018",
    citation=AlgoCitation(
        key="sutton2018reinforcement",
        title="Reinforcement Learning: An Introduction (2nd ed.)",
        authors="Richard S. Sutton and Andrew G. Barto",
        venue="MIT Press",
        year=2018,
        url="https://incompleteideas.net/book/the-book-2nd.html",
        bibtex=(
            "@book{sutton2018reinforcement,\n"
            "  title={Reinforcement Learning: An Introduction},\n"
            "  author={Sutton, Richard S. and Barto, Andrew G.},\n"
            "  edition={2nd},\n"
            "  year={2018},\n"
            "  publisher={MIT Press},\n"
            "  url={https://incompleteideas.net/book/the-book-2nd.html}\n"
            "}"
        ),
        ieee='R. S. Sutton and A. G. Barto, "Reinforcement Learning: An Introduction," 2nd ed., MIT Press, 2018.',
        apa="Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.",
    ),
)


@dataclass
class MonteCarloConfig:
    total_timesteps: int = 15_000
    gamma: float = 0.99
    eps_start: float = 0.4
    eps_end: float = 0.05
    alpha: float | None = 0.1
    bins: int = 8
    eval_episodes: int = 10
    seed: int = 42


def _select_eps_greedy(
    q_row: np.ndarray,
    eps: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> int:
    valid = np.flatnonzero(mask) if mask is not None else np.arange(len(q_row))
    if valid.size == 0:
        valid = np.arange(len(q_row))
    if float(rng.random()) < eps:
        return int(rng.choice(valid))
    best_q = np.max(q_row[valid])
    tied = valid[np.isclose(q_row[valid], best_q)]
    return int(rng.choice(tied))


def train_monte_carlo(
    env_or_factory: Any,
    config: MonteCarloConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train First-Visit Monte Carlo Control on any Discrete-action environment."""
    cfg = config or MonteCarloConfig()
    rng = set_global_seed(cfg.seed)
    env = ensure_gym_env(env_or_factory)
    if not env.is_discrete:
        raise ValueError("monte_carlo supports Discrete action spaces only.")

    is_discrete_obs = isinstance(getattr(env.raw_env, "observation_space", None), gym.spaces.Discrete)
    q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(env.action_dim, dtype=np.float64))
    visit_counts: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(env.action_dim, dtype=np.int64))

    t0 = time.perf_counter()
    steps = 0
    ep_idx = 0
    learning_curve: list[dict[str, float]] = []

    while steps < cfg.total_timesteps:
        obs, info = env.reset(seed=cfg.seed + ep_idx)
        ep_idx += 1
        trajectory: list[tuple[tuple[int, ...], int, float]] = []
        done = False
        ep_ret = 0.0
        frac = min(1.0, steps / max(1, cfg.total_timesteps))
        eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

        while not done and steps < cfg.total_timesteps:
            s_key = discretize_observation(obs, is_discrete_obs=is_discrete_obs, bins=cfg.bins)
            act = _select_eps_greedy(q_table[s_key], eps, rng, mask=info.get("action_mask"))
            next_obs, reward, term, trunc, info = env.step(act)
            trajectory.append((s_key, act, float(reward)))
            ep_ret += float(reward)
            steps += 1
            obs = next_obs
            done = bool(term or trunc)

        # [AI-Prompt Hook]: Replace backward G_t accumulation below with TD(lambda) eligibility traces or n-step returns.
        g = 0.0
        first_visit_idx: dict[tuple[tuple[int, ...], int], int] = {}
        for idx, (s_k, a_k, _) in enumerate(trajectory):
            if (s_k, a_k) not in first_visit_idx:
                first_visit_idx[(s_k, a_k)] = idx

        for idx in range(len(trajectory) - 1, -1, -1):
            s_k, a_k, r_k = trajectory[idx]
            g = cfg.gamma * g + r_k
            if first_visit_idx[(s_k, a_k)] == idx:
                visit_counts[s_k][a_k] += 1
                lr = cfg.alpha if cfg.alpha is not None else (1.0 / visit_counts[s_k][a_k])
                q_table[s_k][a_k] += lr * (g - q_table[s_k][a_k])

        if ep_idx % max(1, 10) == 0 or steps >= cfg.total_timesteps:
            learning_curve.append({"step": float(steps), "episode_return": float(ep_ret), "epsilon": float(eps)})

    def policy_fn(obs_in: np.ndarray, info_in: dict[str, Any] | None = None) -> int:
        s_key = discretize_observation(obs_in, is_discrete_obs=is_discrete_obs, bins=cfg.bins)
        q_row = q_table[s_key] if s_key in q_table else np.zeros(env.action_dim, dtype=np.float64)
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
        train_diagnostics={"visited_states": len(q_table), "final_epsilon": float(cfg.eps_end)},
        eval_stats=eval_stats,
    )
