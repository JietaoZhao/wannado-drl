"""SARSA: On-Policy Temporal-Difference Control (Rummery & Niranjan, 1994).

===============================================================================
[Academic Provenance & Lineage]
- Tier          : C (Classic On-Policy TD Learning Baseline)
- Family        : tabular_classic (On-Policy TD(0))
- Reference     : Rummery, G. A., & Niranjan, M. (1994).
                  "On-line Q-learning using connectionist systems."
                  Cambridge University Engineering Dept., Tech. Rep. CUED/F-INFENG/TR 166.
- Plain Intuition:
    Unlike Monte Carlo (which waits for the whole game to end), SARSA updates each step's
    score right away using the value of the NEXT action the agent actually plans to take.
    Because it accounts for its own clumsy exploration, SARSA learns safer paths near cliffs!
- Diff vs. Monte Carlo (Sutton & Barto):
    Replaces full episodic return G_t with 1-step bootstrapped TD target r + gamma * Q(s', a'),
    where a' comes from the live behavior policy (hence State-Action-Reward-State-Action).
- Extension Directions [AI-Prompt Hook]:
    1. Replace Q(s', a') with greedy max_{a'} Q(s', a') -> `q_learning`.
    2. Use expected next Q under epsilon-greedy policy -> Expected SARSA.
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
    id="sarsa",
    name="SARSA (On-Policy TD Control)",
    family="tabular_classic",
    paradigm="tabular_on_policy",
    tier="C",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete",),
    env_modalities=("tabular", "discrete_control"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="monte_carlo",
    diff_from_parent="Replaces full-episode Monte Carlo returns with 1-step on-policy TD bootstrapping Q(s', a').",
    summary_plain="Step-by-step on-policy Q-table learning that prices in exploration risk (safer near cliffs than Q-Learning).",
    when_to_use="Learning TD fundamentals or small discrete environments where exploratory mistakes are very costly.",
    when_not_to_use="High-dimensional continuous states or when seeking pure optimal greedy path regardless of exploration (use `q_learning` or `dqn`).",
    hyperparam_intuition={
        "alpha": "Learning rate controlling how fast new TD errors overwrite old Q-table estimates.",
        "gamma": "Discount factor balancing immediate safety vs long-term reward.",
        "eps_start": "Initial exploration rate decaying toward eps_end.",
    },
    extension_hooks=(
        "Swap Q(s', a') for max_{a'} Q(s', a') -> `q_learning`",
        "Replace greedy next-action sample with expectation over policy probabilities -> Expected SARSA",
    ),
    module_path="wannado.drl.algos.tabular_classic.sarsa_rummery1994",
    citation=AlgoCitation(
        key="rummery1994online",
        title="On-line Q-learning using connectionist systems",
        authors="Gavin A. Rummery and Mahesan Niranjan",
        venue="Cambridge University Engineering Department, Technical Report CUED/F-INFENG/TR 166",
        year=1994,
        url="https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf",
        bibtex=(
            "@techreport{rummery1994online,\n"
            "  title={On-line Q-learning using connectionist systems},\n"
            "  author={Rummery, Gavin A. and Niranjan, Mahesan},\n"
            "  institution={Cambridge University Engineering Department},\n"
            "  number={CUED/F-INFENG/TR 166},\n"
            "  year={1994}\n"
            "}"
        ),
        ieee='G. A. Rummery and M. Niranjan, "On-line Q-learning using connectionist systems," Cambridge Univ. Eng. Dept., Tech. Rep. CUED/F-INFENG/TR 166, 1994.',
        apa="Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems (Tech. Rep. CUED/F-INFENG/TR 166). Cambridge University Engineering Department.",
    ),
)


@dataclass
class SarsaConfig:
    total_timesteps: int = 15_000
    alpha: float = 0.15
    gamma: float = 0.99
    eps_start: float = 0.4
    eps_end: float = 0.03
    bins: int = 8
    eval_episodes: int = 10
    seed: int = 42


def _eps_greedy(q_row: np.ndarray, eps: float, rng: np.random.Generator, mask: np.ndarray | None = None) -> int:
    valid = np.flatnonzero(mask) if mask is not None else np.arange(len(q_row))
    if valid.size == 0:
        valid = np.arange(len(q_row))
    if float(rng.random()) < eps:
        return int(rng.choice(valid))
    best_q = np.max(q_row[valid])
    return int(rng.choice(valid[np.isclose(q_row[valid], best_q)]))


def train_sarsa(
    env_or_factory: Any,
    config: SarsaConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train tabular on-policy SARSA(0) with automatic continuous-obs binning fallback."""
    cfg = config or SarsaConfig()
    rng = set_global_seed(cfg.seed)
    env = ensure_gym_env(env_or_factory)
    if not env.is_discrete:
        raise ValueError("sarsa supports Discrete action spaces only.")

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
        s_key = discretize_observation(obs, is_discrete_obs=is_discrete_obs, bins=cfg.bins)
        eps = cfg.eps_start + min(1.0, steps / max(1, cfg.total_timesteps)) * (cfg.eps_end - cfg.eps_start)
        act = _eps_greedy(q_table[s_key], eps, rng, mask=info.get("action_mask"))
        done = False
        ep_ret = 0.0

        while not done and steps < cfg.total_timesteps:
            next_obs, reward, term, trunc, next_info = env.step(act)
            ep_ret += float(reward)
            steps += 1
            ns_key = discretize_observation(next_obs, is_discrete_obs=is_discrete_obs, bins=cfg.bins)
            eps = cfg.eps_start + min(1.0, steps / max(1, cfg.total_timesteps)) * (cfg.eps_end - cfg.eps_start)
            next_act = _eps_greedy(q_table[ns_key], eps, rng, mask=next_info.get("action_mask"))

            # [AI-Prompt Hook]: SARSA bootstraps via actual next action `next_act` (zeroed ONLY when terminated=True).
            target = float(reward) if term else float(reward) + cfg.gamma * q_table[ns_key][next_act]
            td_err = target - q_table[s_key][act]
            q_table[s_key][act] += cfg.alpha * td_err
            td_errors.append(abs(td_err))

            s_key, act = ns_key, next_act
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
