"""Stage 1 Baseline Sanity: evaluate Random and Heuristic baselines before launching DRL training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from wannado.drl.core.types import EpisodeEvalStats
from wannado.drl.core.wrappers import MultiAgentEnvAdapter, SequenceTaskEnv, UniversalGymAdapter, ensure_gym_env


def evaluate_policy_episodes(
    env_or_factory: Any,
    policy_fn: Callable[[Any, dict[str, Any]], Any],
    *,
    num_episodes: int = 10,
    seed: int = 1000,
    max_steps_per_episode: int | None = None,
) -> EpisodeEvalStats:
    """Evaluate any policy callable deterministically across multiple episodes with sub-reward tracking."""
    policy_spec = getattr(policy_fn, "obs_spec", None)
    needs_structured = bool(policy_spec is not None and getattr(policy_spec, "kind", "vector") != "vector")
    env: UniversalGymAdapter = ensure_gym_env(env_or_factory, structured_obs=needs_structured)
    returns: list[float] = []
    lengths: list[int] = []
    term_count = 0
    trunc_count = 0
    comp_sums: dict[str, float] = {}
    comp_counts: dict[str, int] = {}
    kpi_sums: dict[str, float] = {}
    kpi_counts: dict[str, int] = {}
    ep_costs: list[float] = []
    saw_cost = False

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        # Notify stateful policies (e.g. GRU hidden state reset) if supported.
        reset_hook = getattr(policy_fn, "reset", None)
        if callable(reset_hook):
            reset_hook()

        ep_ret = 0.0
        ep_len = 0
        ep_cost = 0.0
        done = False
        last_info: dict[str, Any] = dict(info)
        while not done:
            action = policy_fn(obs, last_info)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            if "cost" in last_info and isinstance(last_info["cost"], (int, float, np.number)):
                ep_cost += float(last_info["cost"])
                saw_cost = True

            # Aggregate sub-reward terms from info["reward_terms"] if supplied by env_wrapper
            r_terms = last_info.get("reward_terms")
            if isinstance(r_terms, dict):
                for k, v in r_terms.items():
                    if isinstance(v, (int, float)):
                        comp_sums[k] = comp_sums.get(k, 0.0) + float(v)
                        comp_counts[k] = comp_counts.get(k, 0) + 1

            # Aggregate domain KPIs from info["task_metrics"] if supplied
            t_metrics = last_info.get("task_metrics")
            if isinstance(t_metrics, dict):
                for k, v in t_metrics.items():
                    if isinstance(v, (int, float)):
                        kpi_sums[k] = kpi_sums.get(k, 0.0) + float(v)
                        kpi_counts[k] = kpi_counts.get(k, 0) + 1

            if max_steps_per_episode is not None and ep_len >= max_steps_per_episode and not terminated:
                truncated = True
            done = bool(terminated or truncated)
            if done:
                if terminated:
                    term_count += 1
                elif truncated:
                    trunc_count += 1

        returns.append(ep_ret)
        lengths.append(ep_len)
        ep_costs.append(ep_cost)

    arr = np.asarray(returns, dtype=np.float64)
    return EpisodeEvalStats(
        returns=returns,
        lengths=lengths,
        mean_return=float(np.mean(arr)) if len(arr) else 0.0,
        std_return=float(np.std(arr, ddof=0)) if len(arr) else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        reward_components_mean={k: comp_sums[k] / max(comp_counts[k], 1) for k in comp_sums},
        task_metrics_mean={k: kpi_sums[k] / max(kpi_counts[k], 1) for k in kpi_sums},
        terminated_fraction=float(term_count / max(num_episodes, 1)),
        truncated_fraction=float(trunc_count / max(num_episodes, 1)),
        mean_cost=float(np.mean(ep_costs)) if saw_cost and ep_costs else None,
    )


def evaluate_sequence_task(
    task: SequenceTaskEnv,
    generate_fn: Callable[[np.ndarray], np.ndarray],
    *,
    num_prompts: int = 24,
    seed: int = 10_000,
) -> EpisodeEvalStats:
    """Evaluate an autoregressive sequence policy deterministically on verifiable prompts."""
    rng = np.random.default_rng(seed)
    prompts = task.sample_prompts(num_prompts, rng)
    responses = generate_fn(prompts)
    returns: list[float] = []
    lengths: list[int] = []
    kpi_sums: dict[str, float] = {}
    kpi_counts: dict[str, int] = {}

    for i in range(prompts.shape[0]):
        rew, metrics = task.score_sequence(prompts[i], responses[i])
        returns.append(float(rew))
        lengths.append(len(responses[i]))
        for k, v in metrics.items():
            kpi_sums[k] = kpi_sums.get(k, 0.0) + float(v)
            kpi_counts[k] = kpi_counts.get(k, 0) + 1

    arr = np.asarray(returns, dtype=np.float64)
    return EpisodeEvalStats(
        returns=returns,
        lengths=lengths,
        mean_return=float(np.mean(arr)) if len(arr) else 0.0,
        std_return=float(np.std(arr, ddof=0)) if len(arr) else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        task_metrics_mean={k: kpi_sums[k] / max(kpi_counts[k], 1) for k in kpi_sums},
        terminated_fraction=1.0,
        truncated_fraction=0.0,
    )



def evaluate_random_baseline(
    env_or_factory: Any,
    *,
    num_episodes: int = 10,
    seed: int = 1000,
) -> EpisodeEvalStats:
    """Measure the untrained random-policy floor (respecting action_mask when present)."""
    env = ensure_gym_env(env_or_factory)
    rng = np.random.default_rng(seed)

    def _random_policy(_obs: np.ndarray, info: dict[str, Any]) -> Any:
        if env.is_discrete:
            mask = info.get("action_mask")
            if mask is not None:
                valid = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
                if valid.size > 0:
                    return int(rng.choice(valid))
            return int(rng.integers(0, env.action_dim))
        if env.action_nvec is not None:
            return np.asarray([int(rng.integers(0, k)) for k in env.action_nvec], dtype=np.int64)
        low_raw = env.action_low if env.action_low is not None else np.full(env.action_dim, -1.0, dtype=np.float32)
        high_raw = env.action_high if env.action_high is not None else np.full(env.action_dim, 1.0, dtype=np.float32)
        low = np.where(np.isfinite(low_raw), low_raw, -1.0)
        high = np.where(np.isfinite(high_raw), high_raw, 1.0)
        return rng.uniform(low, high).astype(np.float32)

    return evaluate_policy_episodes(env, _random_policy, num_episodes=num_episodes, seed=seed)


def evaluate_heuristic_baseline(
    env_or_factory: Any,
    heuristic_fn: Callable[[np.ndarray, dict[str, Any]], Any] | None = None,
    *,
    num_episodes: int = 10,
    seed: int = 1000,
) -> EpisodeEvalStats | None:
    """Evaluate domain heuristic policy (from explicit arg or env.heuristic_fn)."""
    env = ensure_gym_env(env_or_factory)
    fn = heuristic_fn or env.heuristic_fn
    if fn is None:
        return None
    return evaluate_policy_episodes(env, fn, num_episodes=num_episodes, seed=seed)


def evaluate_marl_episodes(
    env_or_adapter: Any,
    policy_fn: Callable[[dict[str, np.ndarray], dict[str, Any]], dict[str, Any]],
    *,
    num_episodes: int = 10,
    seed: int = 1000,
    max_steps_per_episode: int = 200,
) -> EpisodeEvalStats:
    """Evaluate a multi-agent joint policy deterministically over team episodes."""
    adapter = env_or_adapter if isinstance(env_or_adapter, MultiAgentEnvAdapter) else MultiAgentEnvAdapter(env_or_adapter)
    returns: list[float] = []
    lengths: list[int] = []
    term_count, trunc_count = 0, 0
    for ep in range(num_episodes):
        obs_dict, _state, info = adapter.reset(seed=seed + ep)
        ep_ret = 0.0
        ep_len = 0
        done = False
        while not done and ep_len < max_steps_per_episode:
            actions = policy_fn(obs_dict, info)
            obs_dict, rew_dict, term_dict, trunc_dict, _state, info = adapter.step(actions)
            ep_ret += float(np.mean(list(rew_dict.values())))
            ep_len += 1
            any_term = any(term_dict.values())
            any_trunc = any(trunc_dict.values()) or ep_len >= max_steps_per_episode
            done = bool(any_term or any_trunc)
            if done:
                if any_term:
                    term_count += 1
                else:
                    trunc_count += 1
        returns.append(ep_ret)
        lengths.append(ep_len)

    arr = np.asarray(returns, dtype=np.float64)
    return EpisodeEvalStats(
        returns=returns,
        lengths=lengths,
        mean_return=float(np.mean(arr)) if len(arr) else 0.0,
        std_return=float(np.std(arr, ddof=0)) if len(arr) else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        terminated_fraction=float(term_count / max(num_episodes, 1)),
        truncated_fraction=float(trunc_count / max(num_episodes, 1)),
    )


