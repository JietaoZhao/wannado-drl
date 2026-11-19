"""Stage 1: Automated MDP Preflight & Baseline Sanity Diagnostics.

Checks environment contracts (`terminated` vs `truncated`, NaN/Inf, seeded reset
reproducibility, action bounds/masks, reward scales, sub-reward/KPI instrumentation),
measures real hardware simulation speed (`ms/step`), and establishes Random + Heuristic
baseline floors before spending compute on DRL training.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wannado.drl.audit.probes import MDPProbeReport, probe_transition_batch
from wannado.drl.core.offline_risk import compute_cvar
from wannado.drl.core.types import EpisodeEvalStats
from wannado.drl.core.wrappers import (
    MultiAgentEnvAdapter,
    OfflineTransitionDataset,
    SequenceTaskEnv,
    VectorizedEnvAdapter,
    ensure_gym_env,
)
from wannado.drl.mdp.baselines import evaluate_heuristic_baseline, evaluate_random_baseline


@dataclass
class PreflightReport:
    """Structured Stage 1 MDP preflight & baseline diagnostic card."""

    ok: bool
    env_library: str
    obs_dim: int
    action_space_kind: str
    action_dim: int
    ms_per_step: float
    steps_per_sec: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    random_baseline: EpisodeEvalStats | None = None
    heuristic_baseline: EpisodeEvalStats | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "env_library": self.env_library,
            "obs_dim": self.obs_dim,
            "action_space_kind": self.action_space_kind,
            "action_dim": self.action_dim,
            "ms_per_step": self.ms_per_step,
            "steps_per_sec": self.steps_per_sec,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "recommendations": list(self.recommendations),
            "random_baseline": self.random_baseline.to_dict() if self.random_baseline is not None else None,
            "heuristic_baseline": self.heuristic_baseline.to_dict() if self.heuristic_baseline is not None else None,
            "summary": dict(self.summary),
        }

    def format_report(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"[wannado.drl Stage 1 MDP Preflight: {status}]",
            f"  Env Library : {self.env_library} | obs_dim={self.obs_dim}, action={self.action_space_kind}({self.action_dim})",
            f"  Sim Speed   : {self.ms_per_step:.3f} ms/step (~{self.steps_per_sec:,.0f} steps/s)",
        ]
        if self.random_baseline is not None:
            lines.append(
                f"  Random Floor: return={self.random_baseline.mean_return:.2f} ± {self.random_baseline.std_return:.2f}, "
                f"len={self.random_baseline.mean_length:.1f}"
            )
        if self.heuristic_baseline is not None:
            lines.append(
                f"  Heuristic   : return={self.heuristic_baseline.mean_return:.2f} ± {self.heuristic_baseline.std_return:.2f}, "
                f"len={self.heuristic_baseline.mean_length:.1f}"
            )
        if self.errors:
            lines.append("  Errors:")
            lines.extend(f"    - {e}" for e in self.errors)
        if self.warnings:
            lines.append("  Warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        if self.recommendations:
            lines.append("  Recommendations:")
            lines.extend(f"    - {r}" for r in self.recommendations)
        return "\n".join(lines)


def preflight_env(
    env_or_factory: Any,
    *,
    heuristic_fn: Callable[[np.ndarray, dict[str, Any]], Any] | None = None,
    probe_steps: int = 120,
    baseline_episodes: int = 6,
    seed: int = 42,
    strict: bool = False,
    run_probes: bool = True,
) -> PreflightReport:
    """Run Stage 1 MDP contract validation, speed benchmarking, and Random/Heuristic baselines."""
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []

    probe_target = env_or_factory
    if isinstance(probe_target, VectorizedEnvAdapter):
        probe_target = probe_target.raw_env
    elif (
        callable(env_or_factory)
        and (isinstance(env_or_factory, type) or not hasattr(env_or_factory, "step"))
        and not hasattr(env_or_factory, "possible_agents")
        and not isinstance(env_or_factory, (SequenceTaskEnv, MultiAgentEnvAdapter, OfflineTransitionDataset))
    ):
        probe_target = env_or_factory()

    if isinstance(probe_target, OfflineTransitionDataset):
        t0 = time.perf_counter()
        n = len(probe_target)
        if n < 2:
            errors.append("OfflineTransitionDataset contains fewer than 2 transitions.")

        def _obs_all_finite(arr_or_dict: Any) -> bool:
            if isinstance(arr_or_dict, dict):
                return all(bool(np.all(np.isfinite(np.asarray(v)))) for v in arr_or_dict.values())
            return bool(np.all(np.isfinite(np.asarray(arr_or_dict))))

        def _obs_max_abs(arr_or_dict: Any) -> float:
            if isinstance(arr_or_dict, dict):
                vals = [float(np.max(np.abs(np.asarray(v)))) for v in arr_or_dict.values() if np.asarray(v).size]
                return max(vals) if vals else 0.0
            arr = np.asarray(arr_or_dict)
            return float(np.max(np.abs(arr))) if arr.size else 0.0

        def _obs_prefix_2d(arr_or_dict: Any, limit: int) -> np.ndarray:
            if isinstance(arr_or_dict, dict):
                parts = [
                    np.asarray(v[:limit], dtype=np.float64).reshape(limit, -1)
                    for _, v in sorted(arr_or_dict.items())
                ]
                return np.hstack(parts) if parts else np.zeros((limit, 0), dtype=np.float64)
            return np.asarray(arr_or_dict[:limit], dtype=np.float64).reshape(limit, -1)

        if not _obs_all_finite(probe_target.obs) or not _obs_all_finite(probe_target.next_obs):
            errors.append("OfflineTransitionDataset contains NaN or Inf observations.")
        if not np.all(np.isfinite(probe_target.rewards)):
            errors.append("OfflineTransitionDataset contains NaN or Inf rewards.")
        obs_abs_max = _obs_max_abs(probe_target.obs)
        max_abs_r = float(np.max(np.abs(probe_target.rewards))) if probe_target.rewards.size else 0.0
        if max_abs_r > 100.0:
            warnings.append(
                f"Large per-step reward magnitude in offline log (max |r|={max_abs_r:.1f}); consider reward scaling."
            )
        if obs_abs_max > 50.0:
            recommendations.append(
                f"Offline observations reach |obs|={obs_abs_max:.1f}; keep `normalize_obs=True` (computed strictly on logged obs)."
            )

        term_idx = np.flatnonzero(probe_target.terminateds > 0.5)
        ep_ids_raw = probe_target.extra.get("episode_ids")
        off_ep_rets: list[float] = []
        off_ep_lens: list[int] = []
        if ep_ids_raw is not None and len(ep_ids_raw) == n:
            start = 0
            for i in range(1, n):
                if ep_ids_raw[i] != ep_ids_raw[i - 1]:
                    off_ep_rets.append(float(np.sum(probe_target.rewards[start:i])))
                    off_ep_lens.append(i - start)
                    start = i
            off_ep_rets.append(float(np.sum(probe_target.rewards[start:n])))
            off_ep_lens.append(n - start)
        elif term_idx.size > 0:
            start = 0
            for end_i in term_idx:
                off_ep_rets.append(float(np.sum(probe_target.rewards[start : int(end_i) + 1])))
                off_ep_lens.append(int(end_i) + 1 - start)
                start = int(end_i) + 1
            if start < n:
                off_ep_rets.append(float(np.sum(probe_target.rewards[start:n])))
                off_ep_lens.append(n - start)
        else:
            off_ep_rets.append(float(np.sum(probe_target.rewards)))
            off_ep_lens.append(n)

        beh_stats = EpisodeEvalStats(
            returns=off_ep_rets,
            lengths=off_ep_lens,
            mean_return=float(np.mean(off_ep_rets)) if off_ep_rets else 0.0,
            std_return=float(np.std(off_ep_rets)) if off_ep_rets else 0.0,
            mean_length=float(np.mean(off_ep_lens)) if off_ep_lens else float(n),
        )
        probe_report: MDPProbeReport | None = None
        if run_probes and not errors:
            trunc_arr = getattr(probe_target, "truncateds", None)
            off_dones = (
                np.logical_or(probe_target.terminateds > 0.5, trunc_arr > 0.5)
                if trunc_arr is not None
                else (probe_target.terminateds > 0.5)
            )
            n_lim = min(n, 256)
            probe_report = probe_transition_batch(
                _obs_prefix_2d(probe_target.obs, n_lim),
                probe_target.actions[:n_lim],
                probe_target.rewards[:n_lim],
                _obs_prefix_2d(probe_target.next_obs, n_lim),
                off_dones[:n_lim],
                masks=probe_target.extra["action_masks"][:n_lim] if "action_masks" in probe_target.extra else None,
                n_permutations=30,
                seed=seed,
            )
            for msg in probe_report.warnings:
                if msg not in warnings:
                    warnings.append(msg)
            for msg in probe_report.errors:
                if msg not in errors:
                    errors.append(msg)
            for msg in probe_report.recommendations:
                if msg not in recommendations:
                    recommendations.append(msg)

        dt_ms = max(0.001, (time.perf_counter() - t0) * 1000.0)
        ok = len(errors) == 0
        rep = PreflightReport(
            ok=ok,
            env_library="offline_dataset",
            obs_dim=probe_target.obs_dim,
            action_space_kind="discrete" if probe_target.is_discrete else "continuous",
            action_dim=probe_target.action_dim,
            ms_per_step=round(dt_ms / max(1, min(n, 1000)), 4),
            steps_per_sec=round(1000.0 / max(dt_ms / max(1, min(n, 1000)), 1e-4), 1),
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            random_baseline=beh_stats,
            summary={
                "num_transitions": n,
                "num_episodes_estimated": len(off_ep_rets),
                "behavior_mean_return": beh_stats.mean_return,
                "behavior_cvar_05": compute_cvar(off_ep_rets, alpha=0.05),
                "obs_abs_max": obs_abs_max,
                "max_abs_reward": max_abs_r,
                "saw_action_mask": "action_masks" in probe_target.extra,
                **({"probes": probe_report.to_dict()} if probe_report is not None else {}),
            },
        )
        if strict and not ok:
            raise ValueError("; ".join(errors))
        return rep

    if isinstance(probe_target, SequenceTaskEnv):
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        prompts = probe_target.sample_prompts(min(16, max(4, baseline_episodes)), rng)
        rand_resps = rng.integers(0, probe_target.vocab_size, size=(prompts.shape[0], probe_target.max_response_len))
        scores = [probe_target.score_sequence(prompts[i], rand_resps[i])[0] for i in range(prompts.shape[0])]
        dt_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, prompts.shape[0])
        rand_stats = EpisodeEvalStats(
            returns=[float(s) for s in scores],
            lengths=[probe_target.max_response_len] * len(scores),
            mean_return=float(np.mean(scores)),
            std_return=float(np.std(scores)),
            mean_length=float(probe_target.max_response_len),
        )
        return PreflightReport(
            ok=True,
            env_library="sequence_task",
            obs_dim=probe_target.prompt_len,
            action_space_kind="autoregressive_sequence",
            action_dim=probe_target.vocab_size,
            ms_per_step=round(dt_ms, 4),
            steps_per_sec=round(1000.0 / max(dt_ms, 1e-4), 1),
            random_baseline=rand_stats,
            summary={"task_name": probe_target.task_name},
        )

    if isinstance(probe_target, MultiAgentEnvAdapter) or hasattr(probe_target, "possible_agents"):
        adapter = probe_target if isinstance(probe_target, MultiAgentEnvAdapter) else MultiAgentEnvAdapter(probe_target)
        rng = np.random.default_rng(seed)

        def _sample_marl_actions() -> dict[str, Any]:
            return {
                a: (
                    int(rng.integers(0, adapter.action_dim))
                    if adapter.is_discrete
                    else rng.uniform(
                        adapter.action_low if adapter.action_low is not None else -1.0,
                        adapter.action_high if adapter.action_high is not None else 1.0,
                        size=adapter.action_dim,
                    ).astype(np.float32)
                )
                for a in adapter.agents
            }

        t0 = time.perf_counter()
        _obs_d, _, _ = adapter.reset(seed=seed)
        rew_vals: list[float] = []
        ep_rets: list[float] = []
        ep_lens: list[int] = []
        cur_ret, cur_len = 0.0, 0
        for _ in range(probe_steps):
            _obs_d, rew_dict, terms, truncs, _, _ = adapter.step(_sample_marl_actions())
            step_r = float(np.mean(list(rew_dict.values())))
            rew_vals.append(step_r)
            cur_ret += step_r
            cur_len += 1
            if any(terms.values()) or any(truncs.values()):
                ep_rets.append(cur_ret)
                ep_lens.append(cur_len)
                cur_ret, cur_len = 0.0, 0
                _obs_d, _, _ = adapter.reset(seed=int(rng.integers(0, 100_000)))
        dt_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, probe_steps)

        # Ensure full random baseline episodes across MARL tasks
        for ep_i in range(max(0, baseline_episodes - len(ep_rets))):
            adapter.reset(seed=seed + 500 + ep_i)
            r_sum, l_sum = 0.0, 0
            for _ in range(500):
                _, rew_dict, terms, truncs, _, _ = adapter.step(_sample_marl_actions())
                r_sum += float(np.mean(list(rew_dict.values())))
                l_sum += 1
                if any(terms.values()) or any(truncs.values()):
                    break
            ep_rets.append(r_sum)
            ep_lens.append(l_sum)

        rand_stats = EpisodeEvalStats(
            returns=[float(x) for x in ep_rets],
            lengths=[int(x) for x in ep_lens],
            mean_return=float(np.mean(ep_rets)) if ep_rets else 0.0,
            std_return=float(np.std(ep_rets)) if ep_rets else 0.0,
            mean_length=float(np.mean(ep_lens)) if ep_lens else float(probe_steps),
        )
        return PreflightReport(
            ok=True,
            env_library="multi_agent",
            obs_dim=adapter.obs_dim,
            action_space_kind="multi_agent_discrete" if adapter.is_discrete else "multi_agent_continuous",
            action_dim=adapter.action_dim,
            ms_per_step=round(dt_ms, 4),
            steps_per_sec=round(1000.0 / max(dt_ms, 1e-4), 1),
            random_baseline=rand_stats,
            summary={
                "num_agents": adapter.num_agents,
                "state_dim": adapter.state_dim,
                "probe_rew_mean": float(np.mean(rew_vals)) if rew_vals else 0.0,
                "mean_episode_length": rand_stats.mean_length,
            },
        )

    env = ensure_gym_env(probe_target, heuristic_fn=heuristic_fn)
    act_kind = env.raw_action_kind

    # 1. Seeded reset reproducibility check
    obs_a, _info_a = env.reset(seed=seed)
    obs_b, _ = env.reset(seed=seed)
    reset_samples: list[np.ndarray] = [np.asarray(obs_a, dtype=np.float64).reshape(-1)]
    if run_probes:
        for r_off in (1001, 1002):
            obs_r, _ = env.reset(seed=seed + r_off)
            reset_samples.append(np.asarray(obs_r, dtype=np.float64).reshape(-1))
    if not np.all(np.isfinite(obs_a)):
        errors.append("Initial observation from reset() contains NaN or Inf.")
    if obs_a.shape != (env.obs_dim,):
        errors.append(f"Observation shape mismatch: expected ({env.obs_dim},), got {obs_a.shape}.")
    if not np.allclose(obs_a, obs_b, atol=1e-6):
        warnings.append(
            "reset(seed=...) returned different observations across identical seeds; check environment RNG seeding for reproducibility."
        )

    # 2. Timed random rollout probing numerical health, action masks, rewards, and terminated/truncated
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    rews: list[float] = []
    probe_obs: list[np.ndarray] = []
    probe_next_obs: list[np.ndarray] = []
    probe_acts: list[np.ndarray] = []
    probe_dones: list[bool] = []
    probe_masks: list[np.ndarray] = []
    obs_abs_max = float(np.max(np.abs(obs))) if obs.size else 0.0
    ep_steps = 0
    ep_lengths: list[int] = []
    saw_reward_terms = False
    saw_task_metrics = False
    saw_mask = "action_mask" in info
    one_step_terms = 0

    t0 = time.perf_counter()
    for _ in range(probe_steps):
        step_mask = info.get("action_mask")
        if env.is_discrete:
            mask = step_mask
            if mask is not None:
                saw_mask = True
                m_bool = np.asarray(mask, dtype=np.bool_).reshape(-1)
                probe_masks.append(m_bool)
                valid = np.flatnonzero(m_bool)
                if valid.size == 0:
                    errors.append("action_mask has zero valid actions; agent is trapped with no legal move.")
                    break
                act: Any = int(rng.choice(valid))
            else:
                act = int(rng.integers(0, env.action_dim))
            act_vec = np.array([float(act)], dtype=np.float64)
        elif env.action_nvec is not None:
            act = np.asarray([int(rng.integers(0, k)) for k in env.action_nvec], dtype=np.int64)
            act_vec = np.asarray(act, dtype=np.float64).reshape(-1)
        else:
            low_raw = env.action_low if env.action_low is not None else np.full(env.action_dim, -1.0, dtype=np.float32)
            high_raw = env.action_high if env.action_high is not None else np.full(env.action_dim, 1.0, dtype=np.float32)
            low = np.where(np.isfinite(low_raw), low_raw, -1.0)
            high = np.where(np.isfinite(high_raw), high_raw, 1.0)
            act = rng.uniform(low, high).astype(np.float32)
            act_vec = np.asarray(act, dtype=np.float64).reshape(-1)

        prev_obs = obs
        obs, rew, term, trunc, info = env.step(act)
        ep_steps += 1
        rews.append(float(rew))

        if not np.all(np.isfinite(obs)):
            errors.append("Environment observation became NaN/Inf during rollout.")
            break
        if not np.isfinite(rew):
            errors.append("Environment reward became NaN/Inf during rollout.")
            break
        obs_abs_max = max(obs_abs_max, float(np.max(np.abs(obs))))
        probe_obs.append(np.asarray(prev_obs, dtype=np.float64).reshape(-1))
        probe_next_obs.append(np.asarray(obs, dtype=np.float64).reshape(-1))
        probe_acts.append(act_vec)
        probe_dones.append(bool(term or trunc))

        if isinstance(info.get("reward_terms"), dict) and info["reward_terms"]:
            saw_reward_terms = True
        if isinstance(info.get("task_metrics"), dict) and info["task_metrics"]:
            saw_task_metrics = True

        if term or trunc:
            ep_lengths.append(ep_steps)
            if term and ep_steps == 1:
                one_step_terms += 1
            ep_steps = 0
            obs, info = env.reset(seed=int(rng.integers(0, 100_000)))

    dt_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, len(rews))

    if one_step_terms >= 3:
        warnings.append("Multiple episodes terminated in 1 step; inspect termination threshold.")
    max_abs_r = max((abs(r) for r in rews), default=0.0)
    if rews:
        if max_abs_r > 100.0:
            warnings.append(
                f"Large per-step reward magnitude (max |r|={max_abs_r:.1f}); consider reward scaling to stabilize Critic TD targets."
            )
        if len(rews) >= 20 and float(np.std(rews)) < 1e-12:
            warnings.append(
                "All probed random rewards are identical (zero variance). If rewards are extremely sparse, add shaping or demos (HIL-SERL)."
            )
    if obs_abs_max > 50.0:
        recommendations.append(
            f"Observation values reach |obs|={obs_abs_max:.1f}; set `normalize_obs=True` when training neural policies."
        )
    if not saw_reward_terms or not saw_task_metrics:
        recommendations.append(
            "Consider returning `info['reward_terms']` and `info['task_metrics']` from env_wrapper.py so Stage 3 Auditor can catch Reward Hacking."
        )

    probe_report_gym: MDPProbeReport | None = None
    if run_probes and probe_obs and not errors:
        probe_report_gym = probe_transition_batch(
            np.vstack(probe_obs),
            np.vstack(probe_acts),
            np.asarray(rews[: len(probe_obs)], dtype=np.float64),
            np.vstack(probe_next_obs),
            np.asarray(probe_dones, dtype=np.bool_),
            reset_obs_samples=np.vstack(reset_samples) if reset_samples else None,
            masks=np.vstack(probe_masks) if len(probe_masks) == len(probe_obs) else None,
            n_permutations=40,
            seed=seed,
        )
        for msg in probe_report_gym.warnings:
            if msg not in warnings:
                warnings.append(msg)
        for msg in probe_report_gym.errors:
            if msg not in errors:
                errors.append(msg)
        for msg in probe_report_gym.recommendations:
            if msg not in recommendations:
                recommendations.append(msg)

    # 3. Baseline sanity floors
    rand_stats_single: EpisodeEvalStats | None = (
        evaluate_random_baseline(env, num_episodes=baseline_episodes, seed=seed + 500) if not errors else None
    )
    heur_stats = (
        evaluate_heuristic_baseline(env, heuristic_fn=heuristic_fn, num_episodes=baseline_episodes, seed=seed + 500)
        if not errors
        else None
    )

    ok = len(errors) == 0
    report = PreflightReport(
        ok=ok,
        env_library=env.detected_library,
        obs_dim=env.obs_dim,
        action_space_kind=act_kind,
        action_dim=env.action_dim,
        ms_per_step=round(dt_ms, 4),
        steps_per_sec=round(1000.0 / max(dt_ms, 1e-4), 1),
        warnings=warnings,
        errors=errors,
        recommendations=recommendations,
        random_baseline=rand_stats_single,
        heuristic_baseline=heur_stats,
        summary={
            "obs_abs_max": obs_abs_max,
            "max_abs_reward": max_abs_r,
            "mean_episode_length": float(rand_stats_single.mean_length)
            if rand_stats_single is not None
            else (float(np.mean(ep_lengths)) if ep_lengths else None),
            "saw_action_mask": saw_mask,
            "saw_reward_terms": saw_reward_terms,
            "saw_task_metrics": saw_task_metrics,
            "probed_steps": len(rews),
            "completed_episodes_in_probe": len(ep_lengths),
            **({"is_vector_env": True, "num_envs": env.num_envs} if env.is_vector_env else {}),
            **({"action_nvec": list(env.action_nvec)} if env.action_nvec is not None else {}),
            **({"probes": probe_report_gym.to_dict()} if probe_report_gym is not None else {}),
        },
    )
    if strict and not ok:
        raise ValueError("; ".join(errors))
    return report
