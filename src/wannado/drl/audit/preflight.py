"""Stage 1: Automated MDP Preflight & Baseline Sanity Diagnostics.

Checks environment contracts (`terminated` vs `truncated`, NaN/Inf, seeded reset
reproducibility, action bounds/masks, reward scales, sub-reward/KPI instrumentation),
measures real hardware simulation speed (`ms/step`), and establishes Random + Heuristic
baseline floors before spending compute on DRL training.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from wannado.drl.core.types import EpisodeEvalStats
from wannado.drl.core.wrappers import MultiAgentEnvAdapter, SequenceTaskEnv, ensure_gym_env
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
    probe_steps: int = 80,
    baseline_episodes: int = 6,
    seed: int = 42,
    strict: bool = False,
) -> PreflightReport:
    """Run Stage 1 MDP contract validation, speed benchmarking, and Random/Heuristic baselines."""
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []

    probe_target = env_or_factory
    if (
        callable(env_or_factory)
        and (isinstance(env_or_factory, type) or not hasattr(env_or_factory, "step"))
        and not hasattr(env_or_factory, "possible_agents")
        and not isinstance(env_or_factory, (SequenceTaskEnv, MultiAgentEnvAdapter))
    ):
        probe_target = env_or_factory()

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
        obs_d, _, _ = adapter.reset(seed=seed)
        rew_vals: list[float] = []
        ep_rets: list[float] = []
        ep_lens: list[int] = []
        cur_ret, cur_len = 0.0, 0
        for _ in range(probe_steps):
            obs_d, rews, terms, truncs, _, _ = adapter.step(_sample_marl_actions())
            step_r = float(np.mean(list(rews.values())))
            rew_vals.append(step_r)
            cur_ret += step_r
            cur_len += 1
            if any(terms.values()) or any(truncs.values()):
                ep_rets.append(cur_ret)
                ep_lens.append(cur_len)
                cur_ret, cur_len = 0.0, 0
                obs_d, _, _ = adapter.reset(seed=int(rng.integers(0, 100_000)))
        dt_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, probe_steps)

        # Ensure full random baseline episodes across MARL tasks
        for ep_i in range(max(0, baseline_episodes - len(ep_rets))):
            adapter.reset(seed=seed + 500 + ep_i)
            r_sum, l_sum = 0.0, 0
            for _ in range(500):
                _, rews, terms, truncs, _, _ = adapter.step(_sample_marl_actions())
                r_sum += float(np.mean(list(rews.values())))
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
    act_kind = "discrete" if env.is_discrete else "continuous"

    # 1. Seeded reset reproducibility check
    obs_a, info_a = env.reset(seed=seed)
    obs_b, _ = env.reset(seed=seed)
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
    obs_abs_max = float(np.max(np.abs(obs))) if obs.size else 0.0
    ep_steps = 0
    ep_lengths: list[int] = []
    saw_reward_terms = False
    saw_task_metrics = False
    saw_mask = "action_mask" in info
    one_step_terms = 0

    t0 = time.perf_counter()
    for _ in range(probe_steps):
        if env.is_discrete:
            mask = info.get("action_mask")
            if mask is not None:
                saw_mask = True
                valid = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
                if valid.size == 0:
                    errors.append("action_mask has zero valid actions; agent is trapped with no legal move.")
                    break
                act: Any = int(rng.choice(valid))
            else:
                act = int(rng.integers(0, env.action_dim))
        else:
            low = np.where(np.isfinite(env.action_low), env.action_low, -1.0)
            high = np.where(np.isfinite(env.action_high), env.action_high, 1.0)
            act = rng.uniform(low, high).astype(np.float32)

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

    # 3. Baseline sanity floors
    rand_stats = evaluate_random_baseline(env, num_episodes=baseline_episodes, seed=seed + 500) if not errors else None
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
        random_baseline=rand_stats,
        heuristic_baseline=heur_stats,
        summary={
            "obs_abs_max": obs_abs_max,
            "max_abs_reward": max_abs_r,
            "mean_episode_length": float(rand_stats.mean_length) if rand_stats is not None else (float(np.mean(ep_lengths)) if ep_lengths else None),
            "saw_action_mask": saw_mask,
            "saw_reward_terms": saw_reward_terms,
            "saw_task_metrics": saw_task_metrics,
            "probed_steps": len(rews),
            "completed_episodes_in_probe": len(ep_lengths),
        },
    )
    if strict and not ok:
        raise ValueError("; ".join(errors))
    return report
