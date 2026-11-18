"""Stage 3: Anti-Fake-Convergence Auditor for DRL (Agarwal et al. 2021 & Henderson et al. 2018).

Automatically detects common DRL publication & engineering traps:
1. Single-seed survivorship bias / extreme cross-seed variance (enforces multi-seed IQM).
2. Failure to genuinely beat Random or Heuristic baselines.
3. Suspected Reward Hacking (total return climbs while domain KPIs stall/drop, or trivial suicide/stalling).
4. Evaluation observation-normalization leakage (unfrozen RunningMeanStd / VecNormalize stats).
5. Policy entropy collapse, excessive KL drift, or Critic value divergence (Explained Variance < 0 / Q explosion).
6. Train-vs-Deterministic-Eval collapse.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wannado.drl.audit.preflight import PreflightReport
from wannado.drl.core.types import EpisodeEvalStats, TrainResult
from wannado.drl.core.utils import aggregate_seed_stats
from wannado.drl.mdp.spec import MDPSpec


@dataclass
class RigorAuditReport:
    """Structured Stage 3 Anti-Fake-Convergence Audit report."""

    passed: bool
    red_flags: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    seed_stats: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "red_flags": list(self.red_flags),
            "checks": dict(self.checks),
            "seed_stats": dict(self.seed_stats),
            "metrics": dict(self.metrics),
            "recommendations": list(self.recommendations),
        }

    def format_report(self) -> str:
        verdict = "PASS (Genuine Multi-Seed Learning)" if self.passed else "WARNING (Potential Fake Convergence / Reproducibility Trap)"
        lines = [f"[wannado.drl Rigor Audit: {verdict}]"]
        lines.append(
            f"  Multi-Seed Eval -> mean={self.seed_stats.get('mean', 0.0):.2f} ± {self.seed_stats.get('std', 0.0):.2f}, "
            f"IQM={self.seed_stats.get('iqm', 0.0):.2f}, median={self.seed_stats.get('median', 0.0):.2f}"
        )
        if self.red_flags:
            lines.append("  Red Flags:")
            lines.extend(f"    - {f}" for f in self.red_flags)
        lines.append("  Checks:")
        for k, v in self.checks.items():
            lines.append(f"    - {k}: {'OK' if v else 'FAIL'}")
        if self.recommendations:
            lines.append("  Recommendations:")
            lines.extend(f"    - {r}" for r in self.recommendations)
        return "\n".join(lines)


def audit_drl_rigor(
    seed_results: Sequence[TrainResult],
    *,
    preflight_report: PreflightReport | None = None,
    mdp_spec: MDPSpec | None = None,
    random_baseline: EpisodeEvalStats | None = None,
    heuristic_baseline: EpisodeEvalStats | None = None,
    min_seeds: int = 3,
    primary_kpi_higher_is_better: bool = True,
) -> RigorAuditReport:
    """Audit trained DRL runs across seeds for statistical rigor and fake-convergence traps."""
    if not seed_results:
        return RigorAuditReport(
            passed=False,
            red_flags=["no_runs_provided"],
            checks={"has_runs": False},
            recommendations=["Run at least 3 random seeds before auditing."],
        )

    red_flags: list[str] = []
    recommendations: list[str] = []
    checks: dict[str, bool] = {}

    rand_base = random_baseline or (preflight_report.random_baseline if preflight_report is not None else None)
    heur_base = heuristic_baseline or (preflight_report.heuristic_baseline if preflight_report is not None else None)

    eval_returns = [
        r.eval_stats.mean_return
        for r in seed_results
        if r.eval_stats is not None
    ]
    has_live_eval = len(eval_returns) > 0
    if not eval_returns:
        eval_returns = [
            float(r.learning_curve[-1]["episode_return"])
            for r in seed_results
            if r.learning_curve and "episode_return" in r.learning_curve[-1]
        ]
    ope_returns = [
        float(
            r.train_diagnostics["ope_fqe_value"]
            if "ope_fqe_value" in r.train_diagnostics
            else r.train_diagnostics["ope_wis_return"]
        )
        for r in seed_results
        if ("ope_fqe_value" in r.train_diagnostics or "ope_wis_return" in r.train_diagnostics)
    ]
    has_ope_eval = len(ope_returns) > 0
    if not eval_returns and ope_returns:
        eval_returns = ope_returns
    stats = aggregate_seed_stats(eval_returns)

    checks["has_live_or_ope_evaluation"] = bool(has_live_eval or has_ope_eval or eval_returns)
    if not checks["has_live_or_ope_evaluation"]:
        red_flags.append("no_evaluation_metric_recorded")
        recommendations.append(
            "No live `eval_stats` or offline OPE metric was recorded; supply `eval_env_or_factory` or run `evaluate_offline_ope`."
        )
    elif not has_live_eval and has_ope_eval:
        recommendations.append(
            "Pure offline run evaluated via WIS/FQE Off-Policy Evaluation (no live simulator); validate shift/safety before deployment."
        )

    # 1. Multi-seed survivorship & variance check (Henderson 2018 / Agarwal 2021)
    enough_seeds = len(seed_results) >= min_seeds
    checks["multi_seed_count_ok"] = enough_seeds
    if not enough_seeds:
        red_flags.append("single_seed_survivorship_risk")
        recommendations.append(
            f"Only {len(seed_results)} seed(s) evaluated; run at least {min_seeds} seeds and report IQM + mean ± std."
        )

    spread_ok = True
    if len(eval_returns) >= 2:
        scale = max(abs(stats["mean"]), 1.0)
        if stats["std"] > 0.6 * scale and (stats["max"] - stats["iqm"]) > 0.5 * scale:
            spread_ok = False
            red_flags.append("high_seed_variance_or_lucky_outlier")
            recommendations.append("Report IQM instead of max/best seed, stabilize reward scale, or increase training steps.")
    checks["seed_stability_ok"] = spread_ok

    # 2. Baseline dominance check (Random & Heuristic floor on live environments)
    beat_random = True
    if rand_base is not None and has_live_eval and eval_returns:
        margin = max(0.05 * abs(rand_base.mean_return), 1e-4)
        if stats["iqm"] <= rand_base.mean_return + margin:
            beat_random = False
            red_flags.append("failed_to_beat_random_baseline")
            recommendations.append(
                f"IQM ({stats['iqm']:.2f}) did not beat Random ({rand_base.mean_return:.2f}). Re-check MDP reward, lr, or timesteps."
            )
    checks["beats_random_baseline"] = beat_random

    beat_heuristic = True
    if heur_base is not None and has_live_eval and eval_returns and stats["iqm"] < heur_base.mean_return:
        beat_heuristic = False
        red_flags.append("below_heuristic_baseline")
        recommendations.append(
            f"IQM ({stats['iqm']:.2f}) trails Heuristic ({heur_base.mean_return:.2f}); tune hyperparameters or use heuristic for demos (`hil_serl`)."
        )
    checks["beats_heuristic_baseline"] = beat_heuristic

    # 3. Observation normalization evaluation-freeze guard
    norm_frozen_ok = all(r.obs_rms_frozen_on_eval for r in seed_results)
    checks["obs_rms_frozen_on_eval"] = norm_frozen_ok
    if not norm_frozen_ok:
        red_flags.append("obs_rms_unfrozen_on_eval")
        recommendations.append("Freeze RunningMeanStd / VecNormalize statistics before deterministic evaluation!")

    # 4. Reward hacking detector: KPI divergence, shaping dominance, or trivial short termination
    reward_hacking_ok = True
    if rand_base is not None and rand_base.task_metrics_mean:
        kpi_names = list(rand_base.task_metrics_mean.keys())
        if mdp_spec and mdp_spec.task_kpis:
            kpi_names = [k for k in mdp_spec.task_kpis if k in rand_base.task_metrics_mean] or kpi_names
        if kpi_names:
            k0 = kpi_names[0]
            eval_kpis = [
                r.eval_stats.task_metrics_mean[k0]
                for r in seed_results
                if r.eval_stats is not None and k0 in r.eval_stats.task_metrics_mean
            ]
            if eval_kpis:
                mean_kpi = float(np.mean(eval_kpis))
                base_kpi = float(rand_base.task_metrics_mean[k0])
                ret_improved = stats["mean"] > rand_base.mean_return + 1e-3
                kpi_degraded = (mean_kpi < base_kpi - 1e-4) if primary_kpi_higher_is_better else (mean_kpi > base_kpi + 1e-4)
                if ret_improved and kpi_degraded:
                    reward_hacking_ok = False
                    red_flags.append("suspected_reward_hacking_kpi_divergence")
                    recommendations.append(
                        f"Return improved ({rand_base.mean_return:.2f} -> {stats['mean']:.2f}) while primary KPI '{k0}' degraded ({base_kpi:.3f} -> {mean_kpi:.3f})!"
                    )

    # Check shaping reward dominance or early termination exploit
    for r in seed_results:
        es = r.eval_stats
        if es is None:
            continue
        if es.mean_length <= 2.0 and es.terminated_fraction >= 0.8 and es.mean_return < 0.0:
            reward_hacking_ok = False
            if "suspected_early_termination_exploit" not in red_flags:
                red_flags.append("suspected_early_termination_exploit")
                recommendations.append(
                    "Agent terminates immediately (len<=2) to dodge ongoing negative step penalties; add survival/progress balance."
                )
        if es.reward_components_mean and mdp_spec is not None:
            shaping_names = {t.name for t in mdp_spec.reward_terms if t.role == "shaping"}
            prim_names = {t.name for t in mdp_spec.reward_terms if t.role == "primary_objective"}
            shaping_mag = sum(abs(es.reward_components_mean.get(k, 0.0)) for k in shaping_names)
            prim_mag = sum(abs(es.reward_components_mean.get(k, 0.0)) for k in prim_names)
            if shaping_names and prim_names and shaping_mag > 3.0 * max(prim_mag, 1e-6):
                reward_hacking_ok = False
                if "shaping_reward_dominates_objective" not in red_flags:
                    red_flags.append("shaping_reward_dominates_objective")
                    recommendations.append("Shaping terms outweigh primary objective >3x; rescale reward weights.")
    checks["no_reward_hacking_signals"] = reward_hacking_ok

    # 5. Policy entropy collapse, value divergence, and train-vs-eval mismatch
    policy_health_ok = True
    train_eval_ok = True
    for r in seed_results:
        diag = r.train_diagnostics
        is_cont = (
            float(diag.get("is_continuous", 0.0)) >= 0.5
            or r.algo_id in {"sac", "td3", "ddpg", "hil_serl"}
            or (mdp_spec is not None and mdp_spec.action_spec.kind == "continuous")
        )
        logstd_val = diag.get("final_logstd_mean")
        ent_val = diag.get("final_entropy", diag.get("mean_entropy"))
        ent_collapsed = False
        if logstd_val is not None and (not math.isfinite(float(logstd_val)) or float(logstd_val) < -4.5):
            ent_collapsed = True
        elif ent_val is not None:
            ent_f = float(ent_val)
            if not math.isfinite(ent_f) or (not is_cont and ent_f < 1e-4) or (is_cont and ent_f < -8.0):
                ent_collapsed = True
        if ent_collapsed:
            policy_health_ok = False
            if "policy_entropy_collapse" not in red_flags:
                red_flags.append("policy_entropy_collapse")
                recommendations.append("Policy entropy collapsed near 0; increase `ent_coef` or `clip_high` (DAPO style).")

        ev_val = diag.get("mean_explained_variance")
        if ev_val is not None and (not math.isfinite(float(ev_val)) or float(ev_val) < -0.5):
            policy_health_ok = False
            if "critic_negative_explained_variance" not in red_flags:
                red_flags.append("critic_negative_explained_variance")
                recommendations.append("Critic explained variance < -0.5; check `terminated` vs `truncated` handling or reward scale.")

        q_val = diag.get("max_q_value", diag.get("mean_q_value"))
        if q_val is not None and (not math.isfinite(float(q_val)) or abs(float(q_val)) > 1e4):
            policy_health_ok = False
            if "q_value_explosion" not in red_flags:
                red_flags.append("q_value_explosion")
                recommendations.append("Q-values exploded (>1e4); use Double/Twin Q, LayerNorm critics, or gradient clipping.")

        sat_rate = diag.get("action_saturation_rate")
        if sat_rate is not None and float(sat_rate) > 0.95:
            policy_health_ok = False
            if "policy_action_saturation" not in red_flags:
                red_flags.append("policy_action_saturation")
                recommendations.append(
                    f"Policy actions saturated at box bounds ({float(sat_rate):.1%} of eval steps); inspect action scaling or entropy regularization."
                )

        if bool(diag.get("cost_budget_exceeded", False)):
            policy_health_ok = False
            if "safety_constraint_budget_exceeded" not in red_flags:
                red_flags.append("safety_constraint_budget_exceeded")
                recommendations.append(
                    "Policy exceeded safety constraint cost budget; increase Lagrangian multiplier LR or tighten cost limit."
                )

        if r.eval_stats is not None:
            ret_pts = [p["episode_return"] for p in r.learning_curve[-3:] if "episode_return" in p]
            if len(ret_pts) >= 3:
                late_train = float(np.mean(ret_pts))
                eval_r = r.eval_stats.mean_return
                collapse_floor = 0.25 if abs(late_train) <= 2.0 else 5.0
                if late_train > eval_r + max(0.4 * abs(late_train), collapse_floor):
                    train_eval_ok = False
                    if "train_eval_collapse_mismatch" not in red_flags:
                        red_flags.append("train_eval_collapse_mismatch")
                        recommendations.append(
                            f"Late training return ({late_train:.2f}) far exceeds deterministic eval ({eval_r:.2f}); verify seed generalization & normalization freeze."
                        )

    checks["policy_and_critic_health_ok"] = policy_health_ok
    checks["train_eval_consistency_ok"] = train_eval_ok

    passed = len(red_flags) == 0
    return RigorAuditReport(
        passed=passed,
        red_flags=red_flags,
        checks=checks,
        seed_stats=stats,
        metrics={
            "num_seeds": len(seed_results),
            "per_seed_eval_returns": eval_returns,
            "random_baseline_mean": rand_base.mean_return if rand_base is not None else None,
            "heuristic_baseline_mean": heur_base.mean_return if heur_base is not None else None,
        },
        recommendations=recommendations,
    )
