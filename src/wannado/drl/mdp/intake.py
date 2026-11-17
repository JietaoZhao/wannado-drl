"""Stage 0: RL suitability diagnosis, domain archetypes, and Problem-to-MDP auditing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wannado.drl.mdp.spec import (
    ActionSpaceSpec,
    MDPSpec,
    RewardTermSpec,
    StateFeatureSpec,
    SuitabilityDiagnosis,
    TerminationSpec,
)


def diagnose_rl_suitability(
    *,
    has_sequential_coupling: bool = True,
    known_exact_model_and_convex: bool = False,
    online_interaction_allowed: bool = True,
    num_agents: int = 1,
    is_autoregressive_sequence: bool = False,
    has_verifiable_outcome_reward: bool = False,
    has_human_demos: bool = False,
    has_human_intervention: bool = False,
) -> SuitabilityDiagnosis:
    """Assess whether RL is appropriate and identify the right decision paradigm before coding."""
    warnings: list[str] = []
    alternatives: list[str] = []

    if is_autoregressive_sequence and has_verifiable_outcome_reward:
        return SuitabilityDiagnosis(
            is_rl_appropriate=True,
            recommended_paradigm="group_relative",
            reasoning_plain=(
                "Your task generates multi-token/step sequences with a verifiable final scorer (like math, code, "
                "or tool-use chains). Critic-free Group-Relative RL (GRPO / DAPO / GSPO / ARPO) fits cleanly."
            ),
        )

    if not has_sequential_coupling:
        alternatives = [
            "Supervised regression/classification if ground-truth optimal labels exist",
            "Contextual Bandits if decisions affect only immediate reward without changing future state",
        ]
        warnings.append(
            "No sequential state transition detected: full DRL may be overkill and add unnecessary variance."
        )
        return SuitabilityDiagnosis(
            is_rl_appropriate=False,
            recommended_paradigm="contextual_bandit_or_supervised",
            reasoning_plain=(
                "Today's decision does not affect tomorrow's state. Try supervised learning or a single-step bandit "
                "first before committing to multi-step DRL."
            ),
            alternatives_if_not_rl=alternatives,
            warnings=warnings,
        )

    if known_exact_model_and_convex:
        alternatives = [
            "Model Predictive Control (MPC) / Convex Optimization (CVXPY/SciPy) as primary baseline",
            "Use DRL only if online solver latency is too slow or system dynamics are uncertain/nonlinear",
        ]
        warnings.append(
            "Known exact convex dynamics detected: always benchmark an MPC/convex solver baseline before claiming DRL wins."
        )

    if not online_interaction_allowed:
        return SuitabilityDiagnosis(
            is_rl_appropriate=True,
            recommended_paradigm="offline",
            reasoning_plain=(
                "Live trial-and-error is forbidden; you must learn purely from logged historical trajectories. "
                "Use conservative Offline RL (CQL / IQL) to avoid hallucinating optimistic actions."
            ),
            warnings=warnings,
        )

    if num_agents > 1:
        return SuitabilityDiagnosis(
            is_rl_appropriate=True,
            recommended_paradigm="multi_agent_ctde",
            reasoning_plain=(
                f"Detected {num_agents} coupled decision-makers. Use Multi-Agent RL under Centralized Training with "
                "Decentralized Execution (MAPPO / QMIX / VDN / MADDPG) so agents coordinate without non-stationarity collapse."
            ),
            warnings=warnings,
        )

    if has_human_demos or has_human_intervention:
        return SuitabilityDiagnosis(
            is_rl_appropriate=True,
            recommended_paradigm="off_policy",
            reasoning_plain=(
                "You have expert demonstrations or live human takeover signals. Off-policy RLPD / HIL-SERL "
                "can bootstrap rapidly with 50/50 demo-online replay instead of wasting millions of random steps."
            ),
            warnings=warnings,
        )

    return SuitabilityDiagnosis(
        is_rl_appropriate=True,
        recommended_paradigm="on_or_off_policy_online",
        reasoning_plain=(
            "Your decision changes future states sequentially and cannot be solved analytically at run-time. "
            "Standard online MDP reinforcement learning is a great fit."
        ),
        alternatives_if_not_rl=alternatives,
        warnings=warnings,
    )


def build_mdp_archetype(
    archetype: str = "custom_sim",
    *,
    problem_name: str | None = None,
) -> MDPSpec:
    """Return a structured starting MDPSpec template for common research & engineering scenarios."""
    key = archetype.lower()
    if key == "wireless_scheduling":
        return MDPSpec(
            problem_name=problem_name or "WirelessResourceScheduling",
            domain="wireless",
            goal_plain="Maximize long-term sum throughput / QoS satisfaction under power, queue, and interference constraints.",
            decision_interval_plain="One scheduling slot / TTI frame (e.g., 1 ms).",
            suitability=diagnose_rl_suitability(has_sequential_coupling=True),
            state_features=[
                StateFeatureSpec("queue_lengths", dim=4, scale_hint="[0, 1] normalized buffer occupancy"),
                StateFeatureSpec("past_channel_cqi", dim=4, scale_hint="CSI/SNR observed at slot start"),
                StateFeatureSpec("remaining_power_budget", dim=1, scale_hint="[0, 1]"),
            ],
            partial_observability=False,
            action_spec=ActionSpaceSpec(
                kind="discrete",
                dim=6,
                has_invalid_actions=True,
                safety_projection="action_mask over infeasible RB/power allocations",
                notes="Mask out allocations violating empty queues or hard power caps.",
            ),
            reward_terms=[
                RewardTermSpec("throughput_utility", "primary_objective", 1.0, (0.0, 2.0), "Compare against actual delivered bits KPI"),
                RewardTermSpec("queue_delay_penalty", "safety_penalty", -0.2, (-1.0, 0.0), "Avoid dropping packets to keep queue artificially low"),
            ],
            termination_spec=TerminationSpec(["queue_overflow_outage"], truncated_horizon=200, bootstrap_on_truncated=True),
            task_kpis=["mean_throughput_mbps", "outage_rate", "p95_queue_delay"],
            reward_hacking_risks=[
                "Dropping low-rate users permanently to maximize average throughput without fairness KPI",
                "Leaking future slot channel fading gain into current state observation",
            ],
            future_leakage_checklist=[
                "Verify channel_gain[t] uses ONLY causal CSI available at slot t (not post-transmission SINR[t+1])",
                "Verify traffic arrivals[t+1] are hidden until after action[t] is chosen",
            ],
            heuristic_baseline_idea="Proportional-Fair (PF) or Max-SNR greedy scheduler",
            gamma=0.95,
        )

    if key == "continuous_control":
        return MDPSpec(
            problem_name=problem_name or "ContinuousPhysicalControl",
            domain="control",
            goal_plain="Track target state smoothly while respecting actuator torque/velocity limits and minimizing energy.",
            decision_interval_plain="Fixed control dt (e.g., 20 ms / 50 Hz).",
            suitability=diagnose_rl_suitability(has_sequential_coupling=True),
            state_features=[
                StateFeatureSpec("positions_and_angles", dim=4, scale_hint="rad or meters, normalized"),
                StateFeatureSpec("velocities", dim=4, scale_hint="rad/s or m/s (needed for Markov property)"),
            ],
            partial_observability=False,
            action_spec=ActionSpaceSpec(
                kind="continuous",
                dim=2,
                bounds=(-1.0, 1.0),
                safety_projection="clip & rescale from [-1, 1] to physical torque limits",
            ),
            reward_terms=[
                RewardTermSpec("tracking_progress", "primary_objective", 1.0, (-1.0, 1.0), "Evaluate pure tracking error separately"),
                RewardTermSpec("action_energy_cost", "step_cost", -0.05, (-0.5, 0.0), "Avoid overpowering alive bonus"),
            ],
            termination_spec=TerminationSpec(["safety_bound_violated"], truncated_horizon=500, bootstrap_on_truncated=True),
            task_kpis=["tracking_rmse", "success_rate", "mean_control_effort"],
            reward_hacking_risks=[
                "If alive_bonus > 0 dominates, agent freezes near safe edge instead of solving task",
                "Mistaking truncated=True at step 500 for terminated=True corrupts Critic values near horizon",
            ],
            future_leakage_checklist=[
                "Ensure velocity/sensor readings only include delayed or current measurements, never next-step state",
            ],
            heuristic_baseline_idea="PD / LQR / proportional feedback controller",
            gamma=0.99,
        )

    if key == "multi_agent_coop":
        return MDPSpec(
            problem_name=problem_name or "CooperativeMultiAgentTask",
            domain="multi_agent",
            goal_plain="Coordinate N decentralized agents from local observations to optimize a shared team objective.",
            decision_interval_plain="Synchronous decision tick across all agents.",
            suitability=diagnose_rl_suitability(has_sequential_coupling=True, num_agents=3),
            state_features=[
                StateFeatureSpec("agent_local_obs", dim=6, scale_hint="per-agent local view"),
                StateFeatureSpec("global_state_for_critic", dim=18, scale_hint="training-only centralized state"),
            ],
            partial_observability=True,
            action_spec=ActionSpaceSpec(kind="multi_agent_discrete", dim=5, has_invalid_actions=False),
            reward_terms=[
                RewardTermSpec("team_completion_reward", "primary_objective", 1.0, (-1.0, 2.0)),
            ],
            termination_spec=TerminationSpec(["team_goal_achieved", "collision_failure"], truncated_horizon=100, bootstrap_on_truncated=True),
            task_kpis=["team_success_rate", "coordination_efficiency"],
            reward_hacking_risks=["Lazy-agent credit assignment where 1 agent works and others stay idle"],
            future_leakage_checklist=["Decentralized actors must NOT read global_state() during execution"],
            heuristic_baseline_idea="Independent greedy nearest-target assignment",
            gamma=0.99,
            num_agents=3,
        )

    if key == "reasoning_sequence":
        return MDPSpec(
            problem_name=problem_name or "VerifiableSequenceReasoning",
            domain="reasoning_sequence",
            goal_plain="Generate multi-step reasoning/action sequences maximizing verifiable final outcome accuracy.",
            decision_interval_plain="One token or reasoning/tool-call step per autoregressive generation tick.",
            suitability=diagnose_rl_suitability(
                has_sequential_coupling=True,
                is_autoregressive_sequence=True,
                has_verifiable_outcome_reward=True,
            ),
            state_features=[
                StateFeatureSpec("prompt_and_prefix_tokens", dim=16, scale_hint="discrete token IDs"),
            ],
            partial_observability=False,
            action_spec=ActionSpaceSpec(kind="autoregressive_sequence", dim=32, notes="Vocabulary size"),
            reward_terms=[
                RewardTermSpec("verifiable_outcome_score", "primary_objective", 1.0, (0.0, 1.0)),
                RewardTermSpec("overlong_soft_penalty", "shaping", -0.2, (-0.5, 0.0)),
            ],
            termination_spec=TerminationSpec(["eos_token_generated"], truncated_horizon=64, bootstrap_on_truncated=False),
            task_kpis=["exact_match_rate", "mean_response_length", "tool_efficiency"],
            reward_hacking_risks=[
                "Length inflation or repetitive padding tokens if outcome reward leaks length bias",
                "Entropy collapse in GRPO without Clip-Higher or KL constraint",
            ],
            future_leakage_checklist=["Verifier reward must use held-out/ground-truth evaluator without test leakage"],
            heuristic_baseline_idea="Greedy decoding (SFT / base policy without RL)",
            gamma=1.0,
        )

    if key == "offline_control":
        return MDPSpec(
            problem_name=problem_name or "OfflineStaticDatasetPolicy",
            domain="offline_rl",
            goal_plain="Extract a safe policy strictly better than behavior logs without live environment rollouts.",
            decision_interval_plain="Matches historical dataset logging step.",
            suitability=diagnose_rl_suitability(has_sequential_coupling=True, online_interaction_allowed=False),
            state_features=[StateFeatureSpec("logged_state", dim=8, scale_hint="standardized from train split")],
            partial_observability=False,
            action_spec=ActionSpaceSpec(kind="continuous", dim=2, bounds=(-1.0, 1.0)),
            reward_terms=[RewardTermSpec("logged_reward", "primary_objective", 1.0, (-2.0, 2.0))],
            termination_spec=TerminationSpec(["logged_terminal"], truncated_horizon=200, bootstrap_on_truncated=True),
            task_kpis=["normalized_offline_score", "ood_q_gap"],
            reward_hacking_risks=["Q-value overestimation on out-of-distribution (OOD) actions absent in dataset"],
            future_leakage_checklist=["Normalize observations using training dataset statistics ONLY"],
            heuristic_baseline_idea="Behavior Cloning (BC) on logged actions",
            gamma=0.99,
            is_offline_only=True,
        )

    if key in ("finance_portfolio_risk", "finance", "portfolio"):
        return MDPSpec(
            problem_name=problem_name or "FinancePortfolioRiskControl",
            domain="finance",
            goal_plain="Dynamically rebalance asset weights on the probability simplex maximizing risk-adjusted return (Sharpe/CVaR) net of transaction costs.",
            decision_interval_plain="One trading bar / rebalancing period (daily or intraday).",
            suitability=diagnose_rl_suitability(has_sequential_coupling=True),
            state_features=[
                StateFeatureSpec("lagged_returns_and_vol", dim=12, scale_hint="strictly causal rolling window"),
                StateFeatureSpec("current_portfolio_weights", dim=4, scale_hint="simplex in [0, 1], sum == 1"),
            ],
            partial_observability=True,
            action_spec=ActionSpaceSpec(
                kind="continuous",
                dim=4,
                bounds=(-1.0, 1.0),
                safety_projection="apply_simplex_projection / PortfolioSimplexWrapper (w >= 0, sum == 1)",
                notes="Wrap continuous Box actions via PortfolioSimplexWrapper so weights always sum to 1.",
            ),
            reward_terms=[
                RewardTermSpec("net_portfolio_log_return", "primary_objective", 1.0, (-0.1, 0.1), "Compare out-of-sample walk-forward wealth"),
                RewardTermSpec("turnover_and_cvar_penalty", "safety_penalty", -0.2, (-0.5, 0.0), "Control drawdown tail risk (CVaR) and slippage"),
            ],
            termination_spec=TerminationSpec(["margin_ruin_or_end_of_window"], truncated_horizon=252, bootstrap_on_truncated=True),
            task_kpis=["sharpe_ratio", "cvar_alpha", "max_drawdown", "portfolio_turnover"],
            reward_hacking_risks=[
                "Look-ahead price leakage or survivorship bias inflating backtest Sharpe",
                "Zero-turnover hold bias or excessive churn if transaction fees are omitted",
            ],
            future_leakage_checklist=[
                "Ensure features[t] use ONLY prices/indicators available before market close[t]",
                "Separate train/validation/test chronologically (never shuffle overlapping windows)",
            ],
            heuristic_baseline_idea="Equal-Weight (1/N) or Volatility-Parity Rebalancing",
            gamma=0.99,
        )

    if key in ("clinical_offline_safety", "clinical", "healthcare", "sepsis"):
        return MDPSpec(
            problem_name=problem_name or "ClinicalOfflineTreatmentSafety",
            domain="healthcare",
            goal_plain="Learn conservative discrete dosing/intervention policies from logged EHR trajectories with strict action masking and Off-Policy Evaluation (WIS/FQE).",
            decision_interval_plain="One clinical chart window / ICU step (e.g., 4 hours).",
            suitability=diagnose_rl_suitability(has_sequential_coupling=True, online_interaction_allowed=False),
            state_features=[
                StateFeatureSpec("patient_vitals_and_labs", dim=16, scale_hint="train-split standardized, missingness flags"),
                StateFeatureSpec("prior_treatment_history", dim=4, scale_hint="causal cumulative doses"),
            ],
            partial_observability=True,
            action_spec=ActionSpaceSpec(
                kind="discrete",
                dim=25,
                has_invalid_actions=True,
                safety_projection="action_mask over clinically contraindicated doses",
                notes="Attach `action_masks` and `next_action_masks` to `OfflineTransitionDataset.extra`.",
            ),
            reward_terms=[
                RewardTermSpec("survival_and_sofa_recovery", "primary_objective", 1.0, (-15.0, 15.0), "Audit alongside subgroup WIS/FQE"),
                RewardTermSpec("organ_toxicity_penalty", "safety_penalty", -0.5, (-5.0, 0.0), "Verify no excessive high-dose recommendations"),
            ],
            termination_spec=TerminationSpec(["icu_discharge_or_mortality"], truncated_horizon=48, bootstrap_on_truncated=True),
            task_kpis=["wis_return", "fqe_initial_value", "subgroup_cvar_05", "clinician_agreement_rate"],
            reward_hacking_risks=[
                "Confounding-by-indication or immortal-time bias causing OPE over-optimism",
                "Recommending rare unlogged extreme doses without CQL pessimism or action masks",
            ],
            future_leakage_checklist=[
                "Verify lab measurements carry timestamps <= decision step t (no retrospective discharge codes)",
                "Log `behavior_probs` and `group_labels` for WIS and demographic subgroup auditing",
            ],
            heuristic_baseline_idea="Clinician Logged Behavior Policy (BC / empirical mode)",
            gamma=0.99,
            is_offline_only=True,
        )

    if key in ("cyber_attack_defense", "cyber", "cybersecurity"):
        return MDPSpec(
            problem_name=problem_name or "CyberDefenseIntrusionResponse",
            domain="cybersecurity",
            goal_plain="Detect, isolate, and patch compromised network nodes over discrete/multi-discrete defense actions while keeping benign service availability high.",
            decision_interval_plain="One network monitoring / SOC incident response tick.",
            suitability=diagnose_rl_suitability(has_sequential_coupling=True),
            state_features=[
                StateFeatureSpec("host_compromise_alerts", dim=12, scale_hint="[0, 1] normalized IDS telemetry"),
                StateFeatureSpec("service_health_vector", dim=6, scale_hint="availability indicators"),
            ],
            partial_observability=True,
            action_spec=ActionSpaceSpec(
                kind="multi_discrete",
                dim=8,
                has_invalid_actions=True,
                safety_projection="action_mask over unreachable hosts or illegal isolation targets",
            ),
            reward_terms=[
                RewardTermSpec("service_uptime_and_containment", "primary_objective", 1.0, (-2.0, 2.0)),
                RewardTermSpec("false_positive_isolation_cost", "safety_penalty", -0.3, (-1.0, 0.0)),
            ],
            termination_spec=TerminationSpec(["crown_jewel_breach_or_mitigated"], truncated_horizon=100, bootstrap_on_truncated=True),
            task_kpis=["containment_rate", "mean_service_availability", "false_isolation_rate"],
            reward_hacking_risks=["Shutting down all hosts immediately to prevent lateral movement (kills service availability)"],
            future_leakage_checklist=["State alerts must come from causal IDS observations, not attacker ground truth state"],
            heuristic_baseline_idea="Rule-based IDS alert quarantine threshold",
            gamma=0.95,
        )

    # Default custom_sim / discrete_control
    return MDPSpec(
        problem_name=problem_name or "CustomDecisionEnvironment",
        domain=key,
        goal_plain="Learn a sequential policy maximizing long-term cumulative task performance.",
        decision_interval_plain="One discrete decision step.",
        suitability=diagnose_rl_suitability(has_sequential_coupling=True),
        state_features=[StateFeatureSpec("state_vector", dim=4, scale_hint="float32 finite")],
        partial_observability=False,
        action_spec=ActionSpaceSpec(kind="discrete", dim=2),
        reward_terms=[RewardTermSpec("task_reward", "primary_objective", 1.0, (-1.0, 1.0))],
        termination_spec=TerminationSpec(["task_done_or_failure"], truncated_horizon=200, bootstrap_on_truncated=True),
        task_kpis=["success_rate"],
        reward_hacking_risks=["Verify reward rises alongside actual task_kpis rather than farming per-step shaping"],
        future_leakage_checklist=["Confirm state_vector contains no t+1 targets or future labels"],
        heuristic_baseline_idea="Simple threshold / greedy rule baseline",
        gamma=0.99,
    )


@dataclass
class MDPAuditReport:
    """Result of auditing an MDPSpec before creating wrapper code or training."""

    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "recommendations": list(self.recommendations),
        }


def audit_mdp_spec(spec: MDPSpec) -> MDPAuditReport:
    """Validate Stage 0 MDP Spec against common DRL modeling failures."""
    warnings: list[str] = list(spec.suitability.warnings)
    errors: list[str] = []
    recommendations: list[str] = []

    if not spec.suitability.is_rl_appropriate:
        warnings.append(
            "Stage 0 suitability check suggests RL may NOT be necessary; justify sequential coupling or run non-RL baseline first."
        )

    if not spec.state_features:
        errors.append("MDPSpec.state_features cannot be empty.")
    for feat in spec.state_features:
        if not feat.observable_at_decision_time:
            errors.append(
                f"Future information leakage! State feature '{feat.name}' is marked observable_at_decision_time=False."
            )

    if spec.partial_observability:
        recommendations.append(
            "Because partial_observability=True, either stack past K observations in env_wrapper.py or pick `ppo_gru`."
        )

    if not spec.reward_terms:
        errors.append("MDPSpec.reward_terms cannot be empty.")
    else:
        max_abs = max(
            max(abs(r.expected_per_step_range[0]), abs(r.expected_per_step_range[1])) * abs(r.weight)
            for r in spec.reward_terms
        )
        if max_abs > 50.0:
            warnings.append(
                f"Large per-step reward scale (~{max_abs:.1f}): rescale rewards into roughly [-5, 5] to stabilize Critic TD targets."
            )
        if all(abs(r.expected_per_step_range[0]) < 1e-6 and abs(r.expected_per_step_range[1]) <= 1.0 for r in spec.reward_terms):
            recommendations.append("Monitor whether reward is overly sparse; add dense potential-based shaping or demo bootstrapping if needed.")

    if not spec.task_kpis:
        warnings.append(
            "No independent `task_kpis` declared. Declare at least 1 raw domain KPI so Stage 3 Auditor can detect Reward Hacking."
        )

    if not spec.termination_spec.bootstrap_on_truncated and spec.action_spec.kind != "autoregressive_sequence":
        warnings.append(
            "bootstrap_on_truncated=False on an infinite/truncated control MDP introduces severe time-limit value bias."
        )

    return MDPAuditReport(
        ok=len(errors) == 0,
        warnings=warnings,
        errors=errors,
        recommendations=recommendations,
    )
