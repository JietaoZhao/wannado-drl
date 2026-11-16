"""Formal Problem-to-MDP contract schemas (Stage 0 joint modeling artifact)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class StateFeatureSpec:
    """Specification of one observable state group with anti-leakage metadata."""

    name: str
    dim: int = 1
    scale_hint: str = "normalized ~[-1, 1]"
    observable_at_decision_time: bool = True
    notes: str = ""


@dataclass
class ActionSpaceSpec:
    """Specification of the agent's control interface and safety/validity rules."""

    kind: Literal[
        "discrete",
        "continuous",
        "multi_agent_discrete",
        "multi_agent_continuous",
        "autoregressive_sequence",
    ] = "discrete"
    dim: int = 2
    bounds: tuple[float, float] | None = None
    has_invalid_actions: bool = False
    safety_projection: str = "none"
    notes: str = ""


@dataclass
class RewardTermSpec:
    """One additive component of the reward function plus anti-hacking guardrails."""

    name: str
    role: Literal["primary_objective", "shaping", "safety_penalty", "step_cost"] = "primary_objective"
    weight: float = 1.0
    expected_per_step_range: tuple[float, float] = (-1.0, 1.0)
    anti_hacking_guard: str = ""


@dataclass
class TerminationSpec:
    """Explicit separation of genuine MDP terminal states vs artificial time-limit truncation."""

    terminated_conditions: list[str] = field(default_factory=lambda: ["goal_reached", "constraint_failure"])
    truncated_horizon: int = 200
    bootstrap_on_truncated: bool = True


@dataclass
class SuitabilityDiagnosis:
    """Stage 0 pre-RL sanity diagnosis answering 'Should we even use RL for this problem?'"""

    is_rl_appropriate: bool
    recommended_paradigm: str
    reasoning_plain: str
    alternatives_if_not_rl: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MDPSpec:
    """Complete joint MDP specification bridging business/physics problem and DRL algorithms."""

    problem_name: str
    domain: str
    goal_plain: str
    decision_interval_plain: str
    suitability: SuitabilityDiagnosis
    state_features: list[StateFeatureSpec]
    partial_observability: bool
    action_spec: ActionSpaceSpec
    reward_terms: list[RewardTermSpec]
    termination_spec: TerminationSpec
    task_kpis: list[str] = field(default_factory=list)
    reward_hacking_risks: list[str] = field(default_factory=list)
    future_leakage_checklist: list[str] = field(default_factory=list)
    heuristic_baseline_idea: str = "Greedy / constant-action domain baseline"
    gamma: float = 0.99
    has_offline_demos: bool = False
    has_human_intervention: bool = False
    is_offline_only: bool = False
    num_agents: int = 1
    sim_ms_per_step_hint: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json() + "\n", encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MDPSpec:
        return cls(
            problem_name=data["problem_name"],
            domain=data["domain"],
            goal_plain=data["goal_plain"],
            decision_interval_plain=data["decision_interval_plain"],
            suitability=SuitabilityDiagnosis(**data["suitability"]),
            state_features=[StateFeatureSpec(**s) for s in data.get("state_features", [])],
            partial_observability=bool(data.get("partial_observability", False)),
            action_spec=ActionSpaceSpec(**data["action_spec"]),
            reward_terms=[RewardTermSpec(**r) for r in data.get("reward_terms", [])],
            termination_spec=TerminationSpec(**data["termination_spec"]),
            task_kpis=list(data.get("task_kpis", [])),
            reward_hacking_risks=list(data.get("reward_hacking_risks", [])),
            future_leakage_checklist=list(data.get("future_leakage_checklist", [])),
            heuristic_baseline_idea=data.get("heuristic_baseline_idea", "Greedy baseline"),
            gamma=float(data.get("gamma", 0.99)),
            has_offline_demos=bool(data.get("has_offline_demos", False)),
            has_human_intervention=bool(data.get("has_human_intervention", False)),
            is_offline_only=bool(data.get("is_offline_only", False)),
            num_agents=int(data.get("num_agents", 1)),
            sim_ms_per_step_hint=data.get("sim_ms_per_step_hint"),
        )

    @classmethod
    def load(cls, path: str | Path) -> MDPSpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def format_formal_mdp(self) -> str:
        """Render publication-ready formal MDP tuple (S, A, R, gamma, Horizon) summary."""
        s_desc = ", ".join(f"{f.name} (dim={f.dim}, {f.scale_hint})" for f in self.state_features) or "obs_vec"
        r_desc = " + ".join(f"{r.weight:+.3g}*{r.name}[{r.role}]" for r in self.reward_terms) or "scalar_reward"
        lines = [
            f"### Formal MDP Specification — {self.problem_name}",
            f"- **Domain & Objective**: `{self.domain}` — {self.goal_plain}",
            f"- **Control Step**: {self.decision_interval_plain}",
            f"- **State Space $\\mathcal{{S}}$**: {s_desc} | `partial_observability={self.partial_observability}`",
            f"- **Action Space $\\mathcal{{A}}$**: `kind={self.action_spec.kind}`, `dim={self.action_spec.dim}`, "
            f"`bounds={self.action_spec.bounds}`, `action_mask={self.action_spec.has_invalid_actions}`",
            f"- **Reward Function $R(s, a, s')$**: `{r_desc}`",
            f"- **Discount & Horizon**: $\\gamma = {self.gamma}$, `truncated_horizon={self.termination_spec.truncated_horizon}`, "
            f"`terminated={self.termination_spec.terminated_conditions}`, `bootstrap_on_truncated={self.termination_spec.bootstrap_on_truncated}`",
            f"- **Audited Task KPIs**: {', '.join(self.task_kpis) if self.task_kpis else 'none declared'}",
        ]
        if self.reward_hacking_risks:
            lines.append("- **Reward Hacking Guards**: " + "; ".join(self.reward_hacking_risks))
        return "\n".join(lines)
