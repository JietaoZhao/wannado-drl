"""Stage 0 Problem-to-MDP brief, Stage 2 plain-language algorithm recommendation, parameter intuition, and hardware ETA."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from wannado.drl.audit.preflight import PreflightReport, preflight_env
from wannado.drl.catalog.build_catalog import get_taxonomy_tree, load_catalog, query_catalog
from wannado.drl.copilot.project_inspector import ProjectScan
from wannado.drl.core.utils import resolve_device
from wannado.drl.mdp.spec import MDPSpec

_FAMILY_LABELS: dict[str, str] = {
    "tabular_classic": "Tabular Classic Foundations (表格型经典入门)",
    "value_based": "Deep Value-Based Q-Learning (深度值函数 / DQN 系列)",
    "policy_gradient": "On-Policy & Recurrent Policy Gradient (策略梯度 / PPO / POMDP)",
    "actor_critic_off": "Continuous Off-Policy & Human-in-the-Loop (连续控制 / SAC / HIL-SERL)",
    "offline_rl": "Offline / Batch RL (离线历史数据策略优化)",
    "multi_agent": "Multi-Agent CTDE Coordination (多智能体协同 MARL)",
    "llm_and_discovered": "2024-2025 Sequence Reasoning & Meta-Discovered RL (大模型推理 / 元学习规则发现)",
}

_CLASSIC_BASELINES = {"monte_carlo", "sarsa", "q_learning", "dqn", "reinforce", "ddpg", "vdn", "cql", "grpo"}
_FRONTIER_SOTA = {"ppo_gru", "hil_serl", "mappo", "maddpg", "dapo", "gspo", "arpo", "discorl"}


def _get_package_version() -> str:
    pkg = sys.modules.get("wannado.drl")
    return str(getattr(pkg, "__version__", "1.0.2"))


@dataclass
class RecommendedAlgoSuite:
    """Recommended Baseline -> SOTA algorithm progression matched to MDP characteristics."""

    goal: str
    paradigm: str
    action_space: str
    algos: list[str]
    rationale: str
    lineage_summary: list[dict[str, str]]
    classic_baselines: list[str] = field(default_factory=list)
    standard_workhorses: list[str] = field(default_factory=list)
    frontier_sota: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeEstimate:
    """Hardware-calibrated multi-seed training + evaluation ETA forecast."""

    device: str
    sim_ms_per_step: float
    net_update_ms_per_batch: float
    total_timesteps_per_seed: int
    num_seeds: int
    per_algo_seconds: dict[str, float]
    total_seconds: float
    advisory: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_summary(self) -> str:
        lines = [
            f"[wannado-drl Runtime Forecast] device={self.device}, sim={self.sim_ms_per_step:.3f} ms/step, "
            f"steps/seed={self.total_timesteps_per_seed:,}, seeds={self.num_seeds}"
        ]
        for aid, sec in self.per_algo_seconds.items():
            lines.append(f"  - {aid:<16s}: ~{sec:.1f}s")
        lines.append(f"  Total Estimated Time: ~{self.total_seconds:.1f}s ({self.total_seconds / 60.0:.2f} min)")
        if self.advisory:
            lines.append(f"  Advisory: {self.advisory}")
        return "\n".join(lines)


def get_sota_lineage(
    *,
    family: str | None = None,
    action_space: str | None = None,
) -> list[dict[str, Any]]:
    """Trace pedagogical-to-frontier evolution chains across the 7 RL/DRL families."""
    cards = query_catalog(family=family, action_space=action_space)
    by_id = {c["id"]: c for c in load_catalog()}
    children_map: dict[str, list[str]] = {}
    for c in cards:
        pid = c.get("parent_id")
        if pid:
            children_map.setdefault(pid, []).append(c["id"])

    entries: list[dict[str, Any]] = []
    for c in cards:
        chain: list[str] = [c["id"]]
        cur = c
        seen = {c["id"]}
        while cur.get("parent_id") and cur["parent_id"] not in seen:
            pid = cur["parent_id"]
            seen.add(pid)
            chain.append(pid)
            cur = by_id.get(pid, {})
            if not cur:
                break
        chain.reverse()
        entries.append(
            {
                "id": c["id"],
                "name": c["name"],
                "family": c["family"],
                "paradigm": c["paradigm"],
                "tier": c["tier"],
                "year": c["citation"]["year"],
                "evolution_path": " -> ".join(chain),
                "diff_from_parent": c["diff_from_parent"],
                "successors_in_scope": children_map.get(c["id"], []),
                "module_path": c["module_path"],
            }
        )
    entries.sort(key=lambda item: (item["family"], item["year"], item["id"]))
    return entries


def recommend_algo_suite(
    *,
    mdp_spec: MDPSpec | None = None,
    action_space: str = "discrete",
    goal: str = "standard",
    partial_observability: bool = False,
    has_invalid_actions: bool = False,
    is_offline_only: bool = False,
    has_human_demos: bool = False,
    num_agents: int = 1,
    sim_ms_per_step: float | None = None,
) -> RecommendedAlgoSuite:
    """Match MDP constraints & simulator speed to a Baseline -> SOTA algorithm roster."""
    if mdp_spec is not None:
        action_space = mdp_spec.action_spec.kind
        partial_observability = mdp_spec.partial_observability
        has_invalid_actions = mdp_spec.action_spec.has_invalid_actions
        is_offline_only = mdp_spec.is_offline_only
        has_human_demos = mdp_spec.has_offline_demos or mdp_spec.has_human_intervention
        num_agents = mdp_spec.num_agents
        sim_ms_per_step = sim_ms_per_step if sim_ms_per_step is not None else mdp_spec.sim_ms_per_step_hint

    if goal not in ("learning", "sanity", "standard", "paper_sota"):
        raise ValueError("goal must be one of 'learning', 'sanity', 'standard', 'paper_sota'.")

    slow_sim = sim_ms_per_step is not None and sim_ms_per_step > 2.0

    if action_space == "autoregressive_sequence":
        paradigm = "group_relative"
        algos = ["grpo"] if goal == "sanity" else (["grpo", "dapo", "gspo"] if goal == "standard" else ["grpo", "dapo", "gspo", "arpo"])
        rationale = (
            "Verifiable sequence generation fits critic-free Group-Relative RL: start with GRPO baseline, "
            "add DAPO (Clip-Higher + Dynamic Sampling) to prevent entropy collapse, GSPO for sequence-level ratio alignment, and ARPO for tool-branching agents."
        )
    elif is_offline_only:
        paradigm = "offline"
        algos = ["iql_offline"] if goal == "sanity" else ["cql", "iql_offline"]
        rationale = (
            "Zero live interaction allowed: compare conservative Q-penalization (CQL) against in-sample expectile regression (IQL) "
            "to avoid out-of-distribution action hallucination."
        )
    elif num_agents > 1 or action_space.startswith("multi_agent"):
        paradigm = "multi_agent_ctde"
        if action_space == "multi_agent_continuous":
            algos = ["mappo"] if goal == "sanity" else ["mappo", "maddpg"]
        else:
            algos = ["mappo"] if goal == "sanity" else (["qmix", "mappo"] if goal == "standard" else ["vdn", "qmix", "mappo"])
        rationale = (
            "Multi-agent coordination uses CTDE: pair value factorization (VDN -> QMIX) with centralized-critic MAPPO (or MADDPG for continuous teams)."
        )
    elif goal == "learning" and action_space == "discrete":
        paradigm = "tabular_to_deep"
        algos = ["monte_carlo", "sarsa", "q_learning", "dqn"]
        rationale = (
            "Pedagogical progression: see full-return Monte Carlo vs on-policy TD (SARSA) vs off-policy Bellman optimality (Q-Learning) before scaling to neural DQN."
        )
    elif action_space == "continuous":
        paradigm = "off_policy" if (slow_sim or has_human_demos) else "on_and_off_policy"
        if has_human_demos:
            algos = ["sac", "hil_serl"]
            rationale = "Human demos/interventions available: pair SAC baseline with HIL-SERL (50/50 demo-online RLPD + LayerNorm critics)."
        elif goal == "sanity":
            algos = ["sac"] if slow_sim else ["ppo"]
            rationale = "Fast smoke test matched to simulator speed."
        elif goal == "standard":
            algos = ["td3", "sac"] if slow_sim else ["ppo", "td3", "sac"]
            rationale = (
                "Compare on-policy stability (PPO) against sample-efficient off-policy replay reuse (TD3 deterministic vs SAC maximum-entropy)."
            )
        else:
            algos = ["ppo", "ddpg", "td3", "sac"]
            rationale = "Full continuous lineage: PPO vs DDPG -> TD3 -> SAC across multiple random seeds."
    else:
        # Discrete single-agent
        paradigm = "on_policy" if not slow_sim else "off_policy"
        if partial_observability:
            algos = ["ppo_gru"] if goal == "sanity" else ["ppo", "ppo_gru"]
            rationale = "Partial observability detected: compare memoryless PPO against recurrent PPO-GRU to verify hidden-state inference gain."
        elif has_invalid_actions:
            algos = ["ppo"] if goal == "sanity" else ["double_dqn", "ppo"]
            rationale = "Invalid action masking present: masked Double DQN vs Action-Masked PPO."
        elif goal == "sanity":
            algos = ["dqn"] if slow_sim else ["ppo"]
            rationale = "Quick smoke test verifying reward signal and learning loop."
        elif goal == "standard":
            algos = ["double_dqn", "ppo"] if not slow_sim else ["dqn", "double_dqn", "dueling_dqn"]
            rationale = "Compare sample-efficient off-policy replay (Double DQN) against rock-solid on-policy PPO."
        else:
            algos = ["q_learning", "dqn", "double_dqn", "dueling_dqn", "c51", "a2c", "ppo"]
            rationale = "Full discrete taxonomy comparison from tabular Q-learning through C51 & PPO."

    by_id = {c["id"]: c for c in load_catalog()}
    lineage = [
        {
            "id": aid,
            "name": by_id[aid]["name"],
            "tier": by_id[aid]["tier"],
            "on_policy": str(by_id[aid]["on_policy"]),
            "role": by_id[aid]["summary_plain"],
        }
        for aid in algos
        if aid in by_id
    ]

    if action_space == "autoregressive_sequence":
        classic_pool, workhorse_pool, frontier_pool = ["reinforce", "grpo"], ["grpo"], ["dapo", "gspo", "arpo"]
    elif is_offline_only:
        classic_pool, workhorse_pool, frontier_pool = ["cql"], ["iql_offline"], ["hil_serl"]
    elif num_agents > 1 or action_space.startswith("multi_agent"):
        classic_pool = ["vdn"] if action_space != "multi_agent_continuous" else ["ddpg"]
        workhorse_pool = ["qmix"] if action_space != "multi_agent_continuous" else ["maddpg"]
        frontier_pool = ["mappo", "maddpg"]
    elif action_space == "continuous":
        classic_pool = ["reinforce", "ddpg"]
        workhorse_pool = ["ppo", "td3", "sac"]
        frontier_pool = ["hil_serl", "ppo_gru", "discorl"]
    else:
        classic_pool = ["monte_carlo", "q_learning", "dqn"]
        workhorse_pool = ["double_dqn", "dueling_dqn", "ppo"]
        frontier_pool = ["ppo_gru", "discorl"]

    return RecommendedAlgoSuite(
        goal=goal,
        paradigm=paradigm,
        action_space=action_space,
        algos=algos,
        rationale=rationale,
        lineage_summary=lineage,
        classic_baselines=[a for a in classic_pool if a in by_id],
        standard_workhorses=[a for a in workhorse_pool if a in by_id],
        frontier_sota=[a for a in frontier_pool if a in by_id],
    )


def build_arsenal_showcase() -> dict[str, Any]:
    """Build the versioned wannado-drl algorithm arsenal showcase (Classics -> Standard -> 2025 Frontier SOTA)."""
    ver = _get_package_version()
    tree = get_taxonomy_tree()
    cards = load_catalog()

    families: list[dict[str, Any]] = []
    classics: list[str] = []
    standards: list[str] = []
    frontiers: list[str] = []

    for fam, fam_cards in tree.items():
        fam_items = []
        for c in fam_cards:
            aid = c["id"]
            if aid in _CLASSIC_BASELINES:
                era = "classic"
                classics.append(aid)
            elif aid in _FRONTIER_SOTA:
                era = "frontier"
                frontiers.append(aid)
            else:
                era = "standard"
                standards.append(aid)
            fam_items.append(
                {
                    "id": aid,
                    "name": c["name"],
                    "year": c["citation"]["year"],
                    "tier": c["tier"],
                    "era": era,
                    "action_spaces": c["action_spaces"],
                    "summary_plain": c["summary_plain"],
                }
            )
        families.append(
            {
                "family": fam,
                "label": _FAMILY_LABELS.get(fam, fam),
                "algorithms": fam_items,
            }
        )

    return {
        "version": ver,
        "headline": f"As of wannado-drl v{ver}, {len(cards)} single-file DRL algorithms across {len(tree)} families are built in and verified.",
        "total_algorithms": len(cards),
        "total_families": len(tree),
        "era_breakdown": {
            "classic_foundations": classics,
            "workhorse_standards": standards,
            "frontier_sota": frontiers,
        },
        "families": families,
    }


def format_arsenal_overview() -> str:
    """Format the versioned algorithm arsenal table ready for user-facing display."""
    arsenal = build_arsenal_showcase()
    lines = [
        f"## wannado-drl v{arsenal['version']} Reinforcement Learning Arsenal ({arsenal['total_algorithms']} Algorithms / {arsenal['total_families']} Families)",
        "",
        f"- **Classic Foundations ({len(arsenal['era_breakdown']['classic_foundations'])})**: `{'`, `'.join(arsenal['era_breakdown']['classic_foundations'])}`",
        f"- **Proven Workhorses ({len(arsenal['era_breakdown']['workhorse_standards'])})**: `{'`, `'.join(arsenal['era_breakdown']['workhorse_standards'])}`",
        f"- **2022-2025 Frontier SOTA ({len(arsenal['era_breakdown']['frontier_sota'])})**: `{'`, `'.join(arsenal['era_breakdown']['frontier_sota'])}`",
        "",
        "| Family | Algorithms (Chronological Classic → Frontier SOTA) | Key Highlights |",
        "| :--- | :--- | :--- |",
    ]
    for fam in arsenal["families"]:
        algo_tags = ", ".join(f"`{a['id']}` ({a['year']})" for a in fam["algorithms"])
        newest = fam["algorithms"][-1]
        lines.append(f"| {fam['label']} | {algo_tags} | Latest: `{newest['id']}` — {newest['summary_plain']} |")
    return "\n".join(lines)


def build_mdp_intake_brief(
    project_scan: ProjectScan | None = None,
    *,
    archetype_hint: str | None = None,
) -> dict[str, Any]:
    """Build Stage 0 Problem-to-MDP plain-language questions before touching algorithms."""
    mode = project_scan.integration_mode if project_scan is not None else "raw_simulator_needs_mdp"
    arch = archetype_hint or (project_scan.likely_archetype if project_scan is not None else "custom_sim")
    return {
        "stage": "Stage 0 — Problem-to-MDP Joint Modeling",
        "detected_integration_mode": mode,
        "suggested_archetype": arch,
        "first_question": "First confirm: does your decision today change tomorrow's system state, or is it a single-step prediction/convex problem?",
        "mdp_questions": [
            {
                "topic": "decision_cycle_and_goal",
                "prompt": "What are you optimizing, and how often does the agent take one decision?",
                "plain_analogy": "Think of setting the game frame rate: does the controller act every millisecond, every scheduling slot, or once per customer order?",
                "pitfall_warning": "If each step only sees independent inputs with zero carryover state, a Contextual Bandit or Supervised Learning beats DRL.",
            },
            {
                "topic": "state_and_leakage",
                "prompt": "What exact measurements can the agent legally observe right BEFORE choosing its action?",
                "plain_analogy": "Think of a driver looking ONLY through the front windshield and dashboard—never peeking at tomorrow's weather or future prices.",
                "pitfall_warning": "Never put next-step targets/future CSI into `obs` (Future Information Leakage), and tell us if key speeds/histories are missing (POMDP -> use PPO-GRU).",
            },
            {
                "topic": "action_and_safety",
                "prompt": "What knobs can the agent turn, and are certain combinations forbidden/physically unsafe?",
                "plain_analogy": "Discrete switches (pick 1 of K channels) vs smooth dials (steering/torque), plus safety guardrails.",
                "pitfall_warning": "Use `action_mask` for invalid discrete choices instead of giving huge -1000 penalties that ruin Critic learning.",
            },
            {
                "topic": "terminated_vs_truncated",
                "prompt": "When does an episode genuinely succeed/crash (`terminated`) vs merely hit a clock limit (`truncated`)?",
                "plain_analogy": "Game-over crash (`terminated=True`, future value = 0) vs pressing pause at round 200 (`truncated=True`, life would continue!).",
                "pitfall_warning": "Treating time-limit cutoffs as `terminated=True` tricks the Critic into expecting sudden zero reward at the horizon.",
            },
            {
                "topic": "reward_and_anti_hacking",
                "prompt": "How do we score good behavior, and what raw domain KPI proves the AI isn't gaming the score?",
                "plain_analogy": "Paying a delivery driver per mile might make them drive in circles; tracking actual parcels delivered (`task_metrics`) catches the cheat!",
                "pitfall_warning": "Always return `info['reward_terms']` and `info['task_metrics']` in `env_wrapper.py` so Stage 3 Auditor detects Reward Hacking.",
            },
        ],
    }


def _knob(
    *,
    parameter: str,
    terminology: str,
    plain_analogy: str,
    equivalent_to: str,
    if_higher_vs_lower: str,
    options: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "parameter": parameter,
        "terminology": terminology,
        "plain_analogy": plain_analogy,
        "equivalent_to": equivalent_to,
        "if_higher_vs_lower": if_higher_vs_lower,
        "options": options,
    }


def explain_parameters(
    *,
    algos: Sequence[str] | None = None,
    action_space: str = "discrete",
    mdp_spec: MDPSpec | None = None,
    preflight_report: PreflightReport | None = None,
    output_dir: str = "./drl_lab/outputs",
) -> dict[str, Any]:
    """Return zero-jargon analogies, equivalents, trade-offs, and storage options tailored to the active DRL algos."""
    algo_set = set(algos or ())
    eff_action = mdp_spec.action_spec.kind if mdp_spec is not None else (
        preflight_report.action_space_kind if preflight_report is not None else action_space
    )
    horizon_hint = (
        int(round(preflight_report.summary["mean_episode_length"]))
        if preflight_report is not None and preflight_report.summary.get("mean_episode_length")
        else (mdp_spec.termination_spec.truncated_horizon if mdp_spec is not None else 200)
    )
    horizon_hint = max(1, horizon_hint)

    is_seq = eff_action == "autoregressive_sequence" or bool(algo_set & {"grpo", "dapo", "gspo", "arpo"})
    is_offline = (mdp_spec is not None and mdp_spec.is_offline_only) or bool(algo_set & {"cql", "iql_offline"})
    uses_epsilon = bool(algo_set & {"monte_carlo", "sarsa", "q_learning", "dqn", "double_dqn", "dueling_dqn", "c51"}) or (
        not algo_set and eff_action == "discrete"
    )
    uses_ppo = bool(algo_set & {"ppo", "ppo_gru", "a2c", "trpo", "mappo"}) or not algo_set
    uses_hil = "hil_serl" in algo_set

    est_small_ep = max(1, 5_000 // horizon_hint)
    est_dev_ep = max(1, 25_000 // horizon_hint)
    est_paper_ep = max(1, 100_000 // horizon_hint)

    total_timesteps_and_seeds = _knob(
        parameter="episodes / total_timesteps & num_seeds",
        terminology="Episode (回合/局), Timestep (决策步), Random Seed (随机种子)",
        plain_analogy=(
            "Think of 1 Episode as playing one complete match from start to finish, 1 Timestep as making one single move inside that match, "
            "and Random Seeds as repeating the whole training 3-5 times from scratch to prove good scores aren't a fluke."
        ),
        equivalent_to=f"At your current ~{horizon_hint} steps/episode horizon, 25,000 steps ≈ {est_dev_ep} complete episodes.",
        if_higher_vs_lower="1 seed or very few episodes finishes fast but risks lucky noise; >=3 seeds lets us report statistically rigorous IQM & mean ± std.",
        options=[
            {
                "timesteps": 5_000,
                "episodes_approx": est_small_ep,
                "seeds": 1,
                "label": f"Quick pipeline smoke test (~{est_small_ep} episodes / 5k steps x 1 seed) — verifies wiring & beats Random floor",
            },
            {
                "timesteps": 25_000,
                "episodes_approx": est_dev_ep,
                "seeds": 3,
                "label": f"Reliable development run (~{est_dev_ep} episodes / 25k steps x 3 seeds) — computes IQM & catches fake convergence [Recommended]",
            },
            {
                "timesteps": 100_000,
                "episodes_approx": est_paper_ep,
                "seeds": 5,
                "label": f"Paper benchmark scale (~{est_paper_ep} episodes / 100k+ steps x 5 seeds) — top-venue statistical rigor",
            },
        ],
    )

    gamma_guide = _knob(
        parameter="gamma",
        terminology="Discount Factor (折扣因子 gamma)",
        plain_analogy="How far ahead the agent cares—like choosing between instant pocket money today vs investing for next month.",
        equivalent_to="Effective planning horizon is roughly 1 / (1 - gamma) steps (e.g. gamma=0.99 ≈ 100 steps ahead).",
        if_higher_vs_lower="Too low (0.80) makes the agent short-sighted; too close to 1.0 (0.999) makes value estimates noisy on long episodes.",
        options=[
            {"value": 0.90, "label": "Short-horizon focus (~10-step lookahead, gamma=0.90) — fast reactive slot decisions"},
            {"value": 0.99, "label": "Balanced long-horizon planning (~100-step lookahead, gamma=0.99) [Recommended]"},
            {"value": 0.995, "label": "Far-sighted planning (~200+ steps lookahead, gamma=0.995) — slow delayed rewards"},
        ],
    )

    ent_guide = _knob(
        parameter="ent_coef",
        terminology="Entropy Bonus Coefficient (策略熵鼓励系数 ent_coef)",
        plain_analogy="Curiosity / willingness to keep trying new moves instead of locking onto the first mediocre habit.",
        equivalent_to="A small bonus reward added simply for keeping action probabilities diverse rather than 100% deterministic too early.",
        if_higher_vs_lower="Too low (0.0) risks premature convergence to a bad local trick; too high keeps the agent acting almost randomly.",
        options=[
            {"value": 0.0, "label": "Pure greedy exploitation (ent_coef=0.0) — fine-tuning already-strong policies"},
            {"value": 0.01, "label": "Gentle curiosity nudge (ent_coef=0.01) — standard PPO balance [Recommended]"},
            {"value": 0.05, "label": "Strong exploration boost (ent_coef=0.05) — deceptive local optima"},
        ],
    )

    storage_guide = _knob(
        parameter="output_dir & workspace isolation",
        terminology="Isolated Lab Directory & Artifact Output Path (非侵入工作区与数据保存路径)",
        plain_analogy=(
            "Think of keeping all RL plumbing in a separate lab notebook folder (`./drl_lab/`) so your original simulator/business code stays 100% clean, "
            "while saving learning curves, charts, LaTeX tables, and BibTeX into a dedicated results folder."
        ),
        equivalent_to=f"Harness lives in `./drl_lab/`; reports, curves (`learning_curves.csv`), PNG/SVG charts, and `.bib` are saved in `{output_dir}`.",
        if_higher_vs_lower="Keeping outputs per experiment subfolder preserves old runs for ablation comparisons instead of overwriting previous charts.",
        options=[
            {"path": output_dir, "label": f"Standard isolated folder `{output_dir}` (auto-saves report `.md`, charts, `.tex`, `.bib`, CSV) [Recommended]"},
            {"path": f"{output_dir}/exp_latest", "label": f"Timestamped/named subfolder `{output_dir}/exp_latest` — keeps historical runs side-by-side"},
        ],
    )

    guide: dict[str, Any] = {
        "paradigm_intuition": {
            "on_policy_vs_off_policy": (
                "On-policy (`ppo`, `a2c`, `grpo`) learns ONLY from fresh rollouts by the current policy—like practicing and immediately tossing old notes "
                "(super stable, loves fast simulators). Off-policy (`dqn`, `double_dqn`, `sac`, `td3`, `hil_serl`) saves past experience in a Replay Buffer and reuses it many times "
                "(great when simulation steps are slow or costly)."
            ),
        },
        "gamma": gamma_guide,
        "ent_coef": ent_guide,
        "total_timesteps_and_seeds": total_timesteps_and_seeds,
        "storage_and_outputs": storage_guide,
        "algorithm_specific": {},
        "terminology_glossary": {
            "Episode vs Step": "One Episode is a complete task round (start to done); one Step is a single decision tick inside that round.",
            "On-policy vs Off-policy": "Fresh practice only (`ppo`/`grpo`) vs saving memories in a Replay Buffer and re-studying them (`dqn`/`sac`/`hil_serl`).",
            "Terminated vs Truncated": "True crash/win game-over (`terminated=True`, future value=0) vs hitting a clock limit (`truncated=True`, life continues).",
            "IQM (Interquartile Mean)": "Drops the top 25% luckiest and bottom 25% worst random seeds, averaging the middle 50% so results aren't cherry-picked.",
        },
    }

    if uses_epsilon:
        guide["algorithm_specific"]["epsilon_exploration"] = _knob(
            parameter="eps_start / eps_end / exploration_fraction",
            terminology="Epsilon-Greedy Exploration Rate (贪心探索率 epsilon)",
            plain_analogy=(
                "Like choosing restaurants in a new city: at the start (`eps_start=1.0`) you pick randomly to discover what exists; "
                "later (`eps_end=0.05`) you eat at the highest-rated place 95% of the time and save 5% randomness to keep checking for hidden gems."
            ),
            equivalent_to="Probability that DQN / Double-DQN / Q-Learning overrides its current best Q-value to take a random legal action.",
            if_higher_vs_lower="Decaying epsilon too fast stops discovering better strategies; keeping epsilon too high wastes steps on random moves late in training.",
            options=[
                {"eps_start": 1.0, "eps_end": 0.05, "exploration_fraction": 0.4, "label": "Standard balanced exploration (100% -> 5% over first 40% of training) [Recommended]"},
                {"eps_start": 1.0, "eps_end": 0.02, "exploration_fraction": 0.25, "label": "Fast exploit (100% -> 2% over 25%) — simple tasks or warm-started Q-tables"},
                {"eps_start": 1.0, "eps_end": 0.10, "exploration_fraction": 0.65, "label": "Deep search (100% -> 10% over 65%) — sparse rewards or deceptive traps"},
            ],
        )

    if uses_ppo:
        guide["algorithm_specific"]["clip_coef"] = _knob(
            parameter="clip_coef",
            terminology="PPO Clip Ratio / Policy Safety Leash (策略更新安全绳 clip_coef)",
            plain_analogy="Like putting a safety leash on how drastically the agent can rewrite its habits after seeing just one noisy batch.",
            equivalent_to="Restricts new/old action probability ratio inside [1 - clip_coef, 1 + clip_coef] (Engstrom et al. / Schulman et al.).",
            if_higher_vs_lower="Too small updates crawl slowly; too large risks collapsing a working policy overnight.",
            options=[
                {"value": 0.2, "label": "Standard PPO clip leash (clip_coef=0.2) — rock-solid default across most benchmarks [Recommended]"},
                {"value": 0.1, "label": "Extra conservative leash (clip_coef=0.1) — ultra-noisy continuous or sensitive control"},
            ],
        )

    if is_seq:
        guide["algorithm_specific"]["group_size_and_clip_high"] = _knob(
            parameter="group_size & clip_high",
            terminology="Group Rollout Size & Clip-Higher (组内候选数量与高上限裁剪)",
            plain_analogy="For each prompt, let the model brainstorm K independent solution drafts (`group_size`) and grade them against each other—no Critic model needed!",
            equivalent_to="Larger K lowers advantage variance per prompt; DAPO `clip_high > clip_low` prevents rare reasoning tokens from being silenced.",
            if_higher_vs_lower="Too small K makes relative ranking noisy; larger K multiplies generation compute linearly.",
            options=[
                {"group_size": 4, "clip_high": 0.28, "label": "Light reasoning check (4 drafts/prompt + DAPO clip_high=0.28)"},
                {"group_size": 8, "clip_high": 0.28, "label": "Balanced reasoning SOTA (8 drafts/prompt + DAPO clip_high=0.28) [Recommended]"},
            ],
        )

    if uses_hil:
        guide["algorithm_specific"]["demo_fraction_and_utd"] = _knob(
            parameter="demo_fraction & utd_ratio",
            terminology="RLPD Symmetric Demo Ratio & Update-to-Data Ratio (人机演示混合比例与高频回放率)",
            plain_analogy="Every study batch mixes half expert/human intervention memories and half live robot trials, practicing multiple gradient updates per real step.",
            equivalent_to="`demo_fraction=0.5` samples 50% human/demo transitions + 50% online transitions; `utd_ratio` sets updates per env step.",
            if_higher_vs_lower="Higher `utd_ratio` learns fast from few real-world trials but needs LayerNorm critics (`Ba et al.`) to stay stable.",
            options=[
                {"demo_fraction": 0.5, "utd_ratio": 4, "label": "HIL-SERL standard (50/50 RLPD replay + UTD=4 + LayerNorm critics) [Recommended]"},
                {"demo_fraction": 0.5, "utd_ratio": 8, "label": "Ultra sample-scarce hardware mode (50/50 RLPD + UTD=8)"},
            ],
        )

    if is_offline:
        guide["algorithm_specific"]["offline_conservatism"] = _knob(
            parameter="cql_alpha / expectile_tau",
            terminology="Offline Pessimism Coefficient (离线保守防幻觉系数)",
            plain_analogy="When learning strictly from old logs without trying live moves, penalize over-optimism on untested actions so the agent stays grounded.",
            equivalent_to="Higher `cql_alpha` (CQL) keeps policy closer to behavior logs; `expectile_tau` (IQL) tilts value toward top recorded outcomes cleanly.",
            if_higher_vs_lower="Too weak hallucinates high Q-values on missing actions; too strong refuses to improve beyond old behavior.",
            options=[
                {"cql_alpha": 1.0, "expectile_tau": 0.7, "label": "Balanced offline safety (CQL alpha=1.0 / IQL tau=0.7) [Recommended]"},
                {"cql_alpha": 5.0, "expectile_tau": 0.6, "label": "Strict high-safety conservatism (stay very close to logged data)"},
            ],
        )

    return guide


def build_goal_confirmation_dialog(project_scan: ProjectScan | None = None) -> dict[str, Any]:
    """Build the mandatory two-step interactive goal-alignment dialog for AI IDEs."""
    conf = project_scan.understanding_confidence if project_scan is not None else "low"
    summary = (
        project_scan.project_summary_guess
        if project_scan is not None and project_scan.project_summary_guess
        else "No local project directory was scanned yet; I need to align with you on what system you are building."
    )
    inferred_goal = (
        project_scan.inferred_drl_goal
        if project_scan is not None and project_scan.inferred_drl_goal
        else "Optimize a sequential decision or control problem using Deep Reinforcement Learning."
    )
    clues = list(project_scan.evidence_clues) if project_scan is not None else []
    opt1_label = (
        f"Yes — use DRL to: {inferred_goal} (Recommended)"
        if conf != "low"
        else f"Rough framing fits — explore: {inferred_goal}"
    )
    return {
        "must_explain_understanding_before_asking": True,
        "ai_understanding_summary": summary,
        "understanding_confidence": conf,
        "evidence_clues": clues,
        "dialog_1_confirm_goal": {
            "question": "Based on reading your project, is this what you want Deep Reinforcement Learning (DRL) to solve?",
            "options": [
                {"id": "confirm_inferred_goal", "label": opt1_label, "is_recommended": conf != "low"},
                {
                    "id": "custom_one_sentence_goal",
                    "label": "No / Not quite — let me describe in one sentence what I want DRL to do here",
                    "is_recommended": conf == "low",
                },
            ],
        },
        "dialog_2_one_sentence_prompt": {
            "trigger": "Show/ask when Option 2 is picked or understanding_confidence == 'low'.",
            "prompt": (
                "In one sentence, what would you like the reinforcement learning agent to optimize or control in this project "
                "(e.g., what decisions it makes each step and what success looks like)?"
            ),
        },
    }


def estimate_runtime(
    env_or_factory: Any | None = None,
    algos: list[str] | RecommendedAlgoSuite = ("ppo",),
    *,
    preflight_report: PreflightReport | None = None,
    total_timesteps: int = 20_000,
    num_seeds: int = 3,
    device: str = "auto",
) -> RuntimeEstimate:
    """Forecast multi-seed DRL training ETA from measured ms/step and neural update micro-benchmark."""
    algo_list = list(algos.algos) if isinstance(algos, RecommendedAlgoSuite) else list(algos)
    dev = resolve_device(device)

    if preflight_report is not None:
        sim_ms = preflight_report.ms_per_step
        obs_dim = max(4, preflight_report.obs_dim)
    elif env_or_factory is not None:
        pf = preflight_env(env_or_factory, probe_steps=30, baseline_episodes=2)
        sim_ms = pf.ms_per_step
        obs_dim = max(4, pf.obs_dim)
    else:
        sim_ms = 0.08
        obs_dim = 8

    # Micro-benchmark a tiny MLP forward+backward update on target hardware (after warmup)
    probe_net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 4)).to(dev)
    probe_x = torch.randn(64, obs_dim, device=dev)
    _ = probe_net(probe_x).sum()
    t0 = time.perf_counter()
    for _ in range(5):
        loss = probe_net(probe_x).pow(2).mean()
        probe_net.zero_grad()
        loss.backward()
    if dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    upd_ms = ((time.perf_counter() - t0) * 1000.0) / 5.0

    by_id = {c["id"]: c for c in load_catalog()}
    per_algo: dict[str, float] = {}
    for aid in algo_list:
        card = by_id.get(aid, {})
        if card.get("family") == "tabular_classic":
            # Tabular Monte Carlo / SARSA / Q-Learning perform zero neural backward passes.
            sec_per_seed = (total_timesteps * (sim_ms + 0.015)) / 1000.0
        else:
            is_off = not card.get("on_policy", True)
            # Off-policy runs gradient updates almost every step; on-policy runs epoch updates per rollout batch.
            grad_updates_per_env_step = 1.0 if is_off else 0.15
            if aid in ("hil_serl", "trpo"):
                grad_updates_per_env_step = 1.8
            sec_per_seed = (total_timesteps * (sim_ms + grad_updates_per_env_step * upd_ms * 0.4)) / 1000.0
        per_algo[aid] = round(max(0.2, sec_per_seed * num_seeds), 2)

    total_sec = round(sum(per_algo.values()), 2)
    if total_sec > 300:
        advisory = f"Heavy multi-seed run (~{total_sec / 60:.1f} min); verify 1-seed sanity at 5k steps first."
    elif total_sec > 60:
        advisory = "Moderate runtime (~1-5 min); well-calibrated for development comparison."
    else:
        advisory = "Fast runtime (<1 min); safe to launch immediately."

    return RuntimeEstimate(
        device=str(dev),
        sim_ms_per_step=round(sim_ms, 4),
        net_update_ms_per_batch=round(upd_ms, 3),
        total_timesteps_per_seed=total_timesteps,
        num_seeds=num_seeds,
        per_algo_seconds=per_algo,
        total_seconds=total_sec,
        advisory=advisory,
    )


def build_interactive_brief(
    *,
    project_scan: ProjectScan | None = None,
    mdp_spec: MDPSpec | None = None,
    preflight_report: PreflightReport | None = None,
    goal: str = "standard",
    num_seeds: int = 3,
    total_timesteps: int = 20_000,
    output_dir: str = "./drl_lab/outputs",
) -> dict[str, Any]:
    """Assemble full Stage 0-2 AI Co-Pilot brief (understanding, arsenal, dialogs, knobs, storage & ETA)."""
    intake = build_mdp_intake_brief(project_scan=project_scan)
    action_sp = "discrete"
    if mdp_spec is not None:
        action_sp = mdp_spec.action_spec.kind
    elif preflight_report is not None:
        action_sp = preflight_report.action_space_kind
    elif project_scan is not None:
        if project_scan.integration_mode == "multi_agent_env":
            action_sp = "multi_agent_discrete"
        elif project_scan.integration_mode == "reasoning_sequence_task":
            action_sp = "autoregressive_sequence"
        elif project_scan.likely_archetype == "continuous_control":
            action_sp = "continuous"

    suite = recommend_algo_suite(
        mdp_spec=mdp_spec,
        action_space=action_sp,
        goal=goal,
        is_offline_only=(project_scan is not None and project_scan.integration_mode == "offline_dataset")
        if mdp_spec is None
        else mdp_spec.is_offline_only,
        sim_ms_per_step=preflight_report.ms_per_step if preflight_report is not None else None,
    )
    param_guide = explain_parameters(
        algos=suite.algos,
        action_space=action_sp,
        mdp_spec=mdp_spec,
        preflight_report=preflight_report,
        output_dir=output_dir,
    )
    eta = estimate_runtime(
        preflight_report=preflight_report,
        algos=suite,
        total_timesteps=total_timesteps,
        num_seeds=num_seeds,
    )
    arsenal = build_arsenal_showcase()
    goal_dialog = build_goal_confirmation_dialog(project_scan)

    questions: list[dict[str, Any]] = [
        {
            "parameter": "algorithm_lineup",
            "terminology": "Algorithm Portfolio (Classic Foundations -> Frontier SOTA)",
            "prompt": "Which classic baseline + modern SOTA algorithms should we run side-by-side?",
            "plain_analogy": param_guide["paradigm_intuition"]["on_policy_vs_off_policy"],
            "equivalent_to": "Runs a simple, trustworthy baseline alongside a stronger SOTA method so improvements are easy to verify.",
            "if_higher_vs_lower": "Including both a classic foundation and a frontier SOTA proves cleanly how much the newer method truly improves.",
            "options": [
                {"algos": suite.algos, "label": f"Recommended balanced progression: {', '.join(suite.algos)} [Recommended]"},
                {"algos": suite.classic_baselines[:1] + suite.frontier_sota[:1], "label": f"Classic vs Frontier pair: {', '.join(suite.classic_baselines[:1] + suite.frontier_sota[:1])}"},
            ],
        },
        param_guide["total_timesteps_and_seeds"],
        *list(param_guide["algorithm_specific"].values()),
        param_guide["gamma"],
        param_guide["storage_and_outputs"],
    ]

    return {
        "interaction_guidelines": {
            "language_rule": "Respond naturally in the user's conversation language.",
            "traverse_and_explain_first_rule": (
                "Explain what you observed in the project FIRST before asking questions. Never cold-ask 'what is your project?' without sharing clues."
            ),
            "arsenal_showcase_rule": (
                f"Share the `wannado-drl v{arsenal['version']}` arsenal summary ({arsenal['total_algorithms']} algorithms, classic foundations -> 2025 frontier SOTA)."
            ),
            "goal_dialog_rule": "Pop `goal_confirmation_dialog` first so the user either confirms the inferred DRL goal or enters one sentence stating what they want DRL to do.",
            "stage0_rule": "After goal confirmation, align Stage 0 MDP (`drl_lab/mdp_spec.json`) and preflight baselines BEFORE running training.",
            "zero_jargon_rule": "Explain EVERY terminology using `plain_analogy`, `equivalent_to`, and `if_higher_vs_lower` before offering options, and confirm `storage_and_outputs` + `runtime_estimate` ETA.",
        },
        "project_scan": project_scan.to_dict() if project_scan is not None else None,
        "arsenal_showcase": arsenal,
        "goal_confirmation_dialog": goal_dialog,
        "stage0_mdp_intake": intake,
        "recommended_suite": suite.to_dict(),
        "parameter_explanations": param_guide,
        "questions_to_ask_user": questions,
        "runtime_estimate": eta.to_dict(),
    }


def build_post_run_debrief(report: Any, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Build a structured plain-language post-run summary, provenance inventory, and actionable next steps."""
    summaries = list(getattr(report, "algo_summaries", []))
    bib = report.verified_bibliography() if hasattr(report, "verified_bibliography") else []
    rand_b = getattr(report, "random_baseline", None)
    heur_b = getattr(report, "heuristic_baseline", None)

    verdicts: list[str] = []
    all_flags: list[str] = []
    for s in summaries:
        flags = list(s.audit.red_flags) if s.audit is not None else []
        all_flags.extend(flags)
        if not flags:
            status = "passed all statistical & anti-reward-hacking checks"
        else:
            status = f"flagged `{', '.join(flags)}`"
        vs_rand = f" (vs Random {rand_b.mean_return:.2f})" if rand_b is not None else ""
        vs_heur = f" (vs Heuristic {heur_b.mean_return:.2f})" if heur_b is not None else ""
        verdicts.append(
            f"{s.display_name} (`{s.algo_id}`): Eval IQM = {s.seed_stats['iqm']:.2f} "
            f"(mean {s.seed_stats['mean']:.2f} ± {s.seed_stats['std']:.2f}){vs_rand}{vs_heur} — {status}."
        )

    next_options: list[dict[str, str]] = []
    if not all_flags:
        next_options.append({"id": "export_paper", "label": "All checks passed — use generated Markdown/LaTeX/BibTeX/charts directly in your paper or README [Recommended]"})
        next_options.append({"id": "scale_seeds", "label": "Scale to 5 seeds / longer budget for tighter confidence intervals"})
    else:
        if "below_heuristic_baseline" in all_flags or "failed_to_beat_random_baseline" in all_flags:
            next_options.append({"id": "increase_budget_or_tune", "label": "Agent hasn't surpassed baseline yet — double training episodes/steps or inspect reward scaling in `drl_lab/interface.py` [Recommended]"})
        if "suspected_reward_hacking_kpi_divergence" in all_flags:
            next_options.append({"id": "fix_reward_hacking", "label": "Reward rose while core KPI dropped — rebalance `reward_terms` weights inside `drl_lab/interface.py` [Recommended]"})
        if "obs_rms_unfrozen_on_eval" in all_flags:
            next_options.append({"id": "freeze_obs_rms", "label": "Freeze observation normalization (`obs_rms.freeze(True)`) during evaluation"})
        if "policy_entropy_collapse" in all_flags:
            next_options.append({"id": "boost_exploration", "label": "Policy stopped exploring too early — raise `ent_coef` (or try DAPO Clip-Higher / higher `eps_end`)"})
        if "single_seed_survivorship_risk" in all_flags:
            next_options.append({"id": "add_seeds", "label": "Re-run with >= 3 random seeds to compute trustworthy IQM & eliminate luck"})

    out_str = str(Path(output_dir).resolve()) if output_dir is not None else "./drl_lab/outputs"
    return {
        "plain_verdict": " ".join(verdicts) if verdicts else "No algorithm runs recorded.",
        "red_flags": list(dict.fromkeys(all_flags)),
        "algorithms_and_citations": [
            {
                "algo_id": s.algo_id,
                "display_name": s.display_name,
                "primary_citation": s.citation.get("ieee", ""),
                "primary_key": s.citation.get("key", ""),
                "optimizations": [
                    {
                        "id": o["id"],
                        "name": o["name"],
                        "citation_key": (o.get("citation") or {}).get("key", ""),
                        "ieee": (o.get("citation") or {}).get("ieee", ""),
                    }
                    for o in s.optimizations_used
                ],
            }
            for s in summaries
        ],
        "verified_bibliography_count": len(bib),
        "storage_directory": out_str,
        "next_step_options": next_options,
    }
