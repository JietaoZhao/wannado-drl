<!-- wannado-drl:start -->
# wannado-drl Claude Code Collaboration Rules

Read `.claude/skills/wannado-drl/SKILL.md` (and `AGENTS.md`) before starting any reinforcement learning task.

- **Scope & Handoff (`wannado-adv`)**: Use `wannado-drl` (`./drl_lab/`) only for MDP design, RL training/auditing, and paper reporting. Hand off adversarial robustness testing cleanly to `./adv_eval/` while preserving frozen `obs_rms_state` via differentiable PyTorch ops in `VictimInterface`.

1. **Isolate & Formalize**: Keep all RL wiring in `./drl_lab/` (`wannado-drl scaffold`) and customize single files via `wannado-drl eject <algo_id>`. Include `reward_terms`, `task_metrics`, and `info['cost']` when constrained (`ppo_lagrangian`, `sac_lagrangian`, `cvar_ppo`).
2. **Machine-Verifiable Completion Gate**: Never report an RL solution ready until `wannado-drl verify --interface ./drl_lab/interface.py --seeds 0 1 2` (or `preflight_env` + `evaluate_drl_suite(..., run_audit=True)` / `audit_external_policy` for SB3/PyTorch policies) passes with zero Future-Leakage Probe errors and zero Stage 3 `RigorAuditReport` red flags.
3. **Statistical Rigor & Verbatim Provenance**: Present `build_post_run_debrief(report)` backed by `paper_summary.md`, 95% Bootstrap CIs + pairwise $P(A > B)$ + Performance Profiles, `results_overview.png`/`.svg`, and verbatim citations pulled strictly via `wannado-drl cite --include-audit-refs` / `OPTIMIZATION_PROVENANCE`.
<!-- wannado-drl:end -->
