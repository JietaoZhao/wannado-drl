# wannado-drl Claude Code Collaboration Rules

Read `.claude/skills/wannado-drl/SKILL.md` (and `AGENTS.md`) before starting any reinforcement learning task.

- **Scope & Handoff**: Use `wannado-drl` (`./drl_lab/`) only for MDP design, RL training, and RL evaluation. If `wannado-adv` is also present and the user asks for adversarial robustness / perturbation testing on a trained policy, hand off cleanly to `./adv_eval/` without re-triggering Stage-0 dialogs, preserving frozen `obs_rms_state` via differentiable PyTorch ops in `VictimInterface`.

1. Run `wannado-drl inspect .` first: share your project understanding + `wannado-drl arsenal` showcase, then pop a 2-step **Goal Confirmation Dialog** (confirm AI-inferred DRL goal OR ask user for 1 sentence) before Stage 0 Problem-to-MDP alignment.
2. Search both **Classic Baselines & Frontier SOTA**, isolate all RL wiring inside `./drl_lab/` (`wannado-drl scaffold`), and use `wannado-drl eject <algo_id>` for CleanRL single-file customization.
3. Run `preflight_env()` (`ms/step` and Random/Heuristic baselines), explain every terminology (`episodes`, `epsilon`, `gamma`, `ent_coef`, etc.) with plain analogies via `explain_parameters()`, and confirm **data storage location** (`./drl_lab/outputs/`) + measured ETA before training.
4. Run `evaluate_drl_suite(..., run_audit=True)` + `report.save()`, present a plain verdict (`build_post_run_debrief`) backed by `paper_summary.md` tables & `results_overview.png`/`.svg` charts, cite BOTH every algorithm AND every active optimization/trick from `OPTIMIZATION_PROVENANCE` / `verified_bibliography`, and export verified citations only via `wannado-drl cite`.
