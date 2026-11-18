# wannado-drl Universal AI Collaboration Rules

This repository uses `wannado-drl` under the open `AGENTS.md` + `.agents/skills/wannado-drl/SKILL.md` standard.

- **Scope & Sibling Coexistence (`wannado-adv`)**: Activate `wannado-drl` strictly for Problem-to-MDP co-design, DRL training/auditing, and experiment reporting (`./drl_lab/`). If `wannado-adv` is also present and the user requests adversarial robustness stress-testing on a trained policy, hand off cleanly to `./adv_eval/` and embed frozen `obs_rms_state` (`mean`, `var`) via differentiable PyTorch ops inside `VictimInterface`.

Enforce these 4 engineering gates on every RL task:

1. **Non-Intrusive `./drl_lab/` Boundary & Safe/CleanRL Ejection**: Never scatter RL training loops inside user source files. Wire simulators cleanly in `./drl_lab/interface.py` (`wannado-drl scaffold`), define explicit `terminated` vs `truncated` + `reward_terms` + `task_metrics` (+ `info['cost']` for Safe RL `ppo_lagrangian` / `sac_lagrangian` / `cvar_ppo`), and customize single files via `wannado-drl eject <algo_id> --dest ./drl_lab/algos` preserving `[AI-Prompt Hook]` and `optimizations_used`.
2. **Optional Onboarding Clarity**: If project requirements or simulation details are ambiguous, run `wannado-drl inspect .` first, summarize what you observed before asking clarifying questions, align parameters in plain language (`explain_parameters`), and confirm storage (`./drl_lab/outputs/`) + measured `estimate_runtime()` ETA.
3. **Hard Verification Gate Before Completion (`wannado-drl verify`)**: Never mark an RL task done until `wannado-drl verify --interface ./drl_lab/interface.py --seeds 0 1 2` (or `preflight_env` + `evaluate_drl_suite(..., run_audit=True)` + `audit_external_policy` for SB3/PyTorch models) exits `0` with:
   - Zero Stage 1 Future-Leakage Probe alerts & valid action masks,
   - Stage 3 `RigorAuditReport` pass (beats Random & Heuristic baselines, frozen `obs_rms`, no reward hacking, entropy collapse, action saturation, or cost budget violations).
4. **rliable Statistics & Zero-Hallucination Citations**: Report IQM with 95% Bootstrap CIs, pairwise $P(A > B)$, Performance Profiles, and charts (`results_overview.png`/`.svg`, `paper_summary.md`, `paper_table.tex`) via `report.save()` + `build_post_run_debrief(report)`. Cite every algorithm and active optimization verbatim exclusively from `OPTIMIZATION_PROVENANCE` / `verified_bibliography` / `wannado-drl cite --include-audit-refs`.
