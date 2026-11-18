---
name: wannado-drl
description: Engineering protocol & machine-verifiable verification gate for wannado-drl (Problem-to-MDP formalization, active Future-Leakage Probes, 30 CleanRL-style algorithms across 8 families including Safe/CMDP/CVaR RL, external SB3/PyTorch policy auditing, rliable Bootstrap CIs, and zero-hallucination BibTeX).
---

# WannaDo-DRL Engineering & Verification Protocol

This skill turns any AI coding assistant into a disciplined DRL engineer backed by **hard automated verification gates** rather than guesswork.

---

## 1. Activation Scope & Sibling Handoff (`wannado-adv`)
- **Scope (`wannado-drl` owns MDP, training, safety constraints, & rigor auditing)**: Activate when formalizing an MDP (`./drl_lab/mdp_spec.json`), building/benchmarking DRL policies, auditing an external policy (`SB3` / `nn.Module` / `PolicyBundle`), or producing statistical paper reports inside `./drl_lab/`.
- **Clean Handoff to `wannado-adv`**: If the user asks for **adversarial robustness or perturbation stress-testing** on an already-trained RL policy and `wannado-adv` is present:
  - Do NOT alter `./drl_lab/` or re-run DRL intake dialogs; hand off directly to `./adv_eval/` (`VictimInterface`).
  - **Frozen Normalization Gradient Rule**: When the trained RL policy used `normalize_obs=True` (`obs_rms_state` with `mean`, `var`, `frozen=True`), apply `(obs - mean) / torch.sqrt(var + 1e-8)` (clamped `[-10, 10]`) via differentiable PyTorch ops inside `VictimInterface`.

---

## 2. Non-Intrusive Boundary (`./drl_lab/`) & Problem -> MDP Discipline
1. **Never scatter RL loops across user source files.** Scaffold one external lab via:
   ```bash
   wannado-drl scaffold --dir ./drl_lab --archetype <archetype> --mode <mode>
   ```
   Wire the simulator strictly in `./drl_lab/interface.py` (`EnvInterface`), returning `info['reward_terms']`, `info['task_metrics']`, and (when constrained) `info['cost']` / `info['constraint_cost']`.
2. **Optional Project Intake**: If project intent or domain assumptions are genuinely unclear, run `wannado-drl inspect . --json` (`scan_project`, `build_interactive_brief`) to inspect files, clarify ambiguities concisely in the user's language, and confirm storage (`./drl_lab/outputs/`) and hardware ETA (`estimate_runtime`) before multi-hour runs. For parameters or algorithmic intuition, call `explain_parameters()`.
3. **Stage 0 MDP Contract (`./drl_lab/mdp_spec.json`)**:
   - Verify sequential state coupling (`diagnose_rl_suitability()` — advise Bandits/SL/MPC when RL is unnecessary).
   - Enforce `terminated` (real terminal failure/goal, bootstrap=0) vs `truncated` (time horizon limit, bootstrap!=0).
   - Pick from 30 single-file algorithms across 8 families (`recommend_algo_suite()`, `wannado-drl catalog`), including **Safe & Risk-Constrained RL** (`ppo_lagrangian`, `sac_lagrangian`, `cvar_ppo`).
   - Use `wannado-drl eject <algo_id> --dest ./drl_lab/algos` when customizing ONE standalone CleanRL file while preserving `[AI-Prompt Hook]` and `optimizations_used`.

---

## 3. Production Engine & External Policy Interop
- **VectorEnv & Multimodal Encoders**: Pass `num_envs=4..8` (`vec_mode="sync"` or `"async"`) and callable env factories. `extractor_type="auto"` automatically builds MLP / `NatureCNNExtractor` / `ImpalaResNetExtractor` / `SetAttentionExtractor` / `DictCombinedExtractor` with native `uint8` frame buffers and frozen `DictRunningMeanStd` on eval.
- **Checkpoints & Resume**: Enable `checkpoint_dir`, `eval_freq`, `checkpoint_freq`, `save_replay_buffer=True`, `resume_from=...`, and portable `load_policy(path)`.
- **Audit Your Own SB3 / PyTorch / Heuristic Models**: Use `audit_external_policy()` or `evaluate_drl_suite(..., external_policies={"sb3_ppo": sb3_model, ...})` (or CLI `wannado-drl verify --policy model.zip`) to detect unfrozen SB3 `VecNormalize(training=True)`, action saturation collapse, or baseline failures directly alongside `wannado-drl` baselines.

---

## 4. Hard Machine-Verifiable Completion Gate (`wannado-drl verify`)
**NEVER declare a DRL pipeline or benchmark complete solely because training finished without crashing.** You MUST pass the machine-verifiable gate:

```bash
wannado-drl verify --interface ./drl_lab/interface.py --algos <algo_ids> --seeds 0 1 2 --out ./drl_lab/outputs --json
```
*(Or programmatically run `preflight_env(..., run_probes=True)` -> `evaluate_drl_suite(..., seeds=(0,1,2), run_audit=True)` -> `report.save(...)`.)*

The gate **fails with exit code 1** unless ALL invariants pass:
1. **Stage 1 Active MDP Probes (`run_mdp_probes` / `preflight_env`)**:
   - Zero NaN/Inf, seeded reset reproducibility, zero action-mask deadlocks, measured `ms/step`, Random + Heuristic baseline floors.
   - **Future-Leakage Probe PASS**: Granger Markov test + shifted-column lookahead scan detect zero `t+1` leaks.
2. **Stage 3 Rigor Audit (`audit_drl_rigor`)**:
   - Multi-seed rigor (`min_seeds >= 3`, never cherry-picked single seed).
   - Mean eval return beats both Random and Heuristic baselines.
   - Normalization `obs_rms_frozen_on_eval == True`.
   - No `suspected_reward_hacking_kpi_divergence`, `policy_entropy_collapse`, `policy_action_saturation`, `critic_negative_explained_variance`, `train_eval_collapse_mismatch`, or `safety_constraint_budget_exceeded`.
3. **Stage 4 Statistical Rigor & Zero-Hallucination Citations**:
   - Report IQM with 95% Stratified Bootstrap CIs, Mann-Whitney pairwise $P(A > B)$ (`pairwise_improvement_probabilities()`), Performance Profiles, and Markdown/LaTeX/chart artifacts (`paper_summary.md`, `paper_table.tex`, `results_overview.png`/`.svg`, `references.bib`).
   - Cite algorithms and active optimizations **strictly verbatim** via `OPTIMIZATION_PROVENANCE`, `verified_bibliography`, or `wannado-drl cite --include-audit-refs`. Summarize results plainly via `build_post_run_debrief(report)`.
