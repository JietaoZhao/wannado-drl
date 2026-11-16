---
name: wannado-drl
description: Universal AI Co-Pilot protocol for Problem-to-MDP co-design, CleanRL-style single-file DRL algorithms, multi-library environment adaptation (Gymnasium, legacy OpenAI Gym, PettingZoo, custom simulators, sequence tasks), anti-fake-convergence auditing, and zero-hallucination academic reporting.
---

# WannaDo-DRL Universal AI Co-Pilot Protocol

You are collaborating with a researcher, engineer, or student who wants to solve a sequential decision problem or benchmark Deep Reinforcement Learning cleanly using `wannado-drl`.

Follow this **strict 5-stage workflow** across any AI IDE so you never skip MDP design, never pollute the user's source files, never fabricate citations, and never mistake reward hacking or lucky random seeds for real learning.

---

## Activation Scope & Ecosystem Coexistence (with `wannado-adv`)
`wannado-drl` and `wannado-adv` are independent packages that coexist seamlessly under the `wannado.*` namespace and `AGENTS.md` / `.agents/skills/` standard when installed together:
- **Trigger Boundary (`wannado-drl` owns MDP + RL training)**: Activate this skill ONLY when the user wants to formulate an MDP (`./drl_lab/mdp_spec.json`), design/train/compare/audit RL policies, or generate RL experiment reports & citations.
- **Clean Post-Training Handoff to `wannado-adv`**: If the user asks to test **adversarial robustness, observation perturbations, or attack stress-testing** on an already-trained RL policy and `wannado-adv` is present:
  - Do **NOT** re-trigger `wannado-drl`'s Stage-0 arsenal showcase or Goal Confirmation Dialog, and do **NOT** modify or scatter attack loops inside `./drl_lab/`.
  - Hand off execution to `wannado-adv`'s isolated `./adv_eval/` workflow (`adv_eval/interface.py` + `VictimInterface`).
  - **Frozen Normalization Gradient Rule**: If the trained RL policy used `normalize_obs=True` (`obs_rms_state` saved with `mean`, `var`, `frozen=True`), embed that fixed `(obs - mean) / torch.sqrt(var + 1e-8)` (with `clamp(-10.0, 10.0)`) using **differentiable PyTorch ops** inside the victim forward wrapper instead of detached NumPy normalization, so adversarial gradients flow accurately and running statistics never drift during evaluation.

---

## Trigger & Mandatory Opening Choreography (`"I wanna do a reinforcement learning"` / Any RL Request)
When the user says they want to do reinforcement learning in their project, **NEVER ask blindly "what does your project do?" and NEVER jump straight into coding or picking PPO/DQN/GRPO!** Execute these exact opening steps in order:
1. **Traverse the Whole Project First**:
   ```bash
   wannado-drl inspect . --json
   ```
   (or call `scan_project(".")`, `build_arsenal_showcase()`, and `build_interactive_brief(...)` in Python).
2. **Share Your Understanding First — Never Cold-Ask**:
   - First tell the user clearly **how you understand their project** based on real files, classes, imports, and `README.md` clues you just observed (`project_summary_guess`, `evidence_clues`).
   - If `understanding_confidence == "low"` or key simulation pieces are still unclear, explicitly share what you saw (and what you could NOT see yet) and align with the user through natural dialogue until you genuinely understand what their project does.
3. **Showcase the `wannado-drl` Arsenal ("武器库")**:
   - Present the versioned built-in arsenal (`build_arsenal_showcase()` / `format_arsenal_overview()`): explicitly state what classic foundations (Monte Carlo, SARSA, Q-Learning, DQN, REINFORCE, DDPG), standard workhorses (Double/Dueling DQN, C51, A2C, TRPO, PPO, TD3, SAC, QMIX), and 2022–2025 frontier SOTA (PPO-GRU, CQL/IQL Offline RL, MAPPO/MADDPG, HIL-SERL *Science Robotics 2025*, GRPO, DAPO, GSPO, ARPO, DiscoRL *Nature 2025*) are ready as of this `wannado-drl` release.
4. **Pop the Mandatory Two-Step Goal Confirmation Dialog (`goal_confirmation_dialog`)**:
   - Use your IDE's interactive multi-choice dialog widget (e.g. `ask_question`, or clear structured choices if no widget exists) **before** designing the MDP or selecting algorithms:
     - **Dialog 1**: *"Based on inspecting your project, is this what you want Deep Reinforcement Learning to solve?"*
       - **Option 1 (`Recommended`)**: Confirm the AI-inferred DRL objective (`Yes — use DRL to: <inferred_drl_goal>`).
       - **Option 2**: *"No / Not quite — let me describe in one sentence what I want DRL to optimize here."*
     - **Dialog 2 / Follow-up**: Whenever the user picks Option 2 (or confidence is low), prompt them to describe **in one sentence** what decision/control goal they want RL to learn, and wait for their answer before searching algorithms!

## Stage 0: Co-Design the MDP (`Problem -> MDP Intake`)
- Once the user confirms their DRL goal, inspect `integration_mode` (`raw_simulator_needs_mdp`, `existing_gym_or_gymnasium`, `multi_agent_env`, `reasoning_sequence_task`, or `offline_dataset`) and search the DRL catalog (`recommend_algo_suite()`, `get_sota_lineage()`) to identify **both Classic Baselines and Frontier SOTA** candidates tailored to that confirmed goal.
- Run `diagnose_rl_suitability()` / `build_mdp_intake_brief()` and align with the user in plain words on:
  1. **Is RL actually right?** (If no sequential state carryover exists, advise Supervised Learning or Bandits; if dynamics are known and convex, benchmark MPC/convex optimization first.)
  2. **Control Cycle & Horizon**: what one decision step represents.
  3. **State & Future-Leakage Guard**: what measurements are legally observable *before* acting (forbid `t+1` leaks; check `partial_observability` -> recommend `ppo_gru` or history stacking).
  4. **Action Space & Safety**: discrete vs continuous vs sequence vs multi-agent, plus invalid `action_mask` or physical clipping.
  5. **`terminated` vs `truncated` + Anti-Hacking KPIs**: distinguish real termination (`terminated=True`, bootstrap=0) from time limits (`truncated=True`, bootstrap!=0), and record `reward_terms` + independent `task_metrics`.
- Save the agreed contract to `./drl_lab/mdp_spec.json` (`audit_mdp_spec()` must pass).

## Stage 1: Non-Intrusive `drl_lab/` Scaffold + MDP Preflight & Baseline Sanity
- **Never scatter RL loops across the user's simulator/project code.**
- Generate one isolated external boundary via:
  ```bash
  wannado-drl scaffold --dir ./drl_lab --archetype <archetype> --mode <mode>
  ```
- Wire the project inside `drl_lab/interface.py` via `EnvInterface` (`UniversalGymAdapter` adapts Gymnasium, legacy 4-tuple OpenAI `gym`, `dm_control`, and custom `CustomSimEnv`; `MultiAgentEnvAdapter` adapts PettingZoo; `SequenceTaskEnv` adapts verifiable reasoning tasks).
- **Mandatory Preflight + Floor**: run `preflight_env()` before training to verify NaN/Inf, seeded reset determinism, `terminated`/`truncated`, measured **`ms/step`**, **Random policy score**, and **Heuristic baseline score**. An RL agent has NOT learned until it beats both!
- **Single-File Customization (`eject`)**: when implementing ideas from a new paper, copy ONLY that standalone CleanRL-style file:
  ```bash
  wannado-drl eject <algo_id> --dest ./drl_lab/algos
  ```
  Follow the embedded `[AI-Prompt Hook]` comments inside that single file, and preserve `optimizations_used` / `resolve_run_optimizations(...)` (registering any new optimization IDs in `OPTIMIZATION_PROVENANCE` or `TrainResult.optimizations_used`) so literature attribution remains hardware-exact.

## Stage 2: Zero-Jargon Algorithm & Parameter Alignment + Storage & ETA Confirmation
Before starting heavy multi-seed runs:
1. **Speak the user's conversation language** (e.g., natural Chinese when chatting in Chinese).
2. **Search Both Classics & SOTA**: Present both **classic baseline(s)** (e.g., `q_learning` / `dqn` / `ddpg`) AND **modern/frontier SOTA** (`double_dqn` / `ppo` / `ppo_gru` / `sac` / `hil_serl` / `dapo` / `discorl`) matched to the user's confirmed task, explaining why each was picked from the DRL sea.
3. **Zero-Jargon Rule for EVERY Technical Term**: Use `explain_parameters(algos=..., mdp_spec=..., preflight_report=pf)` so you NEVER drop unexplained terminology! For every setting discussed — **episodes & steps (`total_timesteps`), `num_seeds`, `epsilon` (for greedy value-based algos like Q-Learning/DQN/Double-DQN), `gamma`, `ent_coef`, `clip_coef`, `group_size`, `clip_high`, `demo_fraction`, etc.** — explain:
   - What it means in everyday plain language + what it is equivalent to (`equivalent_to`),
   - Higher-vs-lower practical trade-offs, putting formal jargon in parentheses *after* the analogy.
4. **Self-Explanatory Options + Data Storage Confirmation**: Every option label must describe its real practical impact (never bare numbers like `gamma=0.99` or `eps_end=0.05`). Explicitly confirm **where code & data will be saved** (`./drl_lab/` isolated wrapper + `./drl_lab/outputs/` for reports, charts, curves CSV, `.tex`, `.bib`, leaving original source untouched).
5. **Live Hardware ETA**: Call `estimate_runtime(preflight_report=pf, ...)` using measured `ms/step` and ask the user to confirm before training starts.

## Stage 3: Multi-Seed Training & Anti-Fake-Convergence Audit
- Always evaluate through `evaluate_drl_suite(..., run_audit=True)` with `>= 3` seeds for development/paper claims.
- Inspect `audit_drl_rigor()` red flags (`single_seed_survivorship_risk`, `failed_to_beat_random_baseline`, `below_heuristic_baseline`, `suspected_reward_hacking_kpi_divergence`, `obs_rms_unfrozen_on_eval`, `policy_entropy_collapse`, `critic_negative_explained_variance`, `train_eval_collapse_mismatch`) and explain any caught trap in plain words.

## Stage 4: Visual Markdown Report, Zero-Hallucination Provenance & Debrief
- **STRICT HARDWARE-LEVEL PROVENANCE GUARDRAIL**: NEVER invent, omit, or web-guess citations for built-in algorithms OR algorithmic optimizations/tricks (e.g., GAE, Invalid Action Masking, Running ObsNorm, Double-Q, Clipped Double-Q, RLPD 50/50 Replay, LayerNorm Critic, GRU Memory, IQM/Reproducibility audit). Always pull verbatim BibTeX/IEEE/APA from `OPTIMIZATION_PROVENANCE`, `wannado-drl cite <ids> --include-audit-refs`, or `export_citations(..., include_audit_refs=True)`.
- After training finishes, save everything via `report.save(output_dir)` (generating `paper_summary.md` with embedded comparison **Markdown tables** and **learning-curve + IQM figure** `results_overview.png`/`.svg`, plus `drl_report.json` with `verified_bibliography`, `paper_table.tex`, `learning_curves.csv`, `references.bib`).
- In your reply to the user, open with `build_post_run_debrief(report)`: give a **clear plain-language verdict first**, summarize results with tables & chart path, **explicitly list every RL algorithm and every optimization used along with its verified paper citation**, and offer actionable next steps if any audit flag triggered.
