# Changelog

All notable changes to `wannado-drl` will be documented in this file.

## [1.1.0] - 2026-11-18

### Added
- **Production Vectorized & Multimodal Training Engine (Phase A)**:
  - True parallel `VectorizedEnvAdapter` (`SyncVectorEnv` / `AsyncVectorEnv` across Gymnasium `SameStep` & `NextStep` autoreset semantics) integrated across online rollouts and off-policy `ReplayBuffer` / `DualDemoReplayBuffer`.
  - Multimodal `ObsSpec` & feature extractors (`NatureCNNExtractor` adaptive to small grids like `10x10` MinAtar, `ImpalaResNetExtractor`, permutation-invariant `SetAttentionExtractor` with entity masks, `DictCombinedExtractor`) with native `uint8` frame buffer storage and mask/image-aware `DictRunningMeanStd` + `RewardNormalizer`.
  - Atomic `CheckpointManager`, standalone `PolicyBundle` / `load_policy()`, periodic eval/best-model persistence, and deterministic RNG/RMS/optimizer resume (`resume_from`).
- **Irreplaceable MDP Causality Probes, *rliable* Statistics & External Policy Interop (Phase B)**:
  - Pure-NumPy Stage 1 **Active MDP Probes (`audit/probes.py`)**: catches non-Markov future reward leakage (K-fold Granger permutation test), shifted-column lookahead clones (distinguishing causal `FrameStack` via reset signatures), exogenous future-innovation columns, high observation condition ratios ($\kappa$), reward volatility, and action-mask deadlocks before training begins.
  - Pure-NumPy ***rliable* Statistical Rigor Suite (`core/utils.py`, `reporter/paper_reporter.py`)**: deterministic stratified bootstrap 95% CIs (`bootstrap_iqm_ci`, `bootstrap_mean_ci`), pairwise Mann-Whitney Probability of Improvement $P(A > B)$ with 95% CIs (`probability_of_improvement`), and score Performance Profiles (`compute_performance_profile`), rendered into 3-panel `results_overview.png`, Markdown tables, LaTeX tables, and JSON reports.
  - **External Policy Adapter (`audit/external_adapter.py`)**: `wrap_external_policy()`, `evaluate_external_policy()`, `audit_external_policy()`, and `evaluate_drl_suite(..., external_policies=...)` natively audit Stable-Baselines3 (`.predict()` + `VecNormalize`), PyTorch `nn.Module`, saved `.pt`/`.zip` checkpoints, and arbitrary callables—automatically catching unfrozen SB3 `VecNormalize(training=True)` and continuous action saturation (`policy_action_saturation`). Also supports direct Gymnasium env ID strings (`"CartPole-v1"`, etc.) across `ensure_gym_env()`, `make_vec_env()`, and `preflight_env()`.
- **Safe & Risk-Constrained RL Family (`8th Family`, 30 Total Algorithms) & Hard AI Verification Gate (Phase C)**:
  - Added single-file `safe_constrained` algorithms: `ppo_lagrangian` (`ray2019benchmarking`, primal-dual cost critic & dual ascent), `sac_lagrangian` (`stooke2020responsive`, twin safety cost Q-critic + PID Lagrangian dual control), and `cvar_ppo` (`chow2018risk`, Rockafellar-Uryasev percentile tail-risk shaping), wired cleanly into `OPTIMIZATION_PROVENANCE`, `catalog/registry.json`, `recommend_algo_suite`, and `safety_constraint_budget_exceeded` rigor auditing.
  - Added **`wannado-drl verify`** CLI subcommand (`verify_interface_gate`) enforcing Stage 1 Preflight + Future-Leakage Probes -> Training / External Policy Eval -> Stage 3 Rigor Audit -> Stage 4 artifacts (`verification_verdict.json`), returning strict non-zero exit status on any failed invariant, and streamlined `SKILL.md` / `AGENTS.md` templates around this machine-verifiable contract.

## [1.0.5] - 2026-11-17

### Changed
- **English-first repository consistency**: Removed remaining Chinese UI strings and parenthetical labels from `copilot/planner.py` (`_FAMILY_LABELS`, `explain_parameters()` terminology strings), `SKILL.md` templates, and the `docs/index.html` feedback portal so all public outputs and documentation default strictly to English.

## [1.0.4] - 2026-11-17

### Fixed
- **Legacy OpenAI `gym`, `Dict`/`Tuple`/`Discrete`/`MultiDiscrete`/`MultiBinary` & VectorEnv compatibility (`core/wrappers.py`, `core/networks.py`)**:
  - Updated `_is_non_numeric_space()`, `_flatten_obs()`, and `_infer_flat_obs_dim()` with structural space duck-typing so legacy `gym==0.26` `Dict`/`Tuple`/`Discrete`/`MultiDiscrete`/`MultiBinary` spaces never trigger `gymnasium.spaces.flatdim()` `NotImplementedError`.
  - Fixed `MultiBinary` misidentification in `_coerce_action_space()` (no longer coerced to scalar `Discrete`) and added full `MultiDiscrete` + `MultiBinary` support via joint discrete mapping (`prod <= 256`) and native factorized `MultiDiscreteActorCritic` (`MultiCategorical` with registered KL) across `ppo`, `a2c`, `trpo`, `reinforce`, and `ppo_gru`.
  - Added native `SyncVectorEnv` / `AsyncVectorEnv` (`VectorEnv`) handling in `UniversalGymAdapter` and `preflight_env()`.
- **Offline RL Rigor Audit crashes & warnings (`audit/rigor_auditor.py`, `reporter/paper_reporter.py`)**:
  - Eliminated `KeyError: 'episode_return'` and `RuntimeWarning: Mean of empty slice` when auditing `train_cql()` / `train_iql_offline()` loss-based `learning_curve` records, and added `mean_q_value`/`max_q_value` diagnostics to `train_iql_offline()`.
  - Enabled end-to-end simulator-free `preflight_env(dataset)` and `evaluate_drl_suite(dataset, ["cql", "iql_offline"], run_audit=True)` backed by automated Off-Policy Evaluation (`WIS` + `FQE`).
- **Heterogeneous asymmetric Multi-Agent observations (`core/wrappers.py`, `algos/multi_agent/mappo_yu2022.py`)**: Zero-pads differing per-agent observation lengths cleanly inside `MultiAgentEnvAdapter` and `_stack_local_obs()`.
- **LaTeX & Co-Pilot domain inference (`reporter/paper_reporter.py`, `copilot/project_inspector.py`, `copilot/planner.py`)**: Added `#`/`^`/`~` escaping in `to_latex()`, continuous `Box` / offline detection in `scan_project()` & `inspect`, and `query` / `task_goal` arguments on `build_interactive_brief()` and `recommend_algo_suite()`.

### Added
- **Cross-domain Offline, OPE & Risk-Control Primitives (`core/offline_risk.py`)**: Added `evaluate_offline_wis()`, `evaluate_offline_fqe()`, `evaluate_offline_ope()` (with subgroup cohort breakdown), `compute_cvar()`, `compute_sharpe_and_drawdown()`, `apply_simplex_projection()`, and `PortfolioSimplexWrapper`, plus `action_masks`/`next_action_masks` support in `CQL`/`IQL` and `OfflineTransitionDataset.from_arrays()`.
- **Expanded Stage 0 Domain Archetypes (`mdp/intake.py`)**: Added `finance_portfolio_risk`, `clinical_offline_safety`, and `cyber_attack_defense` templates.

## [1.0.3] - 2026-11-17

### Fixed
- **Python 3.10 compatibility (`gymnasium` pin)**: Constrained `gymnasium>=1.0,<1.3; python_version < '3.11'` so fresh installs on Python 3.10 pull a compatible `gymnasium` release instead of `1.4.0` (which fails on `typing.Generic[+_T_co]` under Python 3.10).
- **On-policy TimeLimit `truncated` vs `terminated` GAE leakage**: Updated `RolloutBuffer`, `compute_gae`, and all on-policy trainers (`ppo`, `a2c`, `trpo`, `ppo_gru`, `mappo`, `reinforce`) to compute `gamma * V(terminal_next_obs)` before `env.reset()` overwrites observations and pass `done = term or trunc` to reset GAE recursion cleanly at every episode boundary.
- **Anti-fake-convergence audit (`audit_drl_rigor`) blind spots**: Standardized `final_entropy`, `max_q_value`, `mean_explained_variance`, `is_continuous`, and `final_logstd_mean` diagnostics across algorithms and extended `audit_drl_rigor()` to inspect both primary and fallback keys with scale-invariant continuous Gaussian checks.
- **Single-seed audit bypass**: Updated `evaluate_drl_suite(..., min_seeds: int = 3)` to default to `min_seeds=3` rather than clamping `min(3, len(seeds))` so single-seed survivorship risk warnings remain active by default.
- **Bounded/asymmetric continuous Box action scaling**: Added finite `[action_low, action_high]` affine scaling/centering to `ContinuousActorCritic`, `RecurrentActorCriticGRU`, and `train_mappo`.
- **Code quality & static typing**: Removed duplicate `evaluate_sequence_task()` in `mdp/baselines.py`, fixed `TRPO` loop closure bindings (`B023`), `RolloutBuffer` mutable default (`B008`), static `__all__` export lists, and resolved all `mypy` and `ruff` issues.

### Added
- **Engineering governance**: Added PEP 561 `py.typed` marker, `CITATION.cff`, `CHANGELOG.md`, and GitHub Actions CI matrix across Python 3.10–3.13.

## [1.0.2] - 2026-11-16

### Added
- Public release with Stage 0 Problem-to-MDP co-design, 27 CleanRL-style algorithms across 7 families (including `HIL-SERL`, `GRPO`, `DAPO`, `GSPO`, `ARPO`, `DiscoRL`), multi-library adapters, Stage 3 rigor auditor, and verified citation exports.
