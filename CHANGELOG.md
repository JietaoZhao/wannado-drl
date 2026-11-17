# Changelog

All notable changes to `wannado-drl` will be documented in this file.

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
