# wannado-drl Benchmark & End-to-End Validation Suite

This directory verifies `wannado-drl` on real Gymnasium benchmarks, a custom non-Gym wireless scheduling simulator wired through Stage 0→4, frontier group-relative / multi-agent algorithms, and adversarial fake-convergence traps **without adding** plotting packages to [`pyproject.toml`](../pyproject.toml) (core dependencies stay strictly `numpy`, `torch`, `gymnasium`).

## Experiments

| Script | Scope & Verification | Verified Outputs |
| --- | --- | --- |
| [`bench_classic_control.py`](bench_classic_control.py) | Multi-seed `CartPole-v1` (`ppo`, `double_dqn`, `dueling_dqn`, `a2c`) & `Pendulum-v1` (`sac`, `td3`, `ppo`) vs `Random` | [`assets/classic_control_convergence.png`](assets/classic_control_convergence.png), `outputs/classic_control_results.csv` |
| [`bench_wireless_scheduling_and_audit.py`](bench_wireless_scheduling_and_audit.py) | Non-Gym Wireless Downlink Scheduler (Stage 0 `MDPSpec` + Masked `PPO` / `Double DQN` vs `Max-Queue Heuristic` & `Random`) + 4 Fake-Convergence Traps | [`assets/wireless_mdp_and_audit.png`](assets/wireless_mdp_and_audit.png), `outputs/wireless_scheduling_results.csv` |
| [`bench_frontier_sequence_and_marl.py`](bench_frontier_sequence_and_marl.py) | Verifiable Sequence Reasoning (`grpo`, `dapo`, `gspo`, `arpo`) + Cooperative MARL (`vdn`, `qmix`, `mappo`) + `discorl` across multiple seeds | [`assets/frontier_lineage_comparison.png`](assets/frontier_lineage_comparison.png), `outputs/frontier_marl_results.csv` |
| [`bench_vec_scaling.py`](bench_vec_scaling.py) | True `VectorEnv` (`num_envs=1,2,4,8`) throughput scaling vs Stable-Baselines3 on `CartPole-v1` (MLP) and `PixelNav 84x84x3 uint8` (NatureCNN) | [`assets/vec_scaling_throughput.png`](assets/vec_scaling_throughput.png), `outputs/vec_scaling_results.csv` |
| [`bench_vs_sb3.py`](bench_vs_sb3.py) | Multi-seed cross-calibration vs SB3 2.9 (`CartPole-v1`, `Acrobot-v1`, `LunarLander-v3`, `Pendulum-v1 SAC/TD3`) + `MinAtar-Breakout` (`10x10x4`) + Multimodal `FleetDispatch-Dict` (`CNN + SetAttention + ActionMask + Audit`) | [`assets/sb3_cross_calibration.png`](assets/sb3_cross_calibration.png), `outputs/sb3_cross_calibration.csv` |
| [`bench_safe_and_audit.py`](bench_safe_and_audit.py) | Constrained CMDP (`ppo_lagrangian`, `sac_lagrangian`), heavy-tail `cvar_ppo`, Stage 1 Future-Leakage Probes (`5` clean + `3` leaked), & live SB3 `VecNormalize` audit | [`assets/safe_and_audit_moat.png`](assets/safe_and_audit_moat.png), `outputs/safe_and_audit_results.csv` |

## Reproducing

```bash
python3 benchmarks/bench_classic_control.py
python3 benchmarks/bench_wireless_scheduling_and_audit.py
python3 benchmarks/bench_frontier_sequence_and_marl.py
python3 benchmarks/bench_safe_and_audit.py

# Cross-calibration & visual/Box2D benchmarks (requires optional stable-baselines3, minatar, gymnasium[box2d]):
python3 benchmarks/bench_vec_scaling.py
python3 benchmarks/bench_vs_sb3.py
```

Raw CSV outputs (`benchmarks/outputs/`) are gitignored; verified charts (`benchmarks/assets/*.png`) are committed for documentation and PyPI README rendering.
