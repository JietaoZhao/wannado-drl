# wannado-drl Benchmark & End-to-End Validation Suite

This directory verifies `wannado-drl` on real Gymnasium benchmarks, a custom non-Gym wireless scheduling simulator wired through Stage 0→4, frontier group-relative / multi-agent algorithms, and adversarial fake-convergence traps **without adding** plotting packages to [`pyproject.toml`](../pyproject.toml) (core dependencies stay strictly `numpy`, `torch`, `gymnasium`).

## Experiments

| Script | Scope & Verification | Verified Outputs |
| --- | --- | --- |
| [`bench_classic_control.py`](bench_classic_control.py) | Multi-seed `CartPole-v1` (`ppo`, `double_dqn`, `dueling_dqn`, `a2c`) & `Pendulum-v1` (`sac`, `td3`, `ppo`) vs `Random` | [`assets/classic_control_convergence.png`](assets/classic_control_convergence.png), `outputs/classic_control_results.csv` |
| [`bench_wireless_scheduling_and_audit.py`](bench_wireless_scheduling_and_audit.py) | Non-Gym Wireless Downlink Scheduler (Stage 0 `MDPSpec` + Masked `PPO` / `Double DQN` vs `Max-Queue Heuristic` & `Random`) + 4 Fake-Convergence Traps | [`assets/wireless_mdp_and_audit.png`](assets/wireless_mdp_and_audit.png), `outputs/wireless_scheduling_results.csv` |
| [`bench_frontier_sequence_and_marl.py`](bench_frontier_sequence_and_marl.py) | Verifiable Sequence Reasoning (`grpo`, `dapo`, `gspo`, `arpo`) + Cooperative MARL (`vdn`, `qmix`, `mappo`) + `discorl` across multiple seeds | [`assets/frontier_lineage_comparison.png`](assets/frontier_lineage_comparison.png), `outputs/frontier_marl_results.csv` |

## Reproducing

```bash
python3 benchmarks/bench_classic_control.py
python3 benchmarks/bench_wireless_scheduling_and_audit.py
python3 benchmarks/bench_frontier_sequence_and_marl.py
```

Raw CSV outputs (`benchmarks/outputs/`) are gitignored; verified charts (`benchmarks/assets/*.png`) are committed for documentation and PyPI README rendering.
