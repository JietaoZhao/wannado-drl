"""Benchmark 3: Frontier Reasoning RL (GRPO/DAPO/GSPO/ARPO) + Cooperative MARL (VDN/QMIX/MAPPO)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, ClassVar

import matplotlib
import numpy as np
from gymnasium import spaces

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wannado.drl import (
    ARPOConfig,
    DAPOConfig,
    GRPOConfig,
    GSPOConfig,
    MAPPOConfig,
    QMIXConfig,
    SequenceTaskEnv,
    VDNConfig,
    aggregate_seed_stats,
    train_arpo,
    train_dapo,
    train_grpo,
    train_gspo,
    train_mappo,
    train_qmix,
    train_vdn,
)


class _TeamRelayEnv:
    possible_agents: ClassVar[list[str]] = ["tx_0", "tx_1"]

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._target = 0
        self._steps = 0

    def observation_space(self, _agent: str) -> spaces.Space[Any]:
        return spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def action_space(self, _agent: str) -> spaces.Space[Any]:
        return spaces.Discrete(3)

    def state(self) -> np.ndarray:
        oh = np.zeros(3, dtype=np.float32)
        oh[self._target] = 1.0
        return np.concatenate([oh, oh])

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._target = int(self._rng.integers(0, 3))
        oh = np.zeros(3, dtype=np.float32)
        oh[self._target] = 1.0
        return {"tx_0": oh.copy(), "tx_1": oh.copy()}, {}

    def step(self, actions: dict[str, int]) -> tuple[Any, ...]:
        self._steps += 1
        rew = 1.0 if (actions["tx_0"] == self._target and actions["tx_1"] == self._target) else -0.3
        self._target = int(self._rng.integers(0, 3))
        oh = np.zeros(3, dtype=np.float32)
        oh[self._target] = 1.0
        trunc = self._steps >= 12
        return {"tx_0": oh.copy(), "tx_1": oh.copy()}, {"tx_0": rew, "tx_1": rew}, {"tx_0": False, "tx_1": False}, {"tx_0": trunc, "tx_1": trunc}, {}


def main() -> None:
    out_csv = Path(__file__).resolve().parent / "outputs" / "frontier_marl_results.csv"
    out_png = Path(__file__).resolve().parent / "assets" / "frontier_lineage_comparison.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    task = SequenceTaskEnv(vocab_size=8, prompt_len=5, max_response_len=5)

    # Random sequence floor
    rng = np.random.default_rng(42)
    rand_prompts = task.sample_prompts(60, rng)
    rand_resps = rng.integers(0, 8, size=(60, 5))
    rand_seq_score = float(np.mean([task.score_sequence(rand_prompts[i], rand_resps[i])[0] for i in range(60)]))

    seq_results = {}
    for name, runner, cfg_fn in [
        ("grpo", train_grpo, lambda s: GRPOConfig(total_updates=140, prompts_per_batch=8, group_size=8, kl_coef=0.01, lr=8e-3, seed=s)),
        ("dapo", train_dapo, lambda s: DAPOConfig(total_updates=140, prompts_per_batch=8, group_size=8, lr=8e-3, seed=s)),
        ("gspo", train_gspo, lambda s: GSPOConfig(total_updates=140, prompts_per_batch=8, group_size=8, kl_coef=0.01, lr=8e-3, seed=s)),
        ("arpo", train_arpo, lambda s: ARPOConfig(total_updates=140, prompts_per_batch=8, group_size=6, kl_coef=0.01, lr=8e-3, seed=s)),
    ]:
        vals = [runner(task, cfg_fn(s)).eval_stats.mean_return for s in seeds]
        seq_results[name] = aggregate_seed_stats(vals)

    marl_results = {}
    for name, runner, cfg_fn in [
        ("vdn", train_vdn, lambda s: VDNConfig(total_timesteps=3000, learning_starts=150, lr=1e-3, seed=s)),
        ("qmix", train_qmix, lambda s: QMIXConfig(total_timesteps=3000, learning_starts=150, lr=1e-3, seed=s)),
        ("mappo", train_mappo, lambda s: MAPPOConfig(total_timesteps=3000, n_steps=64, lr=3e-3, seed=s)),
    ]:
        vals = [runner(_TeamRelayEnv(), cfg_fn(s)).eval_stats.mean_return for s in seeds]
        marl_results[name] = aggregate_seed_stats(vals)

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "algo_id", "mean", "std", "iqm"])
        w.writerow(["sequence", "random", round(rand_seq_score, 4), 0.0, round(rand_seq_score, 4)])
        for k, st in seq_results.items():
            w.writerow(["sequence", k, round(st["mean"], 4), round(st["std"], 4), round(st["iqm"], 4)])
        for k, st in marl_results.items():
            w.writerow(["marl", k, round(st["mean"], 4), round(st["std"], 4), round(st["iqm"], 4)])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=160)
    s_labels = ["Random"] + [k.upper() for k in seq_results]
    s_vals = [rand_seq_score] + [seq_results[k]["iqm"] for k in seq_results]
    s_errs = [0.0] + [seq_results[k]["std"] for k in seq_results]
    axes[0].bar(s_labels, s_vals, yerr=s_errs, capsize=5, color=["#94a3b8", "#2563eb", "#16a34a", "#9333ea", "#ea580c"], alpha=0.9)
    axes[0].set_ylim(0.0, 1.1)
    axes[0].set_title("Verifiable Sequence Reasoning (3-Seed IQM)", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Mean Verifiable Reward")
    axes[0].grid(axis="y", alpha=0.25)

    m_labels = ["Random\nFloor"] + [k.upper() for k in marl_results]
    m_vals = [-1.56] + [marl_results[k]["iqm"] for k in marl_results]
    m_errs = [0.0] + [marl_results[k]["std"] for k in marl_results]
    axes[1].bar(m_labels, m_vals, yerr=m_errs, capsize=5, color=["#94a3b8", "#0284c7", "#059669", "#7c3aed"], alpha=0.9)
    axes[1].set_title("Cooperative Multi-Agent Relay (3-Seed IQM)", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Mean Team Return")
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("Sequence Results (IQM):", {k: round(v["iqm"], 3) for k, v in seq_results.items()})
    print("MARL Results (IQM):", {k: round(v["iqm"], 3) for k, v in marl_results.items()})


if __name__ == "__main__":
    main()
