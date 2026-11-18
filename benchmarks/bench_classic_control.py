"""Benchmark 1: Multi-seed convergence on Gymnasium Classic Control (CartPole-v1 & Pendulum-v1)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wannado.drl import (
    A2CConfig,
    DoubleDQNConfig,
    DuelingDQNConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    evaluate_drl_suite,
)


def _interpolate_curves(seed_runs: list[dict], max_steps: int, grid_n: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0, max_steps, grid_n)
    ys = []
    for run in seed_runs:
        pts = run.get("learning_curve", [])
        if len(pts) < 2:
            continue
        rx = np.array([p["step"] for p in pts], dtype=np.float64)
        ry = np.array([p["episode_return"] for p in pts], dtype=np.float64)
        ys.append(np.interp(xs, rx, ry))
    mat = np.vstack(ys) if ys else np.zeros((1, grid_n))
    return xs, np.mean(mat, axis=0), np.std(mat, axis=0)


def main() -> None:
    out_csv = Path(__file__).resolve().parent / "outputs" / "classic_control_results.csv"
    out_png = Path(__file__).resolve().parent / "assets" / "classic_control_convergence.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    cp_steps = 20_000
    cp_report = evaluate_drl_suite(
        lambda: gym.make("CartPole-v1"),
        algos=["ppo", "double_dqn", "dueling_dqn", "a2c"],
        seeds=seeds,
        algo_configs={
            "ppo": PPOConfig(total_timesteps=cp_steps, n_steps=256, update_epochs=6, lr=5e-4, eval_episodes=10),
            "double_dqn": DoubleDQNConfig(total_timesteps=cp_steps, learning_starts=400, lr=5e-4, eval_episodes=10),
            "dueling_dqn": DuelingDQNConfig(total_timesteps=cp_steps, learning_starts=400, lr=5e-4, eval_episodes=10),
            "a2c": A2CConfig(total_timesteps=cp_steps, n_steps=32, lr=7e-4, eval_episodes=10),
        },
        task_name="CartPole-v1",
        domain="discrete_control",
    )

    pend_steps = 12_000
    pend_report = evaluate_drl_suite(
        lambda: gym.make("Pendulum-v1"),
        algos=["sac", "td3", "ppo"],
        seeds=seeds,
        algo_configs={
            "sac": SACConfig(total_timesteps=pend_steps, learning_starts=400, actor_lr=1e-3, critic_lr=1e-3, eval_episodes=10),
            "td3": TD3Config(total_timesteps=pend_steps, learning_starts=400, actor_lr=1e-3, critic_lr=1e-3, eval_episodes=10),
            "ppo": PPOConfig(total_timesteps=pend_steps, n_steps=256, lr=1e-3, normalize_obs=True, eval_episodes=10),
        },
        task_name="Pendulum-v1",
        domain="continuous_control",
    )

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["env", "algo_id", "mean_return", "std_return", "iqm_return", "random_baseline", "audit_passed"])
        for env_name, rep in [("CartPole-v1", cp_report), ("Pendulum-v1", pend_report)]:
            rb = rep.random_baseline.mean_return if rep.random_baseline else 0.0
            for s in rep.algo_summaries:
                st = s.seed_stats
                w.writerow([env_name, s.algo_id, round(st["mean"], 2), round(st["std"], 2), round(st["iqm"], 2), round(rb, 2), s.audit.passed if s.audit else True])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=160)
    colors = {"ppo": "#2563eb", "double_dqn": "#16a34a", "dueling_dqn": "#9333ea", "a2c": "#ea580c", "sac": "#059669", "td3": "#dc2626"}

    for ax, title, rep, max_s in [
        (axes[0], "Discrete Control: CartPole-v1 (3 seeds)", cp_report, cp_steps),
        (axes[1], "Continuous Control: Pendulum-v1 (3 seeds)", pend_report, pend_steps),
    ]:
        if rep.random_baseline:
            ax.axhline(rep.random_baseline.mean_return, linestyle="--", color="#64748b", linewidth=1.4, label=f"Random ({rep.random_baseline.mean_return:.1f})")
        for s in rep.algo_summaries:
            xs, m, sd = _interpolate_curves(s.seed_runs, max_s)
            c = colors.get(s.algo_id, "#0f172a")
            ax.plot(xs / 1000.0, m, label=f"{s.algo_id} (IQM={s.seed_stats['iqm']:.1f})", color=c, linewidth=2.0)
            ax.fill_between(xs / 1000.0, m - sd, m + sd, color=c, alpha=0.14)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Environment Steps (k)")
        ax.set_ylabel("Episode Return")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("CartPole-v1 summary:\n", cp_report.format_paper_summary())
    print("\nPendulum-v1 summary:\n", pend_report.format_paper_summary())


if __name__ == "__main__":
    main()
