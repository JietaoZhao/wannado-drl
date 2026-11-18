"""Benchmark A1/A2: True VectorEnv Throughput Scaling & Fair SB3 Comparison (Vector & 84x84 Pixel CNN)."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib
import numpy as np
from gymnasium import spaces

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wannado.drl import PPOConfig, train_ppo


class PixelCartNavEnv(gym.Env[np.ndarray, int]):
    """Lightweight (84, 84, 3) uint8 pixel benchmark exercising CNN feature extraction + uint8 storage."""

    def __init__(self, max_steps: int = 32) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.pos = 2
        self.target = 6
        self.steps = 0

    def _obs(self) -> np.ndarray:
        frame = np.zeros((84, 84, 3), dtype=np.uint8)
        px = min(76, max(4, self.pos * 10))
        tx = min(76, max(4, self.target * 10))
        frame[36:48, px : px + 8, 0] = 240
        frame[36:48, tx : tx + 8, 1] = 240
        return frame

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.pos = int(self.np_random.integers(1, 5))
        self.target = 6
        self.steps = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        self.pos = int(np.clip(self.pos + (1 if int(action) == 1 else -1), 0, 7))
        rew = 1.0 if self.pos == self.target else -0.1 * abs(self.pos - self.target)
        term = self.pos == self.target
        trunc = (self.steps >= self.max_steps) and not term
        return self._obs(), float(rew), term, trunc, {}


def _time_wannado_ppo(env_fn: Any, *, num_envs: int, total_timesteps: int, n_steps: int, extractor: str, seed: int) -> float:
    t0 = time.perf_counter()
    train_ppo(
        env_fn,
        config=PPOConfig(
            total_timesteps=total_timesteps,
            n_steps=n_steps,
            minibatch_size=256,
            update_epochs=4,
            num_envs=num_envs,
            extractor_type=extractor,
            eval_episodes=3,
            device="cpu",
            seed=seed,
        ),
    )
    return float(total_timesteps / max(time.perf_counter() - t0, 1e-6))


def _time_sb3_ppo(env_fn: Any, *, num_envs: int, total_timesteps: int, n_steps: int, policy_name: str, seed: int) -> float:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = sb3_make_vec_env(env_fn, n_envs=num_envs, seed=seed, vec_env_cls=DummyVecEnv)
    t0 = time.perf_counter()
    model = PPO(
        policy_name,
        vec_env,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=4,
        policy_kwargs=None if policy_name == "CnnPolicy" else {"net_arch": [64, 64]},
        device="cpu",
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - t0
    vec_env.close()
    return float(total_timesteps / max(elapsed, 1e-6))


def main() -> None:
    try:
        import stable_baselines3  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install stable-baselines3 in a benchmark env to run bench_vec_scaling.py") from exc

    out_csv = Path(__file__).resolve().parent / "outputs" / "vec_scaling_results.csv"
    out_png = Path(__file__).resolve().parent / "assets" / "vec_scaling_throughput.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    env_counts = [1, 2, 4, 8]
    rows: list[dict[str, Any]] = []

    benchmarks = [
        ("CartPole-v1 (MLP)", lambda: gym.make("CartPole-v1"), 16_384, 256, "mlp", "MlpPolicy"),
        ("PixelNav 84x84x3 uint8 (CNN)", lambda: PixelCartNavEnv(), 8_192, 128, "nature_cnn", "CnnPolicy"),
    ]

    for task_label, env_fn, base_budget, base_n_steps, ext_type, sb3_pol in benchmarks:
        base_wd: float | None = None
        base_sb: float | None = None
        for n_env in env_counts:
            budget = max(base_budget, n_env * base_n_steps * 4)
            wd_fps = _time_wannado_ppo(env_fn, num_envs=n_env, total_timesteps=budget, n_steps=base_n_steps, extractor=ext_type, seed=0)
            sb_fps = _time_sb3_ppo(env_fn, num_envs=n_env, total_timesteps=budget, n_steps=base_n_steps, policy_name=sb3_pol, seed=0)
            if base_wd is None:
                base_wd = wd_fps
            if base_sb is None:
                base_sb = sb_fps
            rows.append(
                {
                    "task": task_label,
                    "num_envs": n_env,
                    "wannado_fps": round(wd_fps, 1),
                    "sb3_fps": round(sb_fps, 1),
                    "wannado_speedup": round(wd_fps / base_wd, 2),
                    "sb3_speedup": round(sb_fps / base_sb, 2),
                }
            )
            print(
                f"[{task_label}] num_envs={n_env}: wannado-drl={wd_fps:,.0f} steps/s ({wd_fps / base_wd:.2f}x) "
                f"| SB3={sb_fps:,.0f} steps/s ({sb_fps / base_sb:.2f}x)"
            )

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "num_envs", "wannado_fps", "sb3_fps", "wannado_speedup", "sb3_speedup"])
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), dpi=160)
    for ax, (task_label, _, _, _, _, _) in zip(axes, benchmarks, strict=True):
        sub = [r for r in rows if r["task"] == task_label]
        xs = [r["num_envs"] for r in sub]
        ax.plot(xs, [r["wannado_fps"] for r in sub], marker="o", linewidth=2.2, color="#2563eb", label="wannado-drl (VectorEnv)")
        ax.plot(xs, [r["sb3_fps"] for r in sub], marker="s", linewidth=2.0, linestyle="--", color="#64748b", label="Stable-Baselines3 (DummyVecEnv)")
        ax.set_xticks(xs)
        ax.set_xlabel("num_envs (Parallel Environments)")
        ax.set_ylabel("Training Throughput (steps / sec)")
        ax.set_title(task_label, fontsize=10.5, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
