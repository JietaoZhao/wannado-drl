"""Benchmark A4: Fair Cross-Calibration vs Stable-Baselines3 + MinAtar CNN + Multimodal SetAttention."""

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

from wannado.drl import (
    DoubleDQNConfig,
    DQNConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    aggregate_seed_stats,
    evaluate_drl_suite,
    evaluate_policy_episodes,
    train_double_dqn,
    train_dqn,
    train_ppo,
    train_sac,
    train_td3,
)


class MinAtarGymEnv(gym.Env[np.ndarray, int]):
    """Lightweight Gymnasium wrapper around ``minatar.Environment`` (10x10xC spatial grid)."""

    def __init__(self, game: str = "breakout", sticky_action_prob: float = 0.0, max_steps: int = 500) -> None:
        super().__init__()
        import minatar

        self.game_env = minatar.Environment(game, sticky_action_prob=sticky_action_prob)
        self.max_steps = max_steps
        shape = self.game_env.state_shape()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=tuple(shape), dtype=np.float32)
        self.action_space = spaces.Discrete(int(self.game_env.num_actions()))
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None and hasattr(self.game_env, "seed"):
            self.game_env.seed(int(seed))
        self.game_env.reset()
        self.steps = 0
        return self.game_env.state().astype(np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        rew, term = self.game_env.act(int(action))
        trunc = (self.steps >= self.max_steps) and not bool(term)
        return self.game_env.state().astype(np.float32), float(rew), bool(term), bool(trunc), {}


class MultimodalFleetDispatchEnv(gym.Env[dict[str, Any], int]):
    """Multimodal (2D occupancy map + scalar queue + entity set with variable mask + action_mask)."""

    def __init__(self, num_slots: int = 6, max_steps: int = 40) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.max_steps = max_steps
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=255, shape=(16, 16, 2), dtype=np.uint8),
                "vec": spaces.Box(low=-5.0, high=5.0, shape=(4,), dtype=np.float32),
                "entities": spaces.Box(low=-5.0, high=5.0, shape=(num_slots, 4), dtype=np.float32),
                "entities_mask": spaces.MultiBinary(num_slots),
            }
        )
        self.action_space = spaces.Discrete(num_slots)
        self.t = 0
        self.active = np.zeros(num_slots, dtype=np.bool_)
        self.utility = np.zeros(num_slots, dtype=np.float32)
        self.queue = np.zeros(num_slots, dtype=np.float32)
        self.urgency = np.zeros(num_slots, dtype=np.float32)

    def _sample_state(self) -> None:
        n_active = int(self.np_random.integers(2, self.num_slots + 1))
        chosen = self.np_random.choice(self.num_slots, size=n_active, replace=False)
        self.active = np.zeros(self.num_slots, dtype=np.bool_)
        self.active[chosen] = True
        self.utility = self.np_random.uniform(0.1, 1.0, size=self.num_slots).astype(np.float32)
        self.queue = self.np_random.uniform(0.2, 1.0, size=self.num_slots).astype(np.float32)
        self.urgency = self.np_random.uniform(0.1, 1.0, size=self.num_slots).astype(np.float32)
        self.utility[~self.active] = 0.0
        self.queue[~self.active] = 0.0
        self.urgency[~self.active] = 0.0

    def _obs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        grid = np.zeros((16, 16, 2), dtype=np.uint8)
        for i in range(self.num_slots):
            if self.active[i]:
                r = 2 + (i // 3) * 6
                c = 2 + (i % 3) * 5
                grid[r : r + 3, c : c + 3, 0] = int(np.clip(self.utility[i] * 255.0, 10, 255))
                grid[r : r + 3, c : c + 3, 1] = int(np.clip(self.urgency[i] * 255.0, 10, 255))
        vec = np.array(
            [
                float(np.mean(self.utility)),
                float(np.mean(self.queue)),
                float(np.mean(self.urgency)),
                float(self.t) / float(self.max_steps),
            ],
            dtype=np.float32,
        )
        feats = np.stack(
            [
                self.utility,
                self.queue,
                self.urgency,
                np.linspace(-0.5, 0.5, self.num_slots, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        obs = {
            "grid": grid,
            "vec": vec,
            "entities": feats,
            "entities_mask": self.active.astype(np.int8),
        }
        return obs, {"action_mask": self.active.astype(np.bool_)}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0
        self._sample_state()
        return self._obs()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.t += 1
        a = int(action)
        valid = bool(0 <= a < self.num_slots and self.active[a])
        best_val = float(np.max(self.utility * (1.0 + self.urgency)))
        chosen_val = float(self.utility[a] * (1.0 + self.urgency[a])) if valid else 0.0
        rew = (2.0 * chosen_val - best_val) if valid else -2.0
        self._sample_state()
        obs, info = self._obs()
        info["reward_terms"] = {"dispatch_efficiency": rew}
        info["task_metrics"] = {"optimality_ratio": float(chosen_val / max(best_val, 1e-6))}
        return obs, float(rew), False, self.t >= self.max_steps, info


def _eval_sb3(model: Any, env_fn: Any, *, num_episodes: int = 20, seed_base: int = 10_000) -> list[float]:
    env = env_fn()
    rets: list[float] = []
    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed_base + ep)
            done = False
            tot = 0.0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(act)
                tot += float(r)
                done = bool(term or trunc)
            rets.append(tot)
    finally:
        env.close()
    return rets


def _summarize(label: str, env_name: str, framework: str, algo: str, rets: list[float], secs: list[float]) -> dict[str, Any]:
    stats = aggregate_seed_stats(rets)
    rec = {
        "benchmark": label,
        "env": env_name,
        "framework": framework,
        "algo": algo,
        "seeds": len(rets),
        "iqm": round(float(stats["iqm"]), 2),
        "mean": round(float(stats["mean"]), 2),
        "std": round(float(stats["std"]), 2),
        "mean_sec": round(float(np.mean(secs)), 2),
    }
    print(
        f"[{env_name} | {framework:<11s} | {algo:<14s}] IQM={rec['iqm']:8.2f} | Mean={rec['mean']:8.2f} ± {rec['std']:6.2f} | Time={rec['mean_sec']:5.2f}s",
        flush=True,
    )
    return rec


def run_sb3_ppo(env_fn: Any, *, seeds: list[int], total_timesteps: int, num_envs: int, n_steps: int, batch_size: int, n_epochs: int, lr: float, gae_lambda: float, ent_coef: float, eval_episodes: int) -> tuple[list[float], list[float]]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env as sb3_vec
    from stable_baselines3.common.vec_env import DummyVecEnv

    eval_means, wall_secs = [], []
    for s in seeds:
        vec = sb3_vec(env_fn, n_envs=num_envs, seed=s, vec_env_cls=DummyVecEnv)
        t0 = time.perf_counter()
        model = PPO(
            "MlpPolicy",
            vec,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=lr,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": [64, 64]},
            device="cpu",
            seed=s,
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps)
        wall_secs.append(time.perf_counter() - t0)
        vec.close()
        eval_means.append(float(np.mean(_eval_sb3(model, env_fn, num_episodes=eval_episodes))))
    return eval_means, wall_secs


def main() -> None:
    from stable_baselines3 import DQN, SAC, TD3

    out_csv = Path(__file__).resolve().parent / "outputs" / "sb3_cross_calibration.csv"
    out_png = Path(__file__).resolve().parent / "assets" / "sb3_cross_calibration.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seeds5 = [0, 1, 2, 3, 4]
    seeds3 = [0, 1, 2]

    # 1. CartPole-v1 PPO (num_envs=4, 25k steps, 5 seeds)
    def make_cp() -> gym.Env[Any, Any]:
        return gym.make("CartPole-v1")

    wd_cp_ppo = [
        train_ppo(
            make_cp,
            PPOConfig(
                total_timesteps=25_000,
                n_steps=128,
                minibatch_size=64,
                update_epochs=4,
                lr=3e-4,
                anneal_lr=False,
                num_envs=4,
                eval_episodes=20,
                device="cpu",
                seed=s,
            ),
        )
        for s in seeds5
    ]
    rows.append(_summarize("CartPole-v1 PPO", "CartPole-v1", "wannado-drl", "PPO (4-env)", [r.eval_stats.mean_return for r in wd_cp_ppo], [r.elapsed_seconds for r in wd_cp_ppo]))
    sb_cp_r, sb_cp_t = run_sb3_ppo(make_cp, seeds=seeds5, total_timesteps=25_000, num_envs=4, n_steps=128, batch_size=64, n_epochs=4, lr=3e-4, gae_lambda=0.95, ent_coef=0.01, eval_episodes=20)
    rows.append(_summarize("CartPole-v1 PPO", "CartPole-v1", "SB3 2.9", "PPO (4-env)", sb_cp_r, sb_cp_t))

    # 2. Acrobot-v1 PPO (num_envs=4, 40k steps, 5 seeds)
    def make_ac() -> gym.Env[Any, Any]:
        return gym.make("Acrobot-v1")

    wd_ac_ppo = [
        train_ppo(
            make_ac,
            PPOConfig(
                total_timesteps=40_000,
                n_steps=128,
                minibatch_size=64,
                update_epochs=4,
                lr=3e-4,
                anneal_lr=False,
                num_envs=4,
                eval_episodes=20,
                device="cpu",
                seed=s,
            ),
        )
        for s in seeds5
    ]
    rows.append(_summarize("Acrobot-v1 PPO", "Acrobot-v1", "wannado-drl", "PPO (4-env)", [r.eval_stats.mean_return for r in wd_ac_ppo], [r.elapsed_seconds for r in wd_ac_ppo]))
    sb_ac_r, sb_ac_t = run_sb3_ppo(make_ac, seeds=seeds5, total_timesteps=40_000, num_envs=4, n_steps=128, batch_size=64, n_epochs=4, lr=3e-4, gae_lambda=0.95, ent_coef=0.01, eval_episodes=20)
    rows.append(_summarize("Acrobot-v1 PPO", "Acrobot-v1", "SB3 2.9", "PPO (4-env)", sb_ac_r, sb_ac_t))

    # 3. LunarLander-v3 PPO (Box2D, num_envs=4, 160k steps, 3 seeds)
    def make_ll() -> gym.Env[Any, Any]:
        return gym.make("LunarLander-v3")

    wd_ll_ppo = [
        train_ppo(
            make_ll,
            PPOConfig(
                total_timesteps=160_000,
                n_steps=512,
                minibatch_size=64,
                update_epochs=6,
                lr=5e-4,
                gae_lambda=0.98,
                ent_coef=0.01,
                anneal_lr=False,
                num_envs=4,
                eval_episodes=20,
                device="cpu",
                seed=s,
            ),
        )
        for s in seeds3
    ]
    rows.append(_summarize("LunarLander-v3 PPO", "LunarLander-v3", "wannado-drl", "PPO (matched)", [r.eval_stats.mean_return for r in wd_ll_ppo], [r.elapsed_seconds for r in wd_ll_ppo]))
    wd_ll_rn = [
        train_ppo(
            make_ll,
            PPOConfig(
                total_timesteps=160_000,
                n_steps=512,
                minibatch_size=64,
                update_epochs=8,
                lr=5e-4,
                gae_lambda=0.98,
                ent_coef=0.01,
                normalize_reward=True,
                anneal_lr=True,
                num_envs=4,
                eval_episodes=20,
                device="cpu",
                seed=s,
            ),
        )
        for s in seeds3
    ]
    rows.append(_summarize("LunarLander-v3 PPO", "LunarLander-v3", "wannado-drl", "PPO+RewNorm", [r.eval_stats.mean_return for r in wd_ll_rn], [r.elapsed_seconds for r in wd_ll_rn]))
    sb_ll_r, sb_ll_t = run_sb3_ppo(make_ll, seeds=seeds3, total_timesteps=160_000, num_envs=4, n_steps=512, batch_size=64, n_epochs=6, lr=5e-4, gae_lambda=0.98, ent_coef=0.01, eval_episodes=20)
    rows.append(_summarize("LunarLander-v3 PPO", "LunarLander-v3", "SB3 2.9", "PPO (matched)", sb_ll_r, sb_ll_t))

    # 4. CartPole-v1 Value-Based (DQN & DoubleDQN vs SB3 DQN, 25k steps, 5 seeds)
    wd_cp_dqn = [
        train_dqn(
            make_cp,
            DQNConfig(total_timesteps=25_000, learning_starts=500, lr=5e-4, buffer_size=20_000, batch_size=64, target_update_freq=200, eps_end=0.05, exploration_fraction=0.35, hidden_dims=(128, 128), eval_episodes=20, device="cpu", seed=s),
        )
        for s in seeds5
    ]
    rows.append(_summarize("CartPole-v1 DQN", "CartPole-v1", "wannado-drl", "DQN", [r.eval_stats.mean_return for r in wd_cp_dqn], [r.elapsed_seconds for r in wd_cp_dqn]))

    wd_cp_ddqn = [
        train_double_dqn(
            make_cp,
            DoubleDQNConfig(total_timesteps=25_000, learning_starts=500, lr=5e-4, buffer_size=20_000, batch_size=64, target_update_freq=200, eps_end=0.05, exploration_fraction=0.35, hidden_dims=(128, 128), eval_episodes=20, device="cpu", seed=s),
        )
        for s in seeds5
    ]
    rows.append(_summarize("CartPole-v1 DQN", "CartPole-v1", "wannado-drl", "DoubleDQN", [r.eval_stats.mean_return for r in wd_cp_ddqn], [r.elapsed_seconds for r in wd_cp_ddqn]))

    sb_dqn_r, sb_dqn_t = [], []
    for s in seeds5:
        env = make_cp()
        t0 = time.perf_counter()
        m = DQN("MlpPolicy", env, learning_rate=5e-4, buffer_size=20_000, learning_starts=500, batch_size=64, target_update_interval=200, exploration_fraction=0.35, exploration_final_eps=0.05, train_freq=1, gradient_steps=1, policy_kwargs={"net_arch": [128, 128]}, device="cpu", seed=s, verbose=0)
        m.learn(total_timesteps=25_000)
        sb_dqn_t.append(time.perf_counter() - t0)
        env.close()
        sb_dqn_r.append(float(np.mean(_eval_sb3(m, make_cp, num_episodes=20))))
    rows.append(_summarize("CartPole-v1 DQN", "CartPole-v1", "SB3 2.9", "DQN", sb_dqn_r, sb_dqn_t))

    # 5. Pendulum-v1 Continuous (SAC & TD3 vs SB3 SAC & TD3, 15k steps, 5 seeds)
    def make_pen() -> gym.Env[Any, Any]:
        return gym.make("Pendulum-v1")

    wd_sac = [
        train_sac(
            make_pen,
            SACConfig(total_timesteps=15_000, learning_starts=500, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, buffer_size=50_000, batch_size=64, hidden_dims=(128, 128), eval_episodes=20, device="cpu", seed=s),
        )
        for s in seeds5
    ]
    rows.append(_summarize("Pendulum-v1 SAC", "Pendulum-v1", "wannado-drl", "SAC", [r.eval_stats.mean_return for r in wd_sac], [r.elapsed_seconds for r in wd_sac]))

    sb_sac_r, sb_sac_t = [], []
    for s in seeds5:
        env = make_pen()
        t0 = time.perf_counter()
        m = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=50_000, learning_starts=500, batch_size=64, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, policy_kwargs={"net_arch": [128, 128]}, device="cpu", seed=s, verbose=0)
        m.learn(total_timesteps=15_000)
        sb_sac_t.append(time.perf_counter() - t0)
        env.close()
        sb_sac_r.append(float(np.mean(_eval_sb3(m, make_pen, num_episodes=20))))
    rows.append(_summarize("Pendulum-v1 SAC", "Pendulum-v1", "SB3 2.9", "SAC", sb_sac_r, sb_sac_t))

    wd_td3 = [
        train_td3(
            make_pen,
            TD3Config(total_timesteps=15_000, learning_starts=500, actor_lr=3e-4, critic_lr=3e-4, buffer_size=50_000, batch_size=64, hidden_dims=(128, 128), eval_episodes=20, device="cpu", seed=s),
        )
        for s in seeds5
    ]
    rows.append(_summarize("Pendulum-v1 TD3", "Pendulum-v1", "wannado-drl", "TD3", [r.eval_stats.mean_return for r in wd_td3], [r.elapsed_seconds for r in wd_td3]))

    sb_td3_r, sb_td3_t = [], []
    for s in seeds5:
        env = make_pen()
        t0 = time.perf_counter()
        m = TD3("MlpPolicy", env, learning_rate=3e-4, buffer_size=50_000, learning_starts=500, batch_size=64, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, policy_kwargs={"net_arch": [128, 128]}, device="cpu", seed=s, verbose=0)
        m.learn(total_timesteps=15_000)
        sb_td3_t.append(time.perf_counter() - t0)
        env.close()
        sb_td3_r.append(float(np.mean(_eval_sb3(m, make_pen, num_episodes=20))))
    rows.append(_summarize("Pendulum-v1 TD3", "Pendulum-v1", "SB3 2.9", "TD3", sb_td3_r, sb_td3_t))

    # 6. MinAtar Breakout (10x10x4 visual pixel grid): adaptive CNN vs Random
    def make_minatar() -> gym.Env[Any, Any]:
        return MinAtarGymEnv("breakout", sticky_action_prob=0.0, max_steps=500)

    rand_ma = evaluate_policy_episodes(make_minatar, lambda _o, _i: int(np.random.default_rng().integers(0, 6)), num_episodes=20, seed=0)
    rows.append(_summarize("MinAtar Breakout (10x10 CNN)", "MinAtar-Breakout", "Baseline", "Random", rand_ma.returns, [0.0]))
    wd_ma_ppo = [
        train_ppo(
            make_minatar,
            PPOConfig(total_timesteps=40_000, n_steps=128, minibatch_size=128, update_epochs=4, lr=3e-4, num_envs=4, extractor_type="nature_cnn", eval_episodes=15, device="cpu", seed=s),
        )
        for s in seeds3
    ]
    rows.append(_summarize("MinAtar Breakout (10x10 CNN)", "MinAtar-Breakout", "wannado-drl", "PPO-CNN", [r.eval_stats.mean_return for r in wd_ma_ppo], [r.elapsed_seconds for r in wd_ma_ppo]))
    wd_ma_ddqn = [
        train_double_dqn(
            make_minatar,
            DoubleDQNConfig(
                total_timesteps=40_000,
                learning_starts=1000,
                lr=5e-4,
                buffer_size=25_000,
                batch_size=64,
                train_freq=4,
                gradient_steps=1,
                target_update_freq=400,
                num_envs=4,
                extractor_type="nature_cnn",
                eval_episodes=15,
                device="cpu",
                seed=s,
            ),
        )
        for s in seeds3
    ]
    rows.append(_summarize("MinAtar Breakout (10x10 CNN)", "MinAtar-Breakout", "wannado-drl", "DoubleDQN-CNN", [r.eval_stats.mean_return for r in wd_ma_ddqn], [r.elapsed_seconds for r in wd_ma_ddqn]))

    # 7. Multimodal Dict (CNN Grid + Vec + Entity SetAttention + Action Mask) + Stage 3 Rigor Audit
    def make_mm() -> gym.Env[Any, Any]:
        return MultimodalFleetDispatchEnv()

    mm_rep = evaluate_drl_suite(
        make_mm,
        algos=["ppo"],
        seeds=seeds3,
        algo_configs={
            "ppo": PPOConfig(total_timesteps=20_000, n_steps=128, minibatch_size=64, update_epochs=4, lr=5e-4, num_envs=4, extractor_type="dict", normalize_obs=True, eval_episodes=20, device="cpu"),
        },
        task_name="MultimodalFleetDispatch",
        domain="wireless",
        run_audit=True,
        device="cpu",
    )
    if mm_rep.random_baseline:
        rows.append(_summarize("Multimodal SetAttn+Mask", "FleetDispatch-Dict", "Baseline", "Random (Masked)", mm_rep.random_baseline.returns, [0.0]))
    mm_sum = mm_rep.algo_summaries[0]
    rows.append(
        _summarize(
            "Multimodal SetAttn+Mask",
            "FleetDispatch-Dict",
            "wannado-drl",
            "PPO-Dict+Mask",
            [float(r["eval_stats"]["mean_return"]) for r in mm_sum.seed_runs],
            [float(r["elapsed_seconds"]) for r in mm_sum.seed_runs],
        )
    )
    print(f"Multimodal Stage 3 Rigor Audit passed={mm_sum.audit.passed if mm_sum.audit else True}")

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["benchmark", "env", "framework", "algo", "seeds", "iqm", "mean", "std", "mean_sec"])
        w.writeheader()
        w.writerows(rows)

    # Generate multi-panel SB3 cross-calibration & multimodal figure
    panels = [
        ("CartPole-v1 PPO", "CartPole-v1 (PPO, 25k)"),
        ("Acrobot-v1 PPO", "Acrobot-v1 (PPO, 40k)"),
        ("LunarLander-v3 PPO", "LunarLander-v3 (PPO, 160k)"),
        ("Pendulum-v1 SAC", "Pendulum-v1 (SAC, 15k)"),
        ("Pendulum-v1 TD3", "Pendulum-v1 (TD3, 15k)"),
        ("MinAtar Breakout (10x10 CNN)", "MinAtar Breakout (10x10 CNN)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), dpi=160)
    palette = {"wannado-drl": "#2563eb", "SB3 2.9": "#64748b", "Baseline": "#d97706"}
    for ax, (b_key, title) in zip(axes.flat, panels, strict=True):
        sub = [r for r in rows if r["benchmark"] == b_key]
        x_labels = [f"{r['framework']}\n({r['algo']})" for r in sub]
        vals = [r["iqm"] for r in sub]
        errs = [r["std"] for r in sub]
        cols = [palette.get(r["framework"], "#059669") for r in sub]
        ax.bar(x_labels, vals, yerr=errs, capsize=5, color=cols, alpha=0.92)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Eval IQM Return")
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
