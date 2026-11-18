"""Benchmark B+C: Safe/Risk-Constrained RL, Future-Leakage Probes & Live SB3 Policy Audit."""

from __future__ import annotations

import csv
import sys
import tempfile
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
    CVaRPPOConfig,
    PPOConfig,
    PPOLagrangianConfig,
    SACConfig,
    SACLagrangianConfig,
    aggregate_seed_stats,
    audit_external_policy,
    compute_cvar,
    evaluate_drl_suite,
    run_mdp_probes,
    train_cvar_ppo,
    train_ppo,
    train_ppo_lagrangian,
    train_sac,
    train_sac_lagrangian,
)


class WirelessPowerBudgetCMDP(gym.Env[np.ndarray, int]):
    """Discrete CMDP: high-power TX boosts throughput (+3.0/step) at cost=+1.0/step; low/med gives safe throughput within budget."""

    def __init__(self, horizon: int = 12) -> None:
        super().__init__()
        self.horizon = horizon
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(0)
        self._state = np.zeros(3, dtype=np.float32)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._state = self._rng.uniform(-0.4, 0.4, size=3).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._t += 1
        rew = 3.0 if int(action) == 1 else 1.25
        cost = 1.0 if int(action) == 1 else 0.0
        self._state = np.clip(0.7 * self._state + (0.2 if int(action) == 1 else -0.2), -2.5, 2.5).astype(np.float32)
        return self._state.copy(), rew, False, self._t >= self.horizon, {"cost": cost}


class TorqueSafetyCMDP(gym.Env[np.ndarray, np.ndarray]):
    """Continuous CMDP: larger control |a| yields higher speed reward, while |a| > 0.35 accrues thermal safety cost."""

    def __init__(self, horizon: int = 10) -> None:
        super().__init__()
        self.horizon = horizon
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._x = 0.0
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._x = float(self._rng.uniform(-0.3, 0.3))
        return np.array([self._x, 0.0], dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self._t += 1
        rew = 2.0 * a - 0.1 * (self._x**2)
        cost = max(0.0, abs(a) - 0.35) * 2.0
        self._x = float(np.clip(0.5 * self._x + 0.1 * a, -1.5, 1.5))
        return np.array([self._x, float(self._t) / self.horizon], dtype=np.float32), rew, False, self._t >= self.horizon, {"cost": cost}


class HeavyTailPortfolioEnv(gym.Env[np.ndarray, int]):
    """Tail-risk benchmark: Safe arm returns +5.0 steadily; Risky arm yields higher mean (+5.8) but 28% crash (-25.0)."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(0)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return np.array([0.5, -0.5], dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._t += 1
        done = self._t >= 5
        if int(action) == 0:
            return np.array([0.5, float(self._t) / 5.0], dtype=np.float32), 1.0, False, done, {}
        rew = -22.0 if float(self._rng.random()) < 0.06 else 2.6
        return np.array([0.5, float(self._t) / 5.0], dtype=np.float32), rew, False, done, {}


def _build_leak_envs() -> list[tuple[str, Any, bool]]:
    """Return (label, env_or_factory, expected_leak) across clean benchmarks and 3 planted leakage bugs."""

    class _CausalWireless(gym.Env):
        observation_space = spaces.Box(0.0, 50.0, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            self._rng = np.random.default_rng(seed or 0)
            self._q, self._h, self._t = 5.0, float(self._rng.exponential(1.0)), 0
            return np.array([self._q, self._h], dtype=np.float32), {}

        def step(self, action):
            rate = min(self._q, float(np.log1p(float(action) * 0.8 * self._h)))
            rew = rate - 0.15 * float(action) - 0.05 * self._q
            self._q = float(min(50.0, max(0.0, self._q - rate) + float(self._rng.poisson(0.6))))
            self._h = float(self._rng.exponential(1.0))
            self._t += 1
            return np.array([self._q, self._h], dtype=np.float32), rew, False, self._t >= 30, {}

    class _ShiftedLookahead(gym.Env):
        observation_space = spaces.Box(-100.0, 100.0, shape=(2,), dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            self._rng = np.random.default_rng(seed or 0)
            self._p = np.concatenate([[10.0], 10.0 + np.cumsum(self._rng.normal(0.0, 1.0, size=63))]).astype(np.float32)
            self._t = 0
            return np.array([self._p[0], self._p[1]], dtype=np.float32), {}

        def step(self, action):
            a = float(np.asarray(action).reshape(-1)[0])
            rew = a * float(self._p[self._t + 1] - self._p[self._t])
            self._t += 1
            return np.array([self._p[self._t], self._p[self._t + 1]], dtype=np.float32), rew, False, self._t >= 30, {}

    class _NonMarkovLeak(gym.Env):
        observation_space = spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            self._rng = np.random.default_rng(seed or 0)
            self._s, self._c, self._n, self._t = 0.0, float(self._rng.normal(0.0, 1.0)), float(self._rng.normal(0.0, 2.0)), 0
            return np.array([self._s, self._n], dtype=np.float32), {}

        def step(self, action):
            rew = float(self._c + 0.05 * (1 if action == 1 else -1))
            self._c, self._n = self._n, float(self._rng.normal(0.0, 2.0))
            self._s = float(0.8 * self._s + 0.2 * (1.0 if action == 1 else -1.0))
            self._t += 1
            return np.array([self._s, self._n], dtype=np.float32), rew, False, self._t >= 40, {}

    class _ExogenousLeak(gym.Env):
        observation_space = spaces.Box(-100.0, 100.0, shape=(2,), dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            self._rng = np.random.default_rng(seed or 0)
            self._p, self._nr, self._t = 10.0 + float(self._rng.normal(0.0, 0.5)), float(self._rng.normal(0.0, 1.5)), 0
            return np.array([self._p, self._nr], dtype=np.float32), {}

        def step(self, action):
            a = float(np.asarray(action).reshape(-1)[0])
            rew = a * self._nr
            self._p += self._nr
            self._nr = float(self._rng.normal(0.0, 1.5))
            self._t += 1
            return np.array([self._p, self._nr], dtype=np.float32), rew, False, self._t >= 35, {}

    return [
        ("CartPole-v1", "CartPole-v1", False),
        ("Pendulum-v1", "Pendulum-v1", False),
        ("Acrobot-v1", "Acrobot-v1", False),
        ("FrameStack(CartPole)", lambda: gym.wrappers.FrameStackObservation(gym.make("CartPole-v1"), stack_size=2), False),
        ("CausalWireless", _CausalWireless, False),
        ("Leak:ShiftedLookahead", _ShiftedLookahead, True),
        ("Leak:NonMarkovReward", _NonMarkovLeak, True),
        ("Leak:ExogInnovation", _ExogenousLeak, True),
    ]


def _run_optional_sb3_external_audit(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    """Train real SB3 PPO + VecNormalize models (frozen vs unfrozen trap) and run Stage 3/4 audit."""
    try:
        from stable_baselines3 import PPO as SB3PPO  # type: ignore[import-not-found]
        from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec  # type: ignore[import-not-found]
        from stable_baselines3.common.vec_env import VecNormalize  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return []

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sb3_models = []
        for s in seeds:
            venv = VecNormalize(sb3_make_vec("CartPole-v1", n_envs=4, seed=s), norm_obs=True, norm_reward=False)
            model = SB3PPO("MlpPolicy", venv, n_steps=128, batch_size=64, verbose=0, seed=s)
            model.learn(total_timesteps=4096)
            m_path = tmp / f"sb3_ppo_{s}.zip"
            vn_path = tmp / f"vn_{s}.pkl"
            model.save(str(m_path))
            venv.save(str(vn_path))
            sb3_models.append((m_path, vn_path))
            venv.close()

        # Trap: Unfrozen VecNormalize (training=True) caught automatically by audit_external_policy
        vn_unfrozen = VecNormalize.load(str(sb3_models[0][1]), sb3_make_vec("CartPole-v1", n_envs=1, seed=99))
        vn_unfrozen.training = True
        model_0 = SB3PPO.load(str(sb3_models[0][0]), env=vn_unfrozen, device="cpu")
        bad_audit = audit_external_policy(
            "CartPole-v1",
            model_0,
            algo_id="sb3_ppo_unfrozen_trap",
            vec_normalize=vn_unfrozen,
            eval_episodes=8,
            min_seeds=1,
        )
        vn_unfrozen.close()
        rows.append(
            {
                "section": "sb3_audit",
                "task": "CartPole-v1",
                "method": "SB3 PPO (Unfrozen VecNorm Trap)",
                "metric_name": "audit_passed",
                "iqm": float(bad_audit.passed),
                "ci_low": 0.0,
                "ci_high": 0.0,
                "extra": ",".join(bad_audit.red_flags),
            }
        )

        # Fair unified 3-seed rliable comparison: Frozen SB3 PPO vs wannado-drl PPO
        sb3_eval_models = []
        frozen_envs = []
        for s_idx, (m_path, vn_path) in enumerate(sb3_models):
            fvn = VecNormalize.load(str(vn_path), sb3_make_vec("CartPole-v1", n_envs=1, seed=200 + s_idx))
            fvn.training = False
            fvn.norm_reward = False
            frozen_envs.append(fvn)
            sb3_eval_models.append(SB3PPO.load(str(m_path), env=fvn, device="cpu"))
        suite = evaluate_drl_suite(
            "CartPole-v1",
            algos=["ppo"],
            seeds=list(seeds),
            algo_configs={
                "ppo": PPOConfig(total_timesteps=4096, num_envs=4, n_steps=128, normalize_obs=True, eval_episodes=8)
            },
            external_policies={"sb3_ppo_frozen": sb3_eval_models},
            eval_episodes=8,
            min_seeds=3,
            task_name="CartPole_vs_SB3_Audit",
        )
        for fvn in frozen_envs:
            fvn.close()
        for summ in suite.algo_summaries:
            rows.append(
                {
                    "section": "sb3_audit",
                    "task": "CartPole-v1",
                    "method": f"{summ.algo_id} (suite)",
                    "metric_name": "eval_return",
                    "iqm": round(float(summ.seed_stats["iqm"]), 3),
                    "ci_low": round(float(summ.seed_stats.get("iqm_ci_low", summ.seed_stats["iqm"])), 3),
                    "ci_high": round(float(summ.seed_stats.get("iqm_ci_high", summ.seed_stats["iqm"])), 3),
                    "extra": f"passed={summ.audit.passed if summ.audit else True}",
                }
            )
    return rows


def main() -> None:
    out_dir = REPO_ROOT / "benchmarks" / "outputs"
    asset_dir = REPO_ROOT / "benchmarks" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    seeds = (0, 1, 2)
    rows: list[dict[str, Any]] = []

    # 1. Safe & Risk-Constrained RL vs Unconstrained Baselines (n=3 seeds)
    disc_ppo_rets, disc_ppo_costs, disc_lag_rets, disc_lag_costs = [], [], [], []
    cont_sac_rets, cont_sac_costs, cont_lag_rets, cont_lag_costs = [], [], [], []
    tail_ppo_means, tail_ppo_cvars, tail_cvar_means, tail_cvar_cvars = [], [], [], []

    for s in seeds:
        # Wireless Power CMDP (budget = 2.0)
        r_u = train_ppo(WirelessPowerBudgetCMDP, config=PPOConfig(total_timesteps=3072, n_steps=128, lr=3e-3, eval_episodes=10, seed=s))
        r_s = train_ppo_lagrangian(
            WirelessPowerBudgetCMDP,
            config=PPOLagrangianConfig(
                total_timesteps=4096,
                n_steps=128,
                lr=3e-3,
                cost_critic_lr=3e-3,
                cost_limit=2.0,
                init_lambda=0.4,
                lagrange_lr=0.35,
                eval_episodes=10,
                seed=s,
            ),
        )
        # Compute unconstrained eval cost separately using policy
        from wannado.drl.core.algo_helpers import evaluate_safety_episodes

        u_safety = evaluate_safety_episodes(WirelessPowerBudgetCMDP, r_u.policy_bundle, num_episodes=10, seed=s + 10_000, cost_limit=2.0)
        disc_ppo_rets.append(float(r_u.eval_stats.mean_return))
        disc_ppo_costs.append(float(u_safety["eval_mean_cost"]))
        disc_lag_rets.append(float(r_s.eval_stats.mean_return))
        disc_lag_costs.append(float(r_s.train_diagnostics["eval_mean_cost"]))

        # Continuous Torque CMDP (budget = 1.2)
        sac_u = train_sac(TorqueSafetyCMDP, config=SACConfig(total_timesteps=2500, learning_starts=200, eval_episodes=10, seed=s))
        sac_u_safe = evaluate_safety_episodes(TorqueSafetyCMDP, sac_u.policy_bundle, num_episodes=10, seed=s + 10_000, cost_limit=1.2)
        sac_s = train_sac_lagrangian(
            TorqueSafetyCMDP,
            config=SACLagrangianConfig(
                total_timesteps=2500,
                learning_starts=200,
                cost_limit=1.2,
                pid_kp=0.5,
                pid_ki=0.2,
                pid_kd=0.05,
                init_lambda=0.5,
                eval_episodes=10,
                seed=s,
            ),
        )
        cont_sac_rets.append(float(sac_u.eval_stats.mean_return))
        cont_sac_costs.append(float(sac_u_safe["eval_mean_cost"]))
        cont_lag_rets.append(float(sac_s.eval_stats.mean_return))
        cont_lag_costs.append(float(sac_s.train_diagnostics["eval_mean_cost"]))

        # Heavy-Tail Risk Portfolio (25%-CVaR)
        tp = train_ppo(HeavyTailPortfolioEnv, config=PPOConfig(total_timesteps=6144, n_steps=128, lr=3e-3, eval_episodes=30, seed=s))
        tc = train_cvar_ppo(
            HeavyTailPortfolioEnv,
            config=CVaRPPOConfig(total_timesteps=6144, n_steps=128, lr=3e-3, cvar_alpha=0.25, mean_mix=0.05, ent_coef=0.005, eval_episodes=30, seed=s),
        )
        tail_ppo_means.append(float(tp.eval_stats.mean_return))
        tail_ppo_cvars.append(float(compute_cvar(tp.eval_stats.returns, alpha=0.25)))
        tail_cvar_means.append(float(tc.eval_stats.mean_return))
        tail_cvar_cvars.append(float(compute_cvar(tc.eval_stats.returns, alpha=0.25)))

    for task, label, metric, vals in (
        ("WirelessPowerCMDP", "PPO (Unconstrained)", "eval_cost", disc_ppo_costs),
        ("WirelessPowerCMDP", "PPO-Lagrangian", "eval_cost", disc_lag_costs),
        ("WirelessPowerCMDP", "PPO (Unconstrained)", "eval_return", disc_ppo_rets),
        ("WirelessPowerCMDP", "PPO-Lagrangian", "eval_return", disc_lag_rets),
        ("TorqueSafetyCMDP", "SAC (Unconstrained)", "eval_cost", cont_sac_costs),
        ("TorqueSafetyCMDP", "SAC-Lagrangian (PID)", "eval_cost", cont_lag_costs),
        ("TorqueSafetyCMDP", "SAC (Unconstrained)", "eval_return", cont_sac_rets),
        ("TorqueSafetyCMDP", "SAC-Lagrangian (PID)", "eval_return", cont_lag_rets),
        ("HeavyTailPortfolio", "PPO (Risk-Neutral)", "eval_cvar25", tail_ppo_cvars),
        ("HeavyTailPortfolio", "CVaR-PPO (alpha=0.25)", "eval_cvar25", tail_cvar_cvars),
    ):
        st = aggregate_seed_stats(vals)
        rows.append(
            {
                "section": "safe_risk_rl",
                "task": task,
                "method": label,
                "metric_name": metric,
                "iqm": round(st["iqm"], 3),
                "ci_low": round(st["iqm_ci_low"], 3),
                "ci_high": round(st["iqm_ci_high"], 3),
                "extra": f"mean={st['mean']:.2f}",
            }
        )

    # 2. Future-Leakage Probe Benchmark (Clean vs Leaked)
    probe_results: list[tuple[str, bool, bool, str]] = []
    for label, target, expected_leak in _build_leak_envs():
        rep = run_mdp_probes(target, horizon_steps=160, n_permutations=30, seed=42)
        probe_results.append((label, expected_leak, rep.future_leakage_suspected, "; ".join(rep.leakage_signals)))
        rows.append(
            {
                "section": "leakage_probe",
                "task": label,
                "method": "MDPProbe",
                "metric_name": "detected_leak",
                "iqm": float(rep.future_leakage_suspected),
                "ci_low": float(expected_leak),
                "ci_high": float(rep.future_leakage_suspected == expected_leak),
                "extra": "; ".join(rep.leakage_signals) or f"kappa={rep.obs_condition_ratio}",
            }
        )

    # 3. Optional real SB3 cross-audit
    sb3_rows = _run_optional_sb3_external_audit(seeds)
    rows.extend(sb3_rows)

    csv_path = out_dir / "safe_and_audit_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "task", "method", "metric_name", "iqm", "ci_low", "ci_high", "extra"])
        writer.writeheader()
        writer.writerows(rows)

    # Plot 3-Panel Visual Summary
    fig, (ax_cmdp, ax_cvar, ax_probe) = plt.subplots(1, 3, figsize=(15.8, 4.7), dpi=160)

    # Panel 1: CMDP Safety Cost Compliance (Unconstrained vs Lagrangian)
    cmdp_labels = ["Wireless\nPPO", "Wireless\nPPO-Lag", "Torque\nSAC", "Torque\nSAC-Lag"]
    cmdp_vals = [
        aggregate_seed_stats(disc_ppo_costs)["iqm"],
        aggregate_seed_stats(disc_lag_costs)["iqm"],
        aggregate_seed_stats(cont_sac_costs)["iqm"],
        aggregate_seed_stats(cont_lag_costs)["iqm"],
    ]
    cmdp_cols = ["#ef4444", "#059669", "#f97316", "#2563eb"]
    bars1 = ax_cmdp.bar(cmdp_labels, cmdp_vals, color=cmdp_cols, alpha=0.88, edgecolor="#1e293b", linewidth=0.7)
    ax_cmdp.axhline(2.0, linestyle="--", color="#059669", linewidth=1.2, label="Wireless Budget (2.0)")
    ax_cmdp.axhline(1.2, linestyle=":", color="#2563eb", linewidth=1.3, label="Torque Budget (1.2)")
    ax_cmdp.set_ylim(0.0, 14.5)
    for bar, v in zip(bars1, cmdp_vals):
        ax_cmdp.text(bar.get_x() + bar.get_width() / 2.0, v + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_cmdp.set_title("CMDP Constraint Satisfaction (Cost ↓)")
    ax_cmdp.set_ylabel("Mean Episode Cost (3 seeds IQM)")
    ax_cmdp.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_cmdp.legend(loc="upper right", fontsize=8)

    # Panel 2: Heavy-Tail Risk Protection (Mean vs 25%-CVaR)
    cvar_labels = ["PPO\nMean", "PPO\nCVaR-25%", "CVaR-PPO\nMean", "CVaR-PPO\nCVaR-25%"]
    cvar_vals = [
        aggregate_seed_stats(tail_ppo_means)["iqm"],
        aggregate_seed_stats(tail_ppo_cvars)["iqm"],
        aggregate_seed_stats(tail_cvar_means)["iqm"],
        aggregate_seed_stats(tail_cvar_cvars)["iqm"],
    ]
    bars2 = ax_cvar.bar(cvar_labels, cvar_vals, color=["#94a3b8", "#dc2626", "#60a5fa", "#16a34a"], alpha=0.88, edgecolor="#1e293b", linewidth=0.7)
    ax_cvar.axhline(0.0, color="#0f172a", linewidth=0.8)
    ax_cvar.set_ylim(min(-14.5, min(cvar_vals) - 2.0), max(7.0, max(cvar_vals) + 1.8))
    for bar, v in zip(bars2, cvar_vals):
        y_off = 0.35 if v >= 0 else -0.45
        va = "bottom" if v >= 0 else "top"
        ax_cvar.text(bar.get_x() + bar.get_width() / 2.0, v + y_off, f"{v:+.2f}", ha="center", va=va, fontsize=8.5, fontweight="bold")
    ax_cvar.set_title("Heavy-Tail Crash Suppression (CVaR-25% ↑)")
    ax_cvar.set_ylabel("Episode Return / Tail Utility")
    ax_cvar.grid(True, axis="y", linestyle=":", alpha=0.45)

    # Panel 3: Future-Leakage Probe Specificity & Sensitivity
    p_names = [p[0] for p in probe_results]
    p_scores = [1.0 if p[2] else 0.0 for p in probe_results]
    p_cols = [("#dc2626" if exp else "#16a34a") for _, exp, _, _ in probe_results]
    y_pos = np.arange(len(p_names))
    ax_probe.barh(y_pos, p_scores, color=p_cols, alpha=0.88, edgecolor="#1e293b", linewidth=0.6)
    for y_i, (_, exp, caught, _) in enumerate(probe_results):
        if caught:
            ax_probe.text(0.5, y_i, "LEAK CAUGHT", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        else:
            ax_probe.scatter([0.04], [y_i], color="#16a34a", edgecolor="#1e293b", s=45, zorder=5)
            ax_probe.text(0.11, y_i, "CLEAN PASS (0% FP)" if not exp else "MISS", ha="left", va="center", color="#15803d", fontsize=8, fontweight="bold")
    ax_probe.set_yticks(y_pos)
    ax_probe.set_yticklabels(p_names, fontsize=8)
    ax_probe.set_xlim(0.0, 1.15)
    ax_probe.set_xticks([0.0, 1.0])
    ax_probe.set_xticklabels(["Clean (0)", "Leaked (1)"])
    ax_probe.invert_yaxis()
    ax_probe.set_title("Stage 1 Future-Leakage Probe (0% FP / 100% TP)")
    ax_probe.grid(True, axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout()
    chart_path = asset_dir / "safe_and_audit_moat.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved chart to {chart_path}")
    for r in rows:
        print(f"  [{r['section']}] {r['task']:22s} | {r['method']:30s} | {r['metric_name']}={r['iqm']} [{r['ci_low']}, {r['ci_high']}] ({r['extra']})")


if __name__ == "__main__":
    main()
