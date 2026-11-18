"""Verification suite for Phase B (Probes, rliable Bootstrap CIs, External Adapters) & Phase C (Safe RL, Verify Gate)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wannado.drl import (
    CVaRPPOConfig,
    PPOConfig,
    PPOLagrangianConfig,
    SACLagrangianConfig,
    aggregate_seed_stats,
    audit_external_policy,
    bootstrap_iqm_ci,
    bootstrap_mean_ci,
    compute_cvar,
    compute_performance_profile,
    evaluate_drl_suite,
    evaluate_external_policy,
    preflight_env,
    probability_of_improvement,
    recommend_algo_suite,
    run_mdp_probes,
    scaffold_lab,
    train_cvar_ppo,
    train_ppo,
    train_ppo_lagrangian,
    train_sac_lagrangian,
)
from wannado.drl.cli import main as cli_main


class _ShiftedLookaheadLeakEnv(gym.Env):
    """Finance-style lookahead bug: obs_t = [price_t, price_{t+1}], reward = action * (price_{t+1} - price_t)."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-100.0, 100.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._prices = np.zeros(64, dtype=np.float32)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        steps = self._rng.normal(0.0, 1.0, size=63)
        self._prices = np.concatenate([[10.0], 10.0 + np.cumsum(steps)]).astype(np.float32)
        self._t = 0
        return np.array([self._prices[0], self._prices[1]], dtype=np.float32), {}

    def step(self, action):
        a = float(np.asarray(action).reshape(-1)[0])
        ret = float(self._prices[self._t + 1] - self._prices[self._t])
        self._t += 1
        trunc = self._t >= 30
        obs = np.array([self._prices[self._t], self._prices[self._t + 1]], dtype=np.float32)
        return obs, a * ret, False, trunc, {}


class _NonMarkovFutureRewardLeakEnv(gym.Env):
    """Hidden 2-step future leak: obs_t reveals next-step exogenous noise z_{t+1} driving r_{t+1} without appearing in s_{t+1}."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(0)
        self._state = 0.0
        self._curr_shock = 0.0
        self._next_shock = 0.0
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._state = float(self._rng.normal(0.0, 0.5))
        self._curr_shock = float(self._rng.normal(0.0, 1.0))
        self._next_shock = float(self._rng.normal(0.0, 2.0))
        return np.array([self._state, self._next_shock], dtype=np.float32), {}

    def step(self, action: int):
        rew = float(self._curr_shock + 0.05 * (1 if action == 1 else -1))
        self._curr_shock = self._next_shock
        self._next_shock = float(self._rng.normal(0.0, 2.0))
        self._state = float(0.8 * self._state + 0.2 * (1.0 if action == 1 else -1.0))
        self._t += 1
        return np.array([self._state, self._next_shock], dtype=np.float32), rew, False, self._t >= 40, {}


class _ExogenousInnovationLeakEnv(gym.Env):
    """Single-step exogenous shock leak: obs_t = [price_t, next_return], r_t = a_t * next_return (never copied into obs_{t+1})."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-100.0, 100.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._price = 10.0
        self._next_ret = 0.0
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._price = 10.0 + float(self._rng.normal(0.0, 0.5))
        self._next_ret = float(self._rng.normal(0.0, 1.5))
        return np.array([self._price, self._next_ret], dtype=np.float32), {}

    def step(self, action):
        a = float(np.asarray(action).reshape(-1)[0])
        rew = a * self._next_ret
        self._price += self._next_ret
        self._next_ret = float(self._rng.normal(0.0, 1.5))
        self._t += 1
        return np.array([self._price, self._next_ret], dtype=np.float32), rew, False, self._t >= 35, {}


class _CausalOpportunisticWirelessEnv(gym.Env):
    """Clean causal stochastic wireless queue: agent observes current causal fading h_t BEFORE choosing TX power."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(0.0, 50.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng(0)
        self._q = 5.0
        self._h = 1.0
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._q = 5.0
        self._h = float(self._rng.exponential(1.0))
        self._t = 0
        return np.array([self._q, self._h], dtype=np.float32), {}

    def step(self, action: int):
        power = float(action) * 0.8
        rate = min(self._q, float(np.log1p(power * self._h)))
        rew = rate - 0.15 * power - 0.05 * self._q
        arrivals = float(self._rng.poisson(0.6))
        self._q = float(min(50.0, max(0.0, self._q - rate) + arrivals))
        self._h = float(self._rng.exponential(1.0))
        self._t += 1
        return np.array([self._q, self._h], dtype=np.float32), rew, False, self._t >= 30, {}


class _PoorlyConditionedEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1e4, 1e4, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(0)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return np.array([self._rng.normal(0.0, 2000.0), self._rng.normal(0.0, 0.005)], dtype=np.float32), {}

    def step(self, action: int):
        self._t += 1
        obs = np.array([self._rng.normal(0.0, 2000.0), self._rng.normal(0.0, 0.005)], dtype=np.float32)
        return obs, 1.0 if action == 1 else -1.0, False, self._t >= 25, {}


class _DiscreteConstrainedCMDP(gym.Env):
    """CMDP where action=1 gives high reward (+3) + high cost (+1), action=0 gives moderate reward (+1.2) + 0 cost."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._s = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._s = self._rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
        return self._s.copy(), {}

    def step(self, action: int):
        self._t += 1
        rew = 3.0 if action == 1 else 1.2
        cost = 1.0 if action == 1 else 0.0
        self._s = (0.7 * self._s + (0.2 if action == 1 else -0.2)).astype(np.float32)
        return self._s.copy(), rew, False, self._t >= 8, {"cost": cost}


class _ContinuousConstrainedCMDP(gym.Env):
    """Continuous CMDP: larger |a| gets bigger speed reward, but |a| > 0.45 incurs safety cost."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._x = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._x = float(self._rng.uniform(-0.3, 0.3))
        return np.array([self._x, 0.0], dtype=np.float32), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        self._t += 1
        rew = 2.0 * a - 0.1 * (self._x**2)
        cost = max(0.0, abs(a) - 0.35) * 2.0
        self._x = float(np.clip(0.5 * self._x + 0.1 * a, -1.5, 1.5))
        return np.array([self._x, float(self._t) / 10.0], dtype=np.float32), rew, False, self._t >= 10, {"cost": cost}


class _HeavyTailHazardEnv(gym.Env):
    """Risk-vs-Safe tradeoff: Safe arm (0) -> deterministic +1.0/step (ep +5.0); Risky arm (1) -> +2.0 82% / -14.0 terminal crash 18% (E > +5.5, CVaR25 << 0)."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng(0)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return np.array([0.5, -0.5], dtype=np.float32), {}

    def step(self, action: int):
        self._t += 1
        done = self._t >= 5
        if action == 0:
            return np.array([0.5, float(self._t) / 5.0], dtype=np.float32), 1.0, False, done, {}
        rew = -22.0 if float(self._rng.random()) < 0.06 else 2.6
        return np.array([0.5, float(self._t) / 5.0], dtype=np.float32), rew, False, done, {}


class _FakeSB3Model:
    """Minimal duck-typed SB3 model supporting .predict(obs, deterministic=True)."""

    def __init__(self, action_value: Any, vec_normalize: Any = None, num_timesteps: int = 5000) -> None:
        self._action_value = action_value
        self._vec_normalize = vec_normalize
        self.num_timesteps = num_timesteps

    def get_vec_normalize_env(self) -> Any:
        return self._vec_normalize

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        del deterministic, obs
        return np.asarray(self._action_value), None


class _FakeVecNormalize:
    def __init__(self, mean: np.ndarray, var: np.ndarray, training: bool) -> None:
        self.obs_rms = type("RMS", (), {"mean": mean, "var": var, "eps": 1e-8})()
        self.training = training
        self.norm_obs = True
        self.clip_obs = 10.0

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10.0, 10.0).astype(np.float32)


class TestPhaseBMoat(unittest.TestCase):
    def test_future_leakage_probes_catch_all_3_modes_without_false_positives(self) -> None:
        # 1. Clean environments must NEVER raise future-leakage false positives
        for clean_target in (
            "CartPole-v1",
            "Pendulum-v1",
            gym.wrappers.FrameStackObservation(gym.make("CartPole-v1"), stack_size=2),
            _CausalOpportunisticWirelessEnv,
        ):
            probe = run_mdp_probes(clean_target, horizon_steps=140, n_permutations=20, seed=42)
            self.assertFalse(probe.future_leakage_suspected, f"False positive on {clean_target}: {probe.leakage_signals}")
            self.assertTrue(probe.ok, probe.errors)
            if isinstance(clean_target, gym.Env):
                clean_target.close()

        # 2. Shifted-column lookahead clone must fail preflight
        pf_shift = preflight_env(_ShiftedLookaheadLeakEnv, probe_steps=140, baseline_episodes=3, strict=False)
        self.assertFalse(pf_shift.ok)
        self.assertTrue(pf_shift.summary["probes"]["future_leakage_suspected"])
        self.assertTrue(any("future information leakage" in e.lower() for e in pf_shift.errors))

        # 3. Non-Markov 2-step future reward leak must fail preflight
        pf_markov = preflight_env(_NonMarkovFutureRewardLeakEnv, probe_steps=160, baseline_episodes=3, strict=False)
        self.assertFalse(pf_markov.ok)
        self.assertTrue(pf_markov.summary["probes"]["future_leakage_suspected"])

        # 4. Exogenous single-step future innovation column in obs must fail preflight
        pf_exog = preflight_env(_ExogenousInnovationLeakEnv, probe_steps=140, baseline_episodes=3, strict=False)
        self.assertFalse(pf_exog.ok)
        self.assertTrue(pf_exog.summary["probes"]["future_leakage_suspected"])

        # 5. Feature conditioning warning on poorly scaled env (without false leakage alarm)
        pf_cond = preflight_env(_PoorlyConditionedEnv, probe_steps=120, baseline_episodes=2)
        self.assertTrue(pf_cond.ok)
        self.assertGreater(pf_cond.summary["probes"]["obs_condition_ratio"], 100.0)
        self.assertTrue(any("conditioning" in w.lower() for w in pf_cond.warnings))

    def test_rliable_bootstrap_ci_poi_profiles_and_report_outputs(self) -> None:
        scores_a = [92.0, 95.0, 91.0, 96.0, 94.0]
        scores_b = [40.0, 45.0, 42.0, 48.0, 44.0]
        iqm, lo, hi = bootstrap_iqm_ci(scores_a, n_bootstrap=500, seed=7)
        self.assertLessEqual(lo, iqm)
        self.assertLessEqual(iqm, hi)
        mean_v, mlo, mhi = bootstrap_mean_ci(scores_a, n_bootstrap=500, seed=7)
        self.assertLessEqual(mlo, mean_v)
        self.assertLessEqual(mean_v, mhi)

        agg = aggregate_seed_stats(scores_a)
        for key in ("mean", "std", "median", "iqm", "min", "max", "iqm_ci_low", "iqm_ci_high", "mean_ci_low", "mean_ci_high"):
            self.assertIn(key, agg)

        poi = probability_of_improvement(scores_a, scores_b, n_bootstrap=400, seed=7)
        self.assertAlmostEqual(poi["prob"], 1.0, places=4)
        self.assertGreater(poi["ci_low"], 0.5)
        self.assertEqual(poi["significant"], 1.0)

        prof = compute_performance_profile({"A": scores_a, "B": scores_b}, num_points=11, normalize=False)
        self.assertEqual(len(prof["thresholds"]), 11)
        self.assertGreater(sum(prof["curves"]["A"]), sum(prof["curves"]["B"]))

    def test_external_policy_adapter_sb3_unfrozen_nn_module_and_suite(self) -> None:
        # 1. Unfrozen VecNormalize (training=True) on SB3-duck policy must trigger obs_rms_unfrozen_on_eval
        unfrozen_vn = _FakeVecNormalize(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32), training=True)
        bad_sb3 = _FakeSB3Model(0, vec_normalize=unfrozen_vn)
        audit_bad = audit_external_policy(
            "CartPole-v1",
            bad_sb3,
            algo_id="sb3_unfrozen",
            eval_episodes=4,
            min_seeds=1,
        )
        self.assertFalse(audit_bad.passed)
        self.assertIn("obs_rms_unfrozen_on_eval", audit_bad.red_flags)

        # 2. Saturated bang-bang continuous policy (+1.0 every step in Pendulum) must trigger policy_action_saturation
        sat_sb3 = _FakeSB3Model(np.array([2.0], dtype=np.float32))
        audit_sat = audit_external_policy(
            "Pendulum-v1",
            sat_sb3,
            algo_id="sb3_saturated",
            eval_episodes=2,
            min_seeds=1,
        )
        self.assertFalse(audit_sat.passed)
        self.assertIn("policy_action_saturation", audit_sat.red_flags)

        # 3. PyTorch nn.Module + callable evaluated cleanly inside unified evaluate_drl_suite
        class _TorchPolicy(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # CartPole upright tilt heuristic encoded in linear weights: move in sign(theta + 0.1*theta_dot)
                w = torch.tensor([[0.0, 0.0, 1.0, 0.15]], dtype=x.dtype, device=x.device)
                score = (x @ w.T).squeeze(-1)
                return torch.stack([-score, score], dim=-1)

        frozen_vn = _FakeVecNormalize(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32), training=False)
        good_sb3 = _FakeSB3Model(0, vec_normalize=frozen_vn)
        suite = evaluate_drl_suite(
            "CartPole-v1",
            algos=["ppo"],
            seeds=(0, 1),
            min_seeds=2,
            algo_configs={"ppo": PPOConfig(total_timesteps=256, n_steps=128, eval_episodes=3)},
            external_policies={"torch_pd": [_TorchPolicy(), _TorchPolicy()], "sb3_const": good_sb3},
            eval_episodes=3,
            task_name="ExternalInteropTest",
        )
        by_id = {s.algo_id: s for s in suite.algo_summaries}
        self.assertIn("ppo", by_id)
        self.assertIn("torch_pd", by_id)
        self.assertIn("sb3_const", by_id)
        self.assertGreater(by_id["torch_pd"].seed_stats["iqm"], 100.0)
        self.assertTrue(len(suite.pairwise_improvement_probabilities()) >= 3)

        with tempfile.TemporaryDirectory() as tmp:
            files = suite.save(tmp)
            md = files["paper_summary_md"].read_text(encoding="utf-8")
            tex = files["paper_table_tex"].read_text(encoding="utf-8")
            self.assertIn("Pairwise Probability of Improvement", md)
            self.assertIn("IQM (95\\% CI)", tex)
            self.assertTrue(files["overview_figure"].exists())

        ext_runs = evaluate_external_policy("CartPole-v1", _TorchPolicy(), algo_id="torch_pd", eval_episodes=3)
        self.assertEqual(len(ext_runs), 1)
        self.assertGreater(ext_runs[0].eval_stats.mean_return, 100.0)


class TestPhaseCSafeRLAndVerifyGate(unittest.TestCase):
    def test_ppo_lagrangian_sac_lagrangian_and_cvar_ppo_real_risk_control(self) -> None:
        # 1. PPO-Lagrangian: satisfies ep cost_limit=1.0 whereas unconstrained PPO exploits unsafe high-cost arm (cost=8.0)
        r_ppo_uncon = train_ppo(
            _DiscreteConstrainedCMDP,
            config=PPOConfig(total_timesteps=2048, n_steps=128, lr=3e-3, eval_episodes=6, seed=0),
        )
        r_ppo_safe = train_ppo_lagrangian(
            _DiscreteConstrainedCMDP,
            config=PPOLagrangianConfig(
                total_timesteps=3072,
                n_steps=128,
                lr=3e-3,
                cost_critic_lr=3e-3,
                cost_limit=1.0,
                init_lambda=0.5,
                lagrange_lr=0.4,
                eval_episodes=6,
                seed=0,
            ),
        )
        self.assertIsNotNone(r_ppo_safe.eval_stats)
        self.assertGreater(r_ppo_uncon.eval_stats.mean_return, 20.0)
        self.assertLessEqual(r_ppo_safe.train_diagnostics["eval_mean_cost"], 1.5)
        self.assertGreaterEqual(r_ppo_safe.eval_stats.mean_return, 9.0)

        # 2. SAC-Lagrangian (Stooke et al. 2020 PID controller) keeps continuous control cost bounded
        r_sac_safe = train_sac_lagrangian(
            _ContinuousConstrainedCMDP,
            config=SACLagrangianConfig(
                total_timesteps=2500,
                learning_starts=200,
                cost_limit=1.2,
                pid_kp=0.5,
                pid_ki=0.2,
                pid_kd=0.05,
                init_lambda=0.5,
                eval_episodes=6,
                seed=1,
            ),
        )
        self.assertIsNotNone(r_sac_safe.eval_stats)
        self.assertLessEqual(r_sac_safe.train_diagnostics["eval_mean_cost"], 1.45)
        self.assertGreater(r_sac_safe.eval_stats.mean_return, 0.0)

        # 3. CVaR-PPO (Chow et al. 2018) suppresses catastrophic tail crashes vs risk-neutral PPO
        r_cvar = train_cvar_ppo(
            _HeavyTailHazardEnv,
            config=CVaRPPOConfig(
                total_timesteps=6144,
                n_steps=128,
                lr=3e-3,
                cvar_alpha=0.25,
                mean_mix=0.05,
                ent_coef=0.005,
                eval_episodes=30,
                seed=2,
            ),
        )
        assert r_cvar.eval_stats is not None
        cvar_25 = compute_cvar(r_cvar.eval_stats.returns, alpha=0.25)
        self.assertGreaterEqual(cvar_25, 4.0)

        # 4. Planner routing recommends Safe/Risk family when constrained/risk_sensitive
        rec_c = recommend_algo_suite(action_space="continuous", has_constraints=True)
        self.assertIn("sac_lagrangian", rec_c.algos)
        rec_r = recommend_algo_suite(action_space="discrete", risk_sensitive=True)
        self.assertIn("cvar_ppo", rec_r.algos)

    def test_cli_verify_gate_passes_clean_and_blocks_leaked_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Clean interface -> exit 0
            clean_lab = root / "clean_lab"
            scaffold_lab(clean_lab, archetype="discrete_control", mode="gymnasium")
            (clean_lab / "interface.py").write_text(
                "from wannado.drl.workspace import EnvInterface\n"
                "def load_env_interface() -> EnvInterface:\n"
                "    return EnvInterface(env_or_factory='CartPole-v1', task_name='CartPoleClean', domain='control')\n",
                encoding="utf-8",
            )
            out_clean = root / "out_clean"
            rc_pass = cli_main(
                [
                    "verify",
                    "--interface",
                    str(clean_lab / "interface.py"),
                    "--algos",
                    "ppo",
                    "--seeds",
                    "0",
                    "1",
                    "--min-seeds",
                    "2",
                    "--timesteps",
                    "1024",
                    "--eval-episodes",
                    "4",
                    "--out",
                    str(out_clean),
                ]
            )
            self.assertEqual(rc_pass, 0)
            verdict_ok = json.loads((out_clean / "verification_verdict.json").read_text(encoding="utf-8"))
            self.assertTrue(verdict_ok["passed"])

            # Planted future-leakage interface -> blocked immediately with exit code 1!
            leak_lab = root / "leak_lab"
            leak_lab.mkdir()
            (leak_lab / "interface.py").write_text(
                "import gymnasium as gym\n"
                "import numpy as np\n"
                "from gymnasium import spaces\n"
                "from wannado.drl.workspace import EnvInterface\n"
                "class _LeakEnv(gym.Env):\n"
                "    observation_space = spaces.Box(-100.0, 100.0, shape=(2,), dtype=np.float32)\n"
                "    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)\n"
                "    def reset(self, *, seed=None, options=None):\n"
                "        self._rng = np.random.default_rng(seed or 0)\n"
                "        self._p = np.concatenate([[10.0], 10.0 + np.cumsum(self._rng.normal(0.0, 1.0, size=63))]).astype(np.float32)\n"
                "        self._t = 0\n"
                "        return np.array([self._p[0], self._p[1]], dtype=np.float32), {}\n"
                "    def step(self, action):\n"
                "        a = float(np.asarray(action).reshape(-1)[0])\n"
                "        r = float(self._p[self._t + 1] - self._p[self._t])\n"
                "        self._t += 1\n"
                "        return np.array([self._p[self._t], self._p[self._t + 1]], dtype=np.float32), a * r, False, self._t >= 30, {}\n"
                "def load_env_interface() -> EnvInterface:\n"
                "    return EnvInterface(env_or_factory=_LeakEnv, task_name='LeakTrap', domain='finance')\n",
                encoding="utf-8",
            )
            out_leak = root / "out_leak"
            rc_fail = cli_main(
                [
                    "verify",
                    "--interface",
                    str(leak_lab / "interface.py"),
                    "--algos",
                    "ppo",
                    "--out",
                    str(out_leak),
                ]
            )
            self.assertEqual(rc_fail, 1)
            verdict_bad = json.loads((out_leak / "verification_verdict.json").read_text(encoding="utf-8"))
            self.assertFalse(verdict_bad["passed"])
            self.assertEqual(verdict_bad["stage_failed"], "stage1_preflight_and_probes")


if __name__ == "__main__":
    unittest.main()
