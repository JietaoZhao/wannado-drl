"""Regression tests covering v1.0.3 TimeLimit GAE, rigor audit, action scaling, and exports."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import wannado.drl as wdrl
from wannado.drl import (
    PPOConfig,
    TrainResult,
    audit_drl_rigor,
    evaluate_drl_suite,
    train_ppo,
)
from wannado.drl import (
    algos as wdrl_algos,
)
from wannado.drl.core.buffers import RolloutBuffer
from wannado.drl.core.types import EpisodeEvalStats
from wannado.drl.core.utils import compute_gae


class _AsymmetricPowerControlEnv(gym.Env):
    """Single-step continuous control on Box([0], [10]) with optimum at a = 8.0."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([0.8, -0.4], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        a = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], 0.0, 10.0))
        reward = -((a - 8.0) ** 2)
        trunc = self.steps >= 10
        return np.array([0.8, -0.4], dtype=np.float32), reward, False, trunc, {}


class _TwoStepTruncatedEnv(gym.Env):
    """Truncates cleanly at step 2 with zero reward until truncation boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.ep = 0
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.ep += 1
        return np.array([float(self.ep % 2), 1.0], dtype=np.float32), {}

    def step(self, action: int):
        self.t += 1
        trunc = self.t >= 2
        return np.array([float(action), 1.0], dtype=np.float32), 1.0, False, trunc, {}


class TestReleaseV103Regressions(unittest.TestCase):
    def test_top_level_and_algo_exports_match_all(self) -> None:
        self.assertTrue(set(wdrl._LAZY_EXPORTS.keys()).issubset(set(wdrl.__all__)))
        self.assertEqual(set(wdrl_algos.ALGO_EXPORTS.keys()), set(wdrl_algos.__all__))

    def test_rollout_buffer_and_compute_gae_prevent_timelimit_cross_episode_leakage(self) -> None:
        buf = RolloutBuffer(num_steps=4, obs_dim=2, action_shape=(), action_dtype=np.dtype(np.int64))
        gamma = 0.99
        v_ep1_terminal = 5.0
        # Episode 1 steps 0..1 truncated at step 1: pre-bootstrap reward += gamma * V(next_obs), done=True
        buf.add(np.zeros(2, dtype=np.float32), 0, -0.5, 1.0, False, 1.0, done=False)
        buf.add(np.zeros(2, dtype=np.float32), 1, -0.5, 1.0 + gamma * v_ep1_terminal, False, 2.0, done=True)
        # Episode 2 starts at step 2 with huge reward +100.0; must NOT leak backward into Episode 1!
        buf.add(np.ones(2, dtype=np.float32), 0, -0.5, 100.0, True, 0.5, done=True)
        buf.compute_returns_and_advantage(next_value=999.0, next_terminated=False, gamma=gamma, gae_lambda=0.95)

        expected_ret_ep1_step1 = 1.0 + gamma * v_ep1_terminal
        self.assertAlmostEqual(float(buf.returns[1]), expected_ret_ep1_step1, places=4)
        self.assertLess(float(buf.returns[0]), 10.0)

        adv, ret = compute_gae(
            rewards=np.array([1.0, 1.0 + gamma * v_ep1_terminal, 100.0], dtype=np.float32),
            values=np.array([1.0, 2.0, 0.5], dtype=np.float32),
            terminateds=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            next_value=999.0,
            next_terminated=False,
            gamma=gamma,
            gae_lambda=0.95,
            dones=np.array([0.0, 1.0, 1.0], dtype=np.float32),
            next_done=True,
        )
        self.assertAlmostEqual(float(ret[1]), expected_ret_ep1_step1, places=4)
        self.assertLess(float(ret[0]), 10.0)
        self.assertAlmostEqual(float(adv[2]), 99.5, places=4)

        # Ensure PPO finishes cleanly on a 2-step truncated environment
        res = train_ppo(_TwoStepTruncatedEnv, config=PPOConfig(total_timesteps=32, n_steps=16, eval_episodes=2, seed=0))
        self.assertGreaterEqual(res.total_timesteps, 16)

    def test_rigor_auditor_catches_fallback_entropy_q_explosion_and_single_seed_suite(self) -> None:
        # 1. Discrete/non-SAC algorithm emitting only mean_entropy=0.0 must trigger policy_entropy_collapse
        for aid in ("a2c", "mappo", "grpo"):
            runs = [
                TrainResult(
                    algo_id=aid,
                    seed=s,
                    total_timesteps=1000,
                    elapsed_seconds=0.1,
                    policy_fn=lambda o, _: 0,
                    model=None,
                    config={},
                    learning_curve=[],
                    train_diagnostics={"mean_entropy": 0.0},
                    eval_stats=EpisodeEvalStats([10.0], [10], 10.0, 0.0, 10.0),
                    obs_rms_frozen_on_eval=True,
                )
                for s in (0, 1, 2)
            ]
            audit = audit_drl_rigor(runs, min_seeds=3)
            self.assertFalse(audit.passed)
            self.assertIn("policy_entropy_collapse", audit.red_flags)

        # 2. Healthy negative SAC target entropy (-1.0) must NOT falsely flag entropy collapse
        sac_ok = [
            TrainResult(
                algo_id="sac",
                seed=s,
                total_timesteps=1000,
                elapsed_seconds=0.1,
                policy_fn=lambda o, _: np.array([0.0]),
                model=None,
                config={},
                learning_curve=[],
                train_diagnostics={"mean_q_value": -8.0, "max_q_value": -12.0, "final_entropy": -1.0, "is_continuous": 1.0},
                eval_stats=EpisodeEvalStats([5.0], [10], 5.0, 0.0, 10.0),
                obs_rms_frozen_on_eval=True,
            )
            for s in (0, 1, 2)
        ]
        self.assertTrue(audit_drl_rigor(sac_ok, min_seeds=3).passed)

        # 3. Off-policy Q explosion without max_q_value (only mean_q_value=1e9) must trigger q_value_explosion
        q_bad = [
            TrainResult(
                algo_id="sac",
                seed=s,
                total_timesteps=1000,
                elapsed_seconds=0.1,
                policy_fn=lambda o, _: np.array([0.0]),
                model=None,
                config={},
                learning_curve=[],
                train_diagnostics={"mean_q_value": 1e9, "final_entropy": -0.5, "is_continuous": 1.0},
                eval_stats=EpisodeEvalStats([5.0], [10], 5.0, 0.0, 10.0),
                obs_rms_frozen_on_eval=True,
            )
            for s in (0, 1, 2)
        ]
        self.assertIn("q_value_explosion", audit_drl_rigor(q_bad, min_seeds=3).red_flags)

        # 4. Single-seed evaluate_drl_suite must surface single_seed_survivorship_risk by default
        rep = evaluate_drl_suite(
            lambda: gym.make("CartPole-v1"),
            algos=["ppo"],
            seeds=[0],
            algo_configs={"ppo": PPOConfig(total_timesteps=128, n_steps=64, eval_episodes=2)},
            run_preflight=False,
            run_audit=True,
        )
        self.assertIsNotNone(rep.algo_summaries[0].audit)
        assert rep.algo_summaries[0].audit is not None
        self.assertFalse(rep.algo_summaries[0].audit.passed)
        self.assertIn("single_seed_survivorship_risk", rep.algo_summaries[0].audit.red_flags)

    def test_ppo_handles_asymmetric_continuous_box_bounds(self) -> None:
        res = train_ppo(
            _AsymmetricPowerControlEnv,
            config=PPOConfig(total_timesteps=2000, n_steps=200, lr=1e-3, eval_episodes=3, seed=42),
        )
        assert res.eval_stats is not None
        # Random uniform on [0, 10] averages ~ -370 over 10 steps; bound-aware PPO improves far past -150
        self.assertGreater(res.eval_stats.mean_return, -150.0)


if __name__ == "__main__":
    unittest.main()
