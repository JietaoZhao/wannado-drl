"""Regression tests covering v1.0.3 TimeLimit GAE, rigor audit, action scaling, and exports."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import wannado.drl as wdrl
from wannado.drl import (
    A2CConfig,
    CQLConfig,
    DQNConfig,
    IQLConfig,
    MAPPOConfig,
    OfflineTransitionDataset,
    PortfolioSimplexWrapper,
    PPOConfig,
    PPOGRUConfig,
    QMIXConfig,
    ReinforceConfig,
    TrainResult,
    TRPOConfig,
    apply_simplex_projection,
    audit_drl_rigor,
    build_interactive_brief,
    build_mdp_archetype,
    compute_cvar,
    compute_sharpe_and_drawdown,
    evaluate_drl_suite,
    evaluate_offline_ope,
    preflight_env,
    recommend_algo_suite,
    scan_project,
    train_a2c,
    train_cql,
    train_dqn,
    train_iql_offline,
    train_mappo,
    train_ppo,
    train_ppo_gru,
    train_qmix,
    train_reinforce,
    train_trpo,
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


class _LegacyDuckDictSpace:
    """Mimics legacy gym.spaces.Dict whose class does not inherit gymnasium.spaces.Space."""

    def __init__(self, sub: dict[str, object]) -> None:
        self.spaces = sub


class _LegacyDuckDiscreteSpace:
    """Mimics legacy gym.spaces.Discrete."""

    shape = ()

    def __init__(self, n: int) -> None:
        self.n = n


class _LegacyDuckBoxSpace:
    """Mimics legacy gym.spaces.Box."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = low
        self.high = high
        self.shape = low.shape


class _Legacy4TupleEnv:
    """Legacy OpenAI gym 4-tuple env whose spaces trigger flatdim() failure in naive wrappers."""

    def __init__(self) -> None:
        self.observation_space = _LegacyDuckDictSpace(
            {
                "state": _LegacyDuckBoxSpace(np.full(2, -5.0, np.float32), np.full(2, 5.0, np.float32)),
                "regime": _LegacyDuckDiscreteSpace(3),
            }
        )
        self.action_space = _LegacyDuckDiscreteSpace(2)
        self.t = 0

    def seed(self, s: int) -> None:
        self._s = s

    def reset(self):
        self.t = 0
        return {"state": np.array([0.5, -0.5], dtype=np.float32), "regime": 1}

    def step(self, action: int):
        self.t += 1
        done = self.t >= 4
        info = {"TimeLimit.truncated": True} if done else {}
        return {"state": np.array([float(action), 0.1], dtype=np.float32), "regime": int(action) % 3}, 1.0, done, info


class _MultiDiscreteSmallEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 4])
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        arr = np.asarray(action)
        assert arr.shape == (2,)
        self.t += 1
        rew = 1.0 if (int(arr[0]) == 2 and int(arr[1]) == 1) else 0.0
        return np.zeros(4, dtype=np.float32), rew, False, self.t >= 4, {}


class _MultiDiscreteLargeEnv(gym.Env):
    """High-cardinality MultiDiscrete (prod = 625 > 256) testing native factorized heads."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([5, 5, 5, 5])
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        arr = np.asarray(action)
        assert arr.shape == (4,)
        self.t += 1
        return np.ones(4, dtype=np.float32) * 0.1, float(np.sum(arr == 2)), False, self.t >= 4, {}


class _StrictMultiBinaryEnv(gym.Env):
    def __init__(self, n_bits: int = 4) -> None:
        super().__init__()
        self.n_bits = n_bits
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(n_bits)
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        arr = np.asarray(action)
        assert arr.shape == (self.n_bits,), f"Expected binary shape ({self.n_bits},), got {arr.shape}"
        assert np.all((arr == 0) | (arr == 1))
        self.t += 1
        return np.ones(3, dtype=np.float32), float(np.sum(arr)), False, self.t >= 4, {}


class _AsymmetricMARLEnv:
    """Two-agent env with heterogeneous observation dimensions (3 vs 7)."""

    possible_agents = ("leader", "follower")

    def observation_space(self, agent: str):
        dim = 3 if agent == "leader" else 7
        return spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)

    def action_space(self, agent: str):
        return spaces.Discrete(2)

    def reset(self, seed: int | None = None):
        self.t = 0
        return {"leader": np.ones(3, dtype=np.float32), "follower": np.ones(7, dtype=np.float32)}, {}

    def step(self, actions):
        self.t += 1
        trunc = self.t >= 4
        return (
            {"leader": np.ones(3, dtype=np.float32), "follower": np.ones(7, dtype=np.float32)},
            {"leader": 1.0, "follower": 0.5},
            {"leader": False, "follower": False},
            {"leader": trunc, "follower": trunc},
            {},
        )


class _ToyPortfolioEnv(gym.Env):
    def __init__(self, n_assets: int = 3) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_assets,), dtype=np.float32)
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        w = np.asarray(action, dtype=np.float32)
        assert abs(float(np.sum(w)) - 1.0) < 1e-4 and np.all(w >= -1e-6)
        self.t += 1
        ret = float(np.dot(w, np.array([0.05, 0.01, -0.02], dtype=np.float32)))
        return np.zeros(4, dtype=np.float32), ret, False, self.t >= 5, {}


class TestReleaseV104CompatibilityAndOfflineSuite(unittest.TestCase):
    def test_legacy_duck_spaces_and_4tuple_env(self) -> None:
        pf = preflight_env(_Legacy4TupleEnv, probe_steps=12, baseline_episodes=2)
        self.assertTrue(pf.ok)
        self.assertEqual(pf.obs_dim, 5)
        self.assertEqual(pf.action_dim, 2)
        res = train_ppo(_Legacy4TupleEnv, config=PPOConfig(total_timesteps=32, n_steps=16, eval_episodes=2, seed=0))
        assert res.eval_stats is not None
        self.assertAlmostEqual(res.eval_stats.mean_return, 4.0, places=3)

    def test_multidiscrete_multibinary_and_sync_vector_env(self) -> None:
        pf_small = preflight_env(_MultiDiscreteSmallEnv, probe_steps=12, baseline_episodes=2)
        self.assertTrue(pf_small.ok)
        self.assertEqual(pf_small.action_space_kind, "multi_discrete")
        self.assertEqual(pf_small.summary["action_nvec"], [3, 4])
        res_dqn = train_dqn(_MultiDiscreteSmallEnv, config=DQNConfig(total_timesteps=64, learning_starts=8, eval_episodes=2, seed=0))
        self.assertIsNotNone(res_dqn.eval_stats)

        pf_large = preflight_env(_MultiDiscreteLargeEnv, probe_steps=12, baseline_episodes=2)
        self.assertTrue(pf_large.ok)
        for runner, cfg in (
            (train_ppo, PPOConfig(total_timesteps=32, n_steps=16, eval_episodes=2, seed=0)),
            (train_a2c, A2CConfig(total_timesteps=32, n_steps=16, eval_episodes=2, seed=0)),
            (train_trpo, TRPOConfig(total_timesteps=32, n_steps=16, eval_episodes=2, seed=0)),
            (train_reinforce, ReinforceConfig(total_timesteps=32, eval_episodes=2, seed=0)),
            (train_ppo_gru, PPOGRUConfig(total_timesteps=32, n_steps=16, eval_episodes=2, seed=0)),
        ):
            r = runner(_MultiDiscreteLargeEnv, config=cfg)
            self.assertIsNotNone(r.eval_stats)

        # MultiBinary must always emit exact binary vector shapes, never scalar int into raw_env.step()
        self.assertIsNotNone(train_dqn(lambda: _StrictMultiBinaryEnv(4), config=DQNConfig(total_timesteps=64, learning_starts=8, eval_episodes=2)).eval_stats)
        self.assertIsNotNone(train_ppo(lambda: _StrictMultiBinaryEnv(10), config=PPOConfig(total_timesteps=32, n_steps=16, eval_episodes=2)).eval_stats)

        # Gymnasium SyncVectorEnv end-to-end
        vec = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(2)])
        try:
            pf_vec = preflight_env(vec, probe_steps=10, baseline_episodes=2)
            self.assertTrue(pf_vec.ok)
            self.assertEqual(pf_vec.env_library, "gymnasium_vector")
            self.assertEqual(pf_vec.summary.get("num_envs"), 2)
            r_vec = train_ppo(vec, config=PPOConfig(total_timesteps=32, n_steps=16, eval_episodes=2, seed=0))
            self.assertIsNotNone(r_vec.eval_stats)
        finally:
            vec.close()

    def test_asymmetric_marl_padded_cleanly(self) -> None:
        r_mappo = train_mappo(_AsymmetricMARLEnv, config=MAPPOConfig(total_timesteps=32, n_steps=16, eval_episodes=2, seed=0))
        r_qmix = train_qmix(_AsymmetricMARLEnv, config=QMIXConfig(total_timesteps=64, learning_starts=8, eval_episodes=2, seed=0))
        self.assertIsNotNone(r_mappo.eval_stats)
        self.assertIsNotNone(r_qmix.eval_stats)

    def test_offline_dataset_suite_ope_audit_and_action_masks(self) -> None:
        rng = np.random.default_rng(7)
        n = 120
        obs = rng.normal(size=(n, 5)).astype(np.float32)
        next_obs = rng.normal(size=(n, 5)).astype(np.float32)
        actions = rng.integers(0, 3, size=n, dtype=np.int64)
        rewards = rng.normal(loc=0.4, scale=0.3, size=n).astype(np.float32)
        terms = np.zeros(n, dtype=np.float32)
        terms[9::10] = 1.0
        masks = np.ones((n, 3), dtype=np.bool_)
        masks[:, 2] = False  # Clinically forbidden action index 2
        actions = np.clip(actions, 0, 1)
        groups = np.where((np.arange(n) // 10) % 2 == 0, "cohort_a", "cohort_b")

        ds = OfflineTransitionDataset.from_arrays(
            obs,
            actions,
            rewards,
            next_obs,
            terms,
            is_discrete=True,
            action_dim=3,
            action_masks=masks,
            next_action_masks=masks,
            behavior_probs=np.full(n, 0.5, dtype=np.float32),
            group_labels=groups,
        )
        pf = preflight_env(ds)
        self.assertTrue(pf.ok)
        self.assertEqual(pf.env_library, "offline_dataset")
        self.assertTrue(pf.summary["saw_action_mask"])

        cql_res = train_cql(ds, config=CQLConfig(num_grad_steps=40, batch_size=16, seed=0))
        self.assertIn(int(cql_res.policy_fn(obs[0], {"action_mask": masks[0]})), (0, 1))
        iql_res = train_iql_offline(ds, config=IQLConfig(num_grad_steps=40, batch_size=16, seed=0))
        self.assertIn("max_q_value", iql_res.train_diagnostics)

        ope = evaluate_offline_ope(ds, cql_res.policy_fn, fqe_steps=20)
        self.assertIn("wis", ope)
        self.assertIn("fqe", ope)
        self.assertEqual(set(ope["subgroups"].keys()), {"cohort_a", "cohort_b"})

        with warnings.catch_warnings(record=True) as w_rec:
            warnings.simplefilter("always", RuntimeWarning)
            report = evaluate_drl_suite(
                ds,
                algos=["cql", "iql_offline"],
                seeds=(0, 1, 2),
                algo_configs={
                    "cql": CQLConfig(num_grad_steps=30, batch_size=16),
                    "iql_offline": IQLConfig(num_grad_steps=30, batch_size=16),
                },
                task_name="Clinical #1 Sepsis Study (95% CI & cohort_a^2 ~ 0)",
                run_audit=True,
            )
            empty_slice_warnings = [w for w in w_rec if "Mean of empty slice" in str(w.message)]
            self.assertEqual(len(empty_slice_warnings), 0)

        for summary in report.algo_summaries:
            self.assertIsNotNone(summary.audit)
            assert summary.audit is not None
            self.assertTrue(summary.audit.passed, f"Unexpected audit failure: {summary.audit.red_flags}")

        tex = report.to_latex()
        self.assertIn("\\#1", tex)
        self.assertIn("\\^{}", tex)
        self.assertIn("\\~{}", tex)

    def test_finance_portfolio_simplex_cvar_and_copilot_domains(self) -> None:
        w = apply_simplex_projection(np.array([2.5, -1.0, 0.5]))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=5)
        self.assertTrue(np.all(w >= 0.0))

        self.assertAlmostEqual(compute_cvar([-10.0, -2.0, 1.0, 5.0], alpha=0.25), -10.0, places=4)
        risk = compute_sharpe_and_drawdown([0.02, -0.05, 0.03, 0.01])
        self.assertGreater(risk["max_drawdown"], 0.0)

        def env_fn():
            return PortfolioSimplexWrapper(_ToyPortfolioEnv(3), turnover_cost=0.001)

        r = train_ppo(env_fn, config=PPOConfig(total_timesteps=40, n_steps=20, eval_episodes=2, seed=0))
        assert r.eval_stats is not None
        self.assertIn("portfolio_turnover", r.eval_stats.task_metrics_mean)

        for arch in ("finance_portfolio_risk", "clinical_offline_safety", "cyber_attack_defense"):
            spec = build_mdp_archetype(arch)
            self.assertTrue(len(spec.task_kpis) >= 1)

        brief = build_interactive_brief("finance portfolio rebalancing with CVaR tail risk")
        self.assertEqual(brief["recommended_suite"]["action_space"], "continuous")
        self.assertIn("sac", brief["recommended_suite"]["frontier_sota"] + brief["recommended_suite"]["algos"])

        rec = recommend_algo_suite(task_goal="clinical sepsis offline logged dataset")
        self.assertEqual(rec.paradigm, "offline")

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "portfolio_env.py"
            p.write_text(
                "from gymnasium import spaces\nclass StockTradingEnv:\n    action_space = spaces.Box(-1, 1, shape=(3,))\n    # portfolio backtest with sharpe and cvar\n",
                encoding="utf-8",
            )
            scan = scan_project(tmp)
            self.assertEqual(scan.likely_archetype, "finance_portfolio_risk")
            self.assertEqual(scan.inferred_action_space, "continuous")
            brief_proj = build_interactive_brief(project_scan=scan)
            self.assertEqual(brief_proj["recommended_suite"]["action_space"], "continuous")


if __name__ == "__main__":
    unittest.main()
