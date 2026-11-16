"""Real learning & convergence smoke tests across ALL 27 CleanRL-style algorithms in 7 families."""

from __future__ import annotations

import unittest
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wannado.drl import (
    A2CConfig,
    ARPOConfig,
    C51Config,
    CQLConfig,
    DAPOConfig,
    DDPGConfig,
    DiscoRLConfig,
    DoubleDQNConfig,
    DQNConfig,
    DuelingDQNConfig,
    GRPOConfig,
    GSPOConfig,
    HILSERLConfig,
    IQLConfig,
    MADDPGConfig,
    MAPPOConfig,
    MonteCarloConfig,
    OfflineTransitionDataset,
    PPOConfig,
    PPOGRUConfig,
    QLearningConfig,
    QMIXConfig,
    ReinforceConfig,
    SACConfig,
    SarsaConfig,
    SequenceTaskEnv,
    TD3Config,
    TRPOConfig,
    VDNConfig,
    evaluate_random_baseline,
    train_a2c,
    train_arpo,
    train_c51,
    train_cql,
    train_dapo,
    train_ddpg,
    train_discorl,
    train_double_dqn,
    train_dqn,
    train_dueling_dqn,
    train_grpo,
    train_gspo,
    train_hil_serl,
    train_iql_offline,
    train_maddpg,
    train_mappo,
    train_monte_carlo,
    train_ppo,
    train_ppo_gru,
    train_q_learning,
    train_qmix,
    train_reinforce,
    train_sac,
    train_sarsa,
    train_td3,
    train_trpo,
    train_vdn,
)


class _DiscreteTargetGridEnv(gym.Env):
    """5-state discrete MDP: goal is state 4 (+2.0 reward), step cost -0.1, action 1 moves right."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(2)
        self.state = 0
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = 0
        self.steps = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        self.state = min(4, self.state + 1) if int(action) == 1 else max(0, self.state - 1)
        term = self.state == 4
        trunc = self.steps >= 15 and not term
        rew = 2.0 if term else -0.1
        return self.state, rew, term, trunc, {}


class _DiscreteMaskedSignEnv(gym.Env):
    """Fast 2-dim continuous-obs, 3-action discrete env with action 2 masked out as invalid."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        self.state = self.np_random.choice([-1.0, 1.0], size=2).astype(np.float32)
        return self.state.copy(), {"action_mask": np.array([True, True, False], dtype=np.bool_)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        assert int(action) in (0, 1), f"Masked invalid action {action} was selected!"
        desired = 1 if self.state[0] > 0 else 0
        rew = 1.0 if int(action) == desired else -1.0
        self.state = self.np_random.choice([-1.0, 1.0], size=2).astype(np.float32)
        trunc = self.steps >= 12
        return self.state.copy(), rew, False, trunc, {"action_mask": np.array([True, True, False], dtype=np.bool_)}


class _ContinuousPointGoalEnv(gym.Env):
    """Fast 2D continuous tracking MDP: steer position toward origin in [-1, 1]^2."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.pos = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        self.pos = self.np_random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        return self.pos.copy(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.pos = np.clip(self.pos + 0.35 * a, -2.0, 2.0).astype(np.float32)
        dist = float(np.linalg.norm(self.pos))
        rew = float(1.0 - dist)
        trunc = self.steps >= 12
        return self.pos.copy(), rew, False, trunc, {}


class _CoopMultiAgentDiscreteEnv:
    """2-agent cooperative match game: team gets +1.0 when both agents match the hidden target bit."""

    possible_agents = ["a0", "a1"]

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._target = 0
        self._steps = 0

    def observation_space(self, _agent: str) -> spaces.Space[Any]:
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def action_space(self, _agent: str) -> spaces.Space[Any]:
        return spaces.Discrete(2)

    def state(self) -> np.ndarray:
        t = float(self._target)
        return np.array([t, 1.0 - t, t, 1.0 - t], dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._target = int(self._rng.integers(0, 2))
        v = np.array([1.0, -1.0] if self._target == 1 else [-1.0, 1.0], dtype=np.float32)
        return {"a0": v.copy(), "a1": v.copy()}, {}

    def step(self, actions: dict[str, int]) -> tuple[Any, ...]:
        self._steps += 1
        rew = 1.0 if (actions["a0"] == self._target and actions["a1"] == self._target) else -0.5
        self._target = int(self._rng.integers(0, 2))
        v = np.array([1.0, -1.0] if self._target == 1 else [-1.0, 1.0], dtype=np.float32)
        obs = {"a0": v.copy(), "a1": v.copy()}
        trunc = self._steps >= 10
        return obs, {"a0": rew, "a1": rew}, {"a0": False, "a1": False}, {"a0": trunc, "a1": trunc}, {}


class _CoopMultiAgentContinuousEnv:
    """2-agent continuous team tracking: move both scalars toward target signal."""

    possible_agents = ["c0", "c1"]

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._target = np.zeros(2, dtype=np.float32)
        self._steps = 0

    def observation_space(self, _agent: str) -> spaces.Space[Any]:
        return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def action_space(self, _agent: str) -> spaces.Space[Any]:
        return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def state(self) -> np.ndarray:
        return self._target.copy()

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._target = self._rng.choice([-0.8, 0.8], size=2).astype(np.float32)
        return {"c0": self._target[0:1], "c1": self._target[1:2]}, {}

    def step(self, actions: dict[str, np.ndarray]) -> tuple[Any, ...]:
        self._steps += 1
        a0 = float(np.asarray(actions["c0"]).reshape(-1)[0])
        a1 = float(np.asarray(actions["c1"]).reshape(-1)[0])
        err = abs(a0 - float(self._target[0])) + abs(a1 - float(self._target[1]))
        rew = 1.0 - err
        self._target = self._rng.choice([-0.8, 0.8], size=2).astype(np.float32)
        trunc = self._steps >= 10
        return (
            {"c0": self._target[0:1], "c1": self._target[1:2]},
            {"c0": rew, "c1": rew},
            {"c0": False, "c1": False},
            {"c0": trunc, "c1": trunc},
            {},
        )


class TestAll27AlgorithmsConvergence(unittest.TestCase):
    def test_1_tabular_classic_family_converges(self) -> None:
        env = _DiscreteTargetGridEnv()
        rand = evaluate_random_baseline(env, num_episodes=8, seed=0)
        mc = train_monte_carlo(env, MonteCarloConfig(total_timesteps=1500, seed=1))
        sa = train_sarsa(env, SarsaConfig(total_timesteps=1500, seed=1))
        ql = train_q_learning(env, QLearningConfig(total_timesteps=1500, seed=1))
        for res in (mc, sa, ql):
            self.assertIsNotNone(res.eval_stats)
            self.assertGreater(res.eval_stats.mean_return, rand.mean_return + 0.5)

    def test_2_value_based_family_converges_with_action_masks(self) -> None:
        env = _DiscreteMaskedSignEnv()
        rand = evaluate_random_baseline(env, num_episodes=8, seed=0)
        r_dqn = train_dqn(env, DQNConfig(total_timesteps=1800, learning_starts=120, exploration_fraction=0.4, seed=2))
        r_ddqn = train_double_dqn(env, DoubleDQNConfig(total_timesteps=1800, learning_starts=120, exploration_fraction=0.4, seed=2))
        r_duel = train_dueling_dqn(env, DuelingDQNConfig(total_timesteps=1800, learning_starts=120, exploration_fraction=0.4, seed=2))
        r_c51 = train_c51(env, C51Config(total_timesteps=1800, learning_starts=120, v_min=-15.0, v_max=15.0, seed=2))
        for res in (r_dqn, r_ddqn, r_duel, r_c51):
            self.assertGreater(res.eval_stats.mean_return, rand.mean_return + 4.0)

    def test_3_policy_gradient_family_converges(self) -> None:
        env = _DiscreteMaskedSignEnv()
        rand = evaluate_random_baseline(env, num_episodes=8, seed=0)
        r_reinf = train_reinforce(env, ReinforceConfig(total_timesteps=1800, lr=5e-3, seed=3))
        r_a2c = train_a2c(env, A2CConfig(total_timesteps=1800, lr=3e-3, seed=3))
        r_trpo = train_trpo(env, TRPOConfig(total_timesteps=2000, n_steps=256, seed=3))
        r_ppo = train_ppo(env, PPOConfig(total_timesteps=2000, n_steps=128, lr=2e-3, normalize_obs=True, seed=3))
        r_gru = train_ppo_gru(env, PPOGRUConfig(total_timesteps=2000, n_steps=64, lr=2e-3, normalize_obs=True, seed=3))
        for res in (r_reinf, r_a2c, r_trpo, r_ppo, r_gru):
            self.assertGreater(res.eval_stats.mean_return, rand.mean_return + 3.0)
            self.assertTrue(res.obs_rms_frozen_on_eval)

    def test_4_off_policy_continuous_and_hil_serl_converge(self) -> None:
        env = _ContinuousPointGoalEnv()
        rand = evaluate_random_baseline(env, num_episodes=8, seed=0)
        r_ddpg = train_ddpg(env, DDPGConfig(total_timesteps=1800, learning_starts=150, seed=4))
        r_td3 = train_td3(env, TD3Config(total_timesteps=1800, learning_starts=150, seed=4))
        r_sac = train_sac(env, SACConfig(total_timesteps=1800, learning_starts=150, seed=4))

        # HIL-SERL with synthetic offline demos + corrective intervention oracle
        demos = OfflineTransitionDataset.collect_from_env(
            env, num_steps=400, policy_fn=lambda obs, _: np.clip(-2.5 * obs, -1.0, 1.0), seed=4
        )
        r_hil = train_hil_serl(
            env,
            HILSERLConfig(total_timesteps=1200, learning_starts=50, utd_ratio=2, seed=4),
            demo_dataset=demos,
            intervention_fn=lambda obs, _act, _info: (bool(np.linalg.norm(obs) > 1.1), np.clip(-2.5 * obs, -1.0, 1.0)),
        )
        for res in (r_ddpg, r_td3, r_sac, r_hil):
            self.assertGreater(res.eval_stats.mean_return, rand.mean_return + 2.0)

    def test_5_offline_rl_family_learns_from_static_data(self) -> None:
        env = _DiscreteMaskedSignEnv()
        # Behavior policy with 85% optimal + 15% random transitions (strip mask for simple dataset check)
        class _CleanSignEnv(gym.Env):
            observation_space = env.observation_space
            action_space = spaces.Discrete(2)

            def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
                o, _ = env.reset(seed=seed)
                return o, {}

            def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
                o, r, tm, tr, _ = env.step(action)
                return o, r, tm, tr, {}

        clean_env = _CleanSignEnv()
        rand = evaluate_random_baseline(clean_env, num_episodes=8, seed=0)
        rng_col = np.random.default_rng(5)
        ds = OfflineTransitionDataset.collect_from_env(
            clean_env,
            num_steps=800,
            policy_fn=lambda obs, _: (1 if obs[0] > 0 else 0) if rng_col.random() < 0.85 else int(rng_col.integers(0, 2)),
            seed=5,
        )
        r_cql = train_cql(ds, CQLConfig(num_grad_steps=600, cql_alpha=0.5, seed=5), eval_env_or_factory=clean_env)
        r_iql = train_iql_offline(ds, IQLConfig(num_grad_steps=600, seed=5), eval_env_or_factory=clean_env)
        self.assertGreater(r_cql.eval_stats.mean_return, rand.mean_return + 5.0)
        self.assertGreater(r_iql.eval_stats.mean_return, rand.mean_return + 5.0)

    def test_6_multi_agent_family_converges(self) -> None:
        d_env = _CoopMultiAgentDiscreteEnv()
        r_vdn = train_vdn(d_env, VDNConfig(total_timesteps=2000, learning_starts=120, lr=1e-3, seed=6))
        r_qmix = train_qmix(d_env, QMIXConfig(total_timesteps=2000, learning_starts=120, lr=1e-3, seed=6))
        r_mappo = train_mappo(d_env, MAPPOConfig(total_timesteps=2000, n_steps=64, lr=3e-3, seed=6))
        for res in (r_vdn, r_qmix, r_mappo):
            self.assertGreater(res.eval_stats.mean_return, 3.0)

        c_env = _CoopMultiAgentContinuousEnv()
        r_maddpg = train_maddpg(c_env, MADDPGConfig(total_timesteps=1800, learning_starts=150, lr=2e-3, seed=6))
        self.assertGreater(r_maddpg.eval_stats.mean_return, 2.0)

    def test_7_frontier_sequence_and_discorl_converge(self) -> None:
        task = SequenceTaskEnv(vocab_size=6, prompt_len=4, max_response_len=4)
        r_grpo = train_grpo(task, GRPOConfig(total_updates=100, prompts_per_batch=8, group_size=8, kl_coef=0.01, lr=8e-3, seed=7))
        r_dapo = train_dapo(task, DAPOConfig(total_updates=100, prompts_per_batch=8, group_size=8, lr=8e-3, seed=7))
        r_gspo = train_gspo(task, GSPOConfig(total_updates=100, prompts_per_batch=8, group_size=8, kl_coef=0.01, lr=8e-3, seed=7))
        r_arpo = train_arpo(task, ARPOConfig(total_updates=100, prompts_per_batch=6, group_size=6, kl_coef=0.01, lr=8e-3, seed=7))
        # Random 6-vocab sequence gets ~1/6 = 0.167 token acc (~0.083 reward); trained models should strongly beat 0.70
        for res in (r_grpo, r_dapo, r_gspo, r_arpo):
            self.assertGreater(res.eval_stats.mean_return, 0.70, f"{res.algo_id} failed to converge!")

        disc_env = _DiscreteMaskedSignEnv()
        rand = evaluate_random_baseline(disc_env, num_episodes=8, seed=0)
        r_disco = train_discorl(disc_env, DiscoRLConfig(total_timesteps=2200, n_steps=32, inner_lr=3e-3, meta_train=True, seed=7))
        self.assertGreater(r_disco.eval_stats.mean_return, rand.mean_return + 3.0)


if __name__ == "__main__":
    unittest.main()
