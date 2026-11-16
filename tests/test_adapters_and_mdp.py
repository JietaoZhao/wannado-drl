"""Tests for multi-library environment adapters (Gymnasium, legacy Gym, dm_control, PettingZoo) & Stage 0 MDP intake."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wannado.drl.audit import preflight_env
from wannado.drl.core.wrappers import MultiAgentEnvAdapter, UniversalGymAdapter, ensure_gym_env
from wannado.drl.mdp import (
    StateFeatureSpec,
    audit_mdp_spec,
    build_mdp_archetype,
    diagnose_rl_suitability,
)


class _LegacyOpenAIGymMock:
    """Simulates legacy OpenAI Gym 0.21-0.25 4-tuple step() + TimeLimit.truncated."""

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self) -> None:
        self.steps = 0

    def seed(self, s: int) -> None:
        self._rng = np.random.default_rng(s)

    def reset(self) -> np.ndarray:
        self.steps = 0
        return np.zeros(3, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.steps += 1
        done = self.steps >= 3
        info = {"TimeLimit.truncated": True} if done else {}
        return np.full(3, float(action), dtype=np.float32), 1.0, done, info


@dataclass
class _DMTimeStepMock:
    observation: dict[str, np.ndarray]
    reward: float | None
    discount: float | None
    _last: bool

    def last(self) -> bool:
        return self._last


class _DMEnvMock:
    observation_space = spaces.Dict({"pos": spaces.Box(-1, 1, (2,)), "vel": spaces.Box(-1, 1, (1,))})
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self) -> _DMTimeStepMock:
        return _DMTimeStepMock({"pos": np.zeros(2), "vel": np.zeros(1)}, None, None, False)

    def step(self, action: np.ndarray) -> _DMTimeStepMock:
        return _DMTimeStepMock({"pos": np.ones(2) * float(action[0]), "vel": np.zeros(1)}, 0.5, 1.0, True)


class _ParallelMARLMock:
    possible_agents = ["agent_0", "agent_1"]

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return spaces.Discrete(2)

    def state(self) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return {"agent_0": np.zeros(2, dtype=np.float32), "agent_1": np.ones(2, dtype=np.float32)}, {}

    def step(self, actions: dict[str, int]) -> tuple[Any, ...]:
        rew = 1.0 if actions["agent_0"] == actions["agent_1"] else 0.0
        obs = {"agent_0": np.zeros(2, dtype=np.float32), "agent_1": np.ones(2, dtype=np.float32)}
        return obs, {"agent_0": rew, "agent_1": rew}, {"agent_0": False, "agent_1": False}, {"agent_0": True, "agent_1": True}, {}


class TestAdaptersAndStage0MDP(unittest.TestCase):
    def test_legacy_openai_gym_4_tuple_and_timelimit_separation(self) -> None:
        env = ensure_gym_env(_LegacyOpenAIGymMock())
        obs, _ = env.reset(seed=7)
        self.assertEqual(obs.shape, (3,))
        for idx in range(3):
            obs, rew, term, trunc, _ = env.step(1)
            if idx < 2:
                self.assertFalse(term or trunc)
            else:
                self.assertFalse(term, "TimeLimit.truncated must map to truncated=True, terminated=False!")
                self.assertTrue(trunc)

    def test_dm_control_timestep_and_dict_obs_flattening(self) -> None:
        env = UniversalGymAdapter(_DMEnvMock())
        self.assertEqual(env.obs_dim, 3)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (3,))
        obs2, rew, term, trunc, _ = env.step(np.array([0.4], dtype=np.float32))
        self.assertAlmostEqual(rew, 0.5)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertAlmostEqual(float(obs2[0]), 0.4, places=5)

    def test_gymnasium_env_and_multiagent_adapter(self) -> None:
        cart = ensure_gym_env(lambda: gym.make("CartPole-v1"))
        self.assertEqual(cart.detected_library, "gymnasium")
        self.assertTrue(cart.is_discrete)
        self.assertEqual(cart.action_dim, 2)
        cart.close()

        marl = MultiAgentEnvAdapter(_ParallelMARLMock())
        self.assertEqual(marl.num_agents, 2)
        self.assertEqual(marl.state_dim, 4)
        obs_d, gs, _ = marl.reset(seed=0)
        self.assertEqual(gs.shape, (4,))
        _, rews, _, truncs, _, _ = marl.step({"agent_0": 1, "agent_1": 1})
        self.assertEqual(rews["agent_0"], 1.0)
        self.assertTrue(truncs["agent_0"])

    def test_stage0_suitability_and_future_leakage_audit(self) -> None:
        non_rl = diagnose_rl_suitability(has_sequential_coupling=False)
        self.assertFalse(non_rl.is_rl_appropriate)
        self.assertTrue(len(non_rl.alternatives_if_not_rl) >= 1)

        spec = build_mdp_archetype("wireless_scheduling")
        self.assertTrue(audit_mdp_spec(spec).ok)

        # Inject a future-information-leakage state feature and verify audit catches it
        spec.state_features.append(
            StateFeatureSpec(name="next_slot_csi", dim=1, observable_at_decision_time=False)
        )
        rep = audit_mdp_spec(spec)
        self.assertFalse(rep.ok)
        self.assertTrue(any("Future information leakage" in e for e in rep.errors))

    def test_discrete_obs_text_dict_and_marl_factory_preflight(self) -> None:
        # 1. Discrete(16) observation space (FrozenLake-v1 style) -> 16-D one-hot vector
        fl = ensure_gym_env(lambda: gym.make("FrozenLake-v1", is_slippery=False))
        self.assertEqual(fl.obs_dim, 16)
        obs, _ = fl.reset(seed=0)
        self.assertEqual(obs.shape, (16,))
        self.assertAlmostEqual(float(obs[0]), 1.0)
        fl.close()

        # 2. Dict obs containing spaces.Text + custom metadata spaces (MiniGrid MissionSpace style)
        class _CustomMetaSpace(spaces.Space[Any]):
            def __init__(self) -> None:
                super().__init__(shape=None, dtype=None)

            def sample(self, mask: Any = None) -> str:
                return "reach goal"

            def contains(self, x: Any) -> bool:
                return isinstance(x, str)

        class _DictWithTextEnv(gym.Env):
            observation_space = spaces.Dict(
                {
                    "image": spaces.Box(0, 10, shape=(2, 2, 1), dtype=np.uint8),
                    "direction": spaces.Discrete(4),
                    "mission": spaces.Text(max_length=32),
                    "meta": _CustomMetaSpace(),
                }
            )
            action_space = spaces.Discrete(2)

            def reset(self, *, seed: int | None = None, options: dict | None = None):
                return {"image": np.ones((2, 2, 1), dtype=np.uint8), "direction": 2, "mission": "reach goal", "meta": "raw"}, {}

            def step(self, action: int):
                return {"image": np.ones((2, 2, 1), dtype=np.uint8), "direction": 1, "mission": "reach goal", "meta": "raw"}, 1.0, True, False, {}

        wrapped = ensure_gym_env(_DictWithTextEnv())
        self.assertEqual(wrapped.obs_dim, 8)
        dobs, _ = wrapped.reset(seed=0)
        self.assertEqual(dobs.shape, (8,))

        # 3. PettingZoo / MARL factory function in preflight_env + random_baseline computation
        pf_marl = preflight_env(lambda: _ParallelMARLMock(), probe_steps=10, baseline_episodes=3)
        self.assertTrue(pf_marl.ok)
        self.assertEqual(pf_marl.env_library, "multi_agent")
        self.assertIsNotNone(pf_marl.random_baseline)
        self.assertGreaterEqual(pf_marl.random_baseline.mean_return, 0.0)


if __name__ == "__main__":
    unittest.main()
