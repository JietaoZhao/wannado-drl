"""Regression test suite verifying all v1.1.1 deep audit fixes across wannado-drl."""

from __future__ import annotations

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
    OfflineTransitionDataset,
    PPOConfig,
    SACConfig,
    SACLagrangianConfig,
    SequenceTaskEnv,
    TD3Config,
    TRPOConfig,
    __version__,
    audit_drl_rigor,
    evaluate_drl_suite,
    evaluate_external_policy,
    evaluate_offline_ope,
    evaluate_offline_wis,
    make_vec_env,
    preflight_env,
    run_mdp_probes,
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
    train_ppo,
    train_sac,
    train_sac_lagrangian,
    train_td3,
    train_trpo,
)
from wannado.drl.algos.llm_and_discovered.discorl_oh2025 import DiscoAgentNet, DiscoMetaUpdateNet
from wannado.drl.cli import main as cli_main
from wannado.drl.core.buffers import DualDemoReplayBuffer, ReplayBuffer
from wannado.drl.core.networks import (
    AutoregressiveSeqPolicy,
    ImpalaResNetExtractor,
    SetAttentionExtractor,
    SingleQCritic,
    TwinQCritic,
    build_feature_extractor,
)
from wannado.drl.core.obs_spec import infer_obs_spec
from wannado.drl.core.types import EpisodeEvalStats, TrainResult
from wannado.drl.core.utils import set_global_seed
from wannado.drl.core.wrappers import DictRunningMeanStd, VectorizedEnvAdapter


class _TerminalInfoVecEnv:
    """Minimal duck-typed Gymnasium VectorEnv returning terminal `final_info` on autoreset."""

    def __init__(self, *, dict_of_arrays: bool = False) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.single_action_space = spaces.Discrete(2)
        self._dict_of_arrays = dict_of_arrays
        self._t = np.zeros(2, dtype=np.int64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._t[:] = 0
        return np.zeros((2, 3), dtype=np.float32), {}

    def step(self, actions: np.ndarray):
        del actions
        self._t += 1
        term = np.array([self._t[0] >= 2, False], dtype=np.bool_)
        trunc = np.array([False, False], dtype=np.bool_)
        obs = np.ones((2, 3), dtype=np.float32) * self._t[:, None]
        rew = np.ones(2, dtype=np.float32)
        if self._dict_of_arrays:
            info: dict[str, Any] = {
                "cost": np.array([0.0, 0.2], dtype=np.float32),
                "final_info": {
                    "cost": np.array([7.5, 0.0], dtype=np.float32),
                    "_cost": np.array([True, False], dtype=np.bool_),
                },
                "_final_info": np.array([bool(term[0]), False], dtype=np.bool_),
            }
        else:
            info = {
                "cost": np.array([0.0, 0.2], dtype=np.float32),
                "final_info": np.array([{"cost": 7.5, "terminal_flag": 1.0} if term[0] else None, None], dtype=object),
                "_final_info": np.array([bool(term[0]), False], dtype=np.bool_),
            }
        if term[0]:
            self._t[0] = 0
        return obs, rew, term, trunc, info


class _MaskedTrapDiscreteEnv(gym.Env):
    """Discrete env where action 2 is illegal in non-terminal states and must stay masked in TD targets."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._t = 0
        self._state = np.zeros(3, dtype=np.float32)

    def _mask(self) -> np.ndarray:
        return np.array([True, True, False], dtype=np.bool_)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._t = 0
        self._state = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        return self._state.copy(), {"action_mask": self._mask()}

    def step(self, action: int):
        self._t += 1
        rew = 1.0 if int(action) == 1 else (0.1 if int(action) == 0 else -100.0)
        self._state = np.array([float(self._t) * 0.1, float(action), 0.2], dtype=np.float32)
        done = self._t >= 8
        return self._state.copy(), rew, done, False, {"action_mask": self._mask()}


class _BoundedScaleContinuousEnv(gym.Env):
    """Continuous Box env with large shifted action bounds [lo, hi] and optional cost signal."""

    def __init__(self, low: float = 0.0, high: float = 1000.0, act_dim: int = 2, with_cost: bool = False) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low, high, shape=(act_dim,), dtype=np.float32)
        self._low = float(low)
        self._high = float(high)
        self._with_cost = with_cost
        self._t = 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return self._rng.normal(0.0, 1.0, size=(4,)).astype(np.float32), {}

    def step(self, action: np.ndarray):
        self._t += 1
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        mid = 0.5 * (self._low + self._high)
        span = max(0.5 * (self._high - self._low), 1e-6)
        dist = float(np.mean(((a - mid) / span) ** 2))
        rew = 1.0 - dist
        info: dict[str, Any] = {}
        if self._with_cost:
            info["cost"] = 1.5
        trunc = self._t >= 10
        obs = self._rng.normal(0.0, 1.0, size=(4,)).astype(np.float32)
        return obs, rew, False, trunc, info


class _UnboundedBoxEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        del action
        return np.zeros(3, dtype=np.float32), 0.0, False, True, {}


class _LargeMultiDiscreteMaskedEnv(gym.Env):
    """MultiDiscrete([10, 10, 10]) -> joint=1000 (>256) factorized with 30-dim mask."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([10, 10, 10])
        self._t = 0
        self._mask = np.zeros(30, dtype=np.bool_)
        self._mask[[0, 1, 10, 11, 20, 21]] = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._t = 0
        return np.zeros(4, dtype=np.float32), {"action_mask": self._mask.copy()}

    def step(self, action: np.ndarray):
        self._t += 1
        arr = np.asarray(action, dtype=np.int64).reshape(3)
        # Every selected sub-action must satisfy the mask
        valid = bool(self._mask[arr[0]] and self._mask[10 + arr[1]] and self._mask[20 + arr[2]])
        if not valid:
            raise AssertionError(f"Illegal MultiDiscrete action sampled: {arr}")
        return np.ones(4, dtype=np.float32) * (self._t / 5.0), 1.0, False, self._t >= 5, {"action_mask": self._mask.copy()}


class _CHWImageEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(0, 255, shape=(3, 8, 8), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._t = 0
        return np.zeros((3, 8, 8), dtype=np.uint8), {}

    def step(self, action: int):
        self._t += 1
        return np.full((3, 8, 8), int(action) + 1, dtype=np.uint8), 1.0, False, self._t >= 4, {"cost": 0.5}


class _DictObsEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "vec": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                "img": spaces.Box(0, 255, shape=(1, 8, 8), dtype=np.uint8),
            }
        )
        self.action_space = spaces.Discrete(2)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._t = 0
        return {"vec": np.zeros(3, dtype=np.float32), "img": np.zeros((1, 8, 8), dtype=np.uint8)}, {}

    def step(self, action: int):
        del action
        self._t += 1
        return (
            {"vec": np.ones(3, dtype=np.float32), "img": np.ones((1, 8, 8), dtype=np.uint8)},
            1.0,
            False,
            self._t >= 4,
            {"cost": 0.25},
        )


class _MutableRmsEnv(gym.Env):
    """Custom env mutating `.obs_rms` during eval without exposing `.training`."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.obs_rms = type("RMS", (), {"mean": np.zeros(2), "var": np.ones(2), "count": 1.0})()
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._t = 0
        return np.array([0.1, 0.2], dtype=np.float32), {}

    def step(self, action: int):
        del action
        self._t += 1
        self.obs_rms.mean = self.obs_rms.mean + 0.05
        self.obs_rms.count += 1.0
        return np.array([0.1 * self._t, 0.2], dtype=np.float32), 1.0, False, self._t >= 4, {}


class _HighDimShiftedLeakEnv(gym.Env):
    """80-dim (>64) env with a shifted-column lookahead clone on columns 0 and 1."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-50.0, 50.0, shape=(80,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._prices = np.zeros(64, dtype=np.float32)
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        steps = self._rng.normal(0.0, 2.0, size=63)
        self._prices = np.concatenate([[10.0], 10.0 + np.cumsum(steps)]).astype(np.float32)
        self._t = 0
        obs = self._rng.normal(0.0, 0.05, size=(80,)).astype(np.float32)
        obs[0] = self._prices[0]
        obs[1] = self._prices[1]
        return obs, {}

    def step(self, action: np.ndarray):
        a = float(np.asarray(action).reshape(-1)[0])
        ret = float(self._prices[self._t + 1] - self._prices[self._t])
        self._t += 1
        obs = self._rng.normal(0.0, 0.05, size=(80,)).astype(np.float32)
        obs[0] = self._prices[self._t]
        obs[1] = self._prices[self._t + 1]
        return obs, a * ret, False, self._t >= 30, {}


class _SignInnovationLeakEnv(gym.Env):
    """Nonlinear sign lookahead: obs_t[1] = sign(price_{t+1} - price_t), r_t = a_t * (price_{t+1} - price_t)."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-100.0, 100.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(0)
        self._price = 10.0
        self._next_ret = 1.0
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._price = 10.0
        self._next_ret = float(self._rng.normal(0.0, 1.5))
        return np.array([self._price, float(np.sign(self._next_ret))], dtype=np.float32), {}

    def step(self, action: np.ndarray):
        a = float(np.asarray(action).reshape(-1)[0])
        rew = a * self._next_ret
        self._price += self._next_ret
        self._next_ret = float(self._rng.normal(0.0, 1.5))
        self._t += 1
        return np.array([self._price, float(np.sign(self._next_ret))], dtype=np.float32), rew, False, self._t >= 35, {}


class TestAuditFixesV111(unittest.TestCase):
    def test_version_111(self) -> None:
        self.assertEqual(__version__, "1.1.1")

    def test_cql_iql_c51_provenance_and_continuous_cql_is(self) -> None:
        rng = np.random.default_rng(7)
        n = 64
        obs = rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)
        next_obs = obs + 0.1
        acts_c = rng.uniform(0.0, 10.0, size=(n, 2)).astype(np.float32)
        rews = rng.normal(1.0, 0.2, size=(n,)).astype(np.float32)
        terms = np.zeros(n, dtype=np.float32)
        terms[15::16] = 1.0

        ds_cont = OfflineTransitionDataset.from_arrays(
            obs, acts_c, rews, next_obs, terms, is_discrete=False, action_low=[0.0, 0.0], action_high=[10.0, 10.0]
        )
        res_cql_c = train_cql(ds_cont, CQLConfig(num_grad_steps=6, batch_size=16, num_random_actions=4, eval_episodes=2))
        self.assertTrue(np.isfinite(res_cql_c.train_diagnostics["mean_cql_ood_gap"]))
        self.assertIn("clipped_double_q", res_cql_c.optimizations_used)

        res_iql_c = train_iql_offline(ds_cont, IQLConfig(num_grad_steps=6, batch_size=16, eval_episodes=2))
        self.assertIn("clipped_double_q", res_iql_c.optimizations_used)

        ds_disc = OfflineTransitionDataset.from_arrays(
            obs, rng.integers(0, 3, size=(n,)), rews, next_obs, terms, is_discrete=True, action_dim=3
        )
        res_cql_d = train_cql(ds_disc, CQLConfig(num_grad_steps=6, batch_size=16, eval_episodes=2))
        self.assertNotIn("clipped_double_q", res_cql_d.optimizations_used)

        res_iql_d = train_iql_offline(ds_disc, IQLConfig(num_grad_steps=6, batch_size=16, eval_episodes=2))
        self.assertNotIn("clipped_double_q", res_iql_d.optimizations_used)

        res_c51 = train_c51(_MaskedTrapDiscreteEnv, C51Config(total_timesteps=40, learning_starts=10, batch_size=8, eval_episodes=2))
        self.assertNotIn("double_q_target", res_c51.optimizations_used)

    def test_sac_lagrangian_pid_init_and_anti_windup(self) -> None:
        res = train_sac_lagrangian(
            lambda: _BoundedScaleContinuousEnv(0.0, 10.0, act_dim=2, with_cost=True),
            SACLagrangianConfig(
                total_timesteps=60,
                learning_starts=10,
                batch_size=8,
                cost_limit=0.5,
                init_lambda=0.5,
                pid_kp=0.2,
                pid_ki=0.01,
                max_lambda=5.0,
                normalize_obs=True,
                eval_episodes=2,
            ),
        )
        # With pid_i initialized to init_lambda=0.5 (not 0.5/0.01=50), lambda stays bounded cleanly <= max_lambda
        self.assertLessEqual(res.train_diagnostics["final_lagrange_lambda"], 5.0 + 1e-6)
        self.assertGreaterEqual(res.train_diagnostics["final_lagrange_lambda"], 0.0)

    def test_vectorized_env_preserves_terminal_final_info(self) -> None:
        for dict_fmt in (False, True):
            vec = VectorizedEnvAdapter(_TerminalInfoVecEnv(dict_of_arrays=dict_fmt))
            vec.reset_batch(seed=0)
            sb1 = vec.step_batch(np.array([0, 1]))
            self.assertFalse(bool(sb1.dones[0]))
            sb2 = vec.step_batch(np.array([0, 1]))
            self.assertTrue(bool(sb2.dones[0]))
            self.assertAlmostEqual(float(sb2.infos[0]["cost"]), 7.5, places=5)

    def test_dqn_family_next_action_masks_and_replay_roundtrip(self) -> None:
        rb = ReplayBuffer(16, 3, 3, discrete_action=True, track_next_masks=True)
        obs = np.zeros((2, 3), dtype=np.float32)
        nmask = np.array([[True, True, False], [True, False, False]], dtype=np.bool_)
        rb.add_batch(obs, np.array([0, 1]), np.array([1.0, 0.5]), obs, np.array([0.0, 0.0]), next_action_masks=nmask)
        st = rb.state_dict()
        rb2 = ReplayBuffer(16, 3, 3, discrete_action=True, track_next_masks=True)
        rb2.load_state_dict(st)
        _, _, _, _, _, b_nm = rb2.sample_with_next_masks(4, rng=np.random.default_rng(0))
        assert b_nm is not None
        self.assertFalse(bool(b_nm[:, 2].any().item()))

        for fn, cfg in (
            (train_dqn, DQNConfig(total_timesteps=36, learning_starts=8, batch_size=8, eval_episodes=2, normalize_obs=True)),
            (train_double_dqn, DoubleDQNConfig(total_timesteps=36, learning_starts=8, batch_size=8, eval_episodes=2)),
            (train_dueling_dqn, DuelingDQNConfig(total_timesteps=36, learning_starts=8, batch_size=8, eval_episodes=2)),
            (train_c51, C51Config(total_timesteps=36, learning_starts=8, batch_size=8, eval_episodes=2)),
        ):
            res = fn(_MaskedTrapDiscreteEnv, cfg)
            self.assertTrue(np.isfinite(res.train_diagnostics["mean_q_value"]))
            self.assertIn("action_masking", res.optimizations_used)

    def test_trpo_multidim_continuous_kl_respects_budget(self) -> None:
        res = train_trpo(
            lambda: _BoundedScaleContinuousEnv(-2.0, 2.0, act_dim=4),
            TRPOConfig(total_timesteps=96, n_steps=32, max_kl=0.01, eval_episodes=2),
        )
        self.assertLessEqual(res.train_diagnostics["mean_kl"], 0.015 + 1e-6)

    def test_seq_rl_temperature_alignment_long_horizon_and_eos_mask(self) -> None:
        pol = AutoregressiveSeqPolicy(vocab_size=8, embed_dim=16, hidden_dim=32, max_seq_len=160)
        prompts = torch.randint(0, 8, (4, 6), dtype=torch.long)
        resps, gen_lps = pol.generate(prompts, max_new_tokens=5, temperature=0.7)
        eval_lps, _ = pol.sequence_logprobs_and_entropy(prompts, resps, temperature=0.7)
        self.assertTrue(torch.allclose(gen_lps, eval_lps, atol=1e-5))

        eos_resps = torch.tensor([[3, 2, 0, 5, 0], [1, 4, 2, 3, 5]], dtype=torch.long)
        mask = AutoregressiveSeqPolicy.eos_valid_mask(eos_resps, eos_token_id=0)
        self.assertEqual(mask[0].tolist(), [1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(mask[1].tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])

        long_task = SequenceTaskEnv(vocab_size=6, prompt_len=70, max_response_len=70, eos_token_id=0)
        for fn, cfg in (
            (train_grpo, GRPOConfig(total_updates=1, prompts_per_batch=1, group_size=2, update_epochs=1, temperature=0.7, eval_prompts=2)),
            (train_dapo, DAPOConfig(total_updates=1, prompts_per_batch=1, group_size=2, update_epochs=1, temperature=0.7, eval_prompts=2)),
            (train_gspo, GSPOConfig(total_updates=1, prompts_per_batch=1, group_size=2, update_epochs=1, temperature=0.7, eval_prompts=2)),
            (train_arpo, ARPOConfig(total_updates=1, prompts_per_batch=1, group_size=2, update_epochs=1, temperature=0.7, eval_prompts=2)),
        ):
            res = fn(long_task, cfg)
            self.assertIsNotNone(res.eval_stats)

    def test_discorl_continuous_bounds_and_truncation_reset(self) -> None:
        lo = np.array([10.0, 10.0], dtype=np.float32)
        hi = np.array([20.0, 20.0], dtype=np.float32)
        agent = DiscoAgentNet(4, 2, is_discrete=False, action_low=lo, action_high=hi)
        pol_out, _ = agent(torch.randn(16, 4) * 50.0)
        self.assertTrue(bool(torch.all(pol_out >= 10.0 - 1e-5) and torch.all(pol_out <= 20.0 + 1e-5)))

        rule = DiscoMetaUpdateNet(latent_dim=4, meta_hidden=16)
        torch.nn.init.normal_(rule.adv_head.weight, std=0.5)
        rew1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rew2 = torch.tensor([1.0, 2.0, 99.0, -88.0])
        nt = torch.ones(4)
        dones = torch.tensor([0.0, 1.0, 0.0, 0.0])  # truncation boundary after step 1
        lp = torch.zeros(4)
        yc = torch.zeros(4, 4)
        yn = torch.zeros(4, 4)
        adv1, _ = rule(rew1, nt, lp, yc, yn, dones=dones)
        adv2, _ = rule(rew2, nt, lp, yc, yn, dones=dones)
        self.assertTrue(torch.allclose(adv1[:2], adv2[:2], atol=1e-6))

        res = train_discorl(
            lambda: _BoundedScaleContinuousEnv(10.0, 20.0, act_dim=2),
            DiscoRLConfig(total_timesteps=40, n_steps=16, eval_episodes=2),
        )
        self.assertIsNotNone(res.eval_stats)

    def test_set_global_seed_deterministic_torch(self) -> None:
        prev = torch.are_deterministic_algorithms_enabled()
        try:
            set_global_seed(123, deterministic_torch=True)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(prev)

    def test_external_adapter_chw_dict_cost_and_unfrozen_rms(self) -> None:
        class _CHWPolicy(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                assert x.ndim == 4 and x.shape[1:] == (3, 8, 8)
                return torch.zeros((x.shape[0], 2), dtype=torch.float32)

        res_img = evaluate_external_policy(_CHWImageEnv, _CHWPolicy(), eval_episodes=2)
        self.assertAlmostEqual(float(res_img[0].train_diagnostics["mean_episode_cost"]), 2.0, places=4)

        class _DictPolicy(torch.nn.Module):
            def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
                assert "vec" in x and "img" in x and x["img"].ndim == 4
                return torch.zeros((x["vec"].shape[0], 2), dtype=torch.float32)

        res_dict = evaluate_external_policy(_DictObsEnv, _DictPolicy(), eval_episodes=2)
        self.assertAlmostEqual(float(res_dict[0].train_diagnostics["mean_episode_cost"]), 1.0, places=4)

        res_mut = evaluate_external_policy(_MutableRmsEnv, lambda _o: 0, eval_episodes=2)
        self.assertFalse(res_mut[0].obs_rms_frozen_on_eval)

    def test_dict_offline_dataset_preflight_subgroups_and_exact_wis(self) -> None:
        rng = np.random.default_rng(11)
        n = 40
        obs = {
            "vec": rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32),
            "img": rng.integers(0, 255, size=(n, 1, 4, 4), dtype=np.uint8),
        }
        next_obs = {
            "vec": rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32),
            "img": rng.integers(0, 255, size=(n, 1, 4, 4), dtype=np.uint8),
        }
        acts = rng.integers(0, 2, size=(n,))
        rews = np.ones(n, dtype=np.float32)
        terms = np.zeros(n, dtype=np.float32)
        terms[9::10] = 1.0
        groups = np.array(["A"] * 20 + ["B"] * 20)
        ds = OfflineTransitionDataset.from_arrays(obs, acts, rews, next_obs, terms, is_discrete=True, group_labels=groups)

        pf = preflight_env(ds, run_probes=True)
        self.assertTrue(pf.ok, pf.errors)
        ope = evaluate_offline_ope(ds, lambda _o, _i: 0, fqe_steps=5)
        self.assertIn("A", ope["subgroups"])
        self.assertIn("B", ope["subgroups"])

        # Exact trajectory WIS vs geometric mean: 2 episodes of length 3; ep0 matches all 3 actions, ep1 matches 1/3
        ds_wis = OfflineTransitionDataset.from_arrays(
            obs=np.zeros((6, 2), dtype=np.float32),
            actions=np.array([1, 1, 1, 1, 0, 0]),
            rewards=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            next_obs=np.zeros((6, 2), dtype=np.float32),
            terminateds=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            is_discrete=True,
            action_dim=2,
            behavior_probs=np.full(6, 0.5, dtype=np.float32),
        )
        wis = evaluate_offline_wis(ds_wis, lambda _o, _i: 1, gamma=1.0, soft_eps=0.1)
        # Trajectory weight ratio: ep0=(0.9/0.5)^3=5.832 vs ep1=(0.9/0.5)*(0.1/0.5)^2=0.072 -> wis_ret ~ 9.88
        self.assertGreater(wis["wis_return"], 9.5)

    def test_dual_demo_replay_buffer_serialization_and_raw_obs_rms(self) -> None:
        buf = DualDemoReplayBuffer(16, 3, 2, discrete_action=False)
        buf.add_demo(np.array([1.0, 2.0, 3.0]), np.array([0.5, -0.5]), 1.0, np.array([2.0, 3.0, 4.0]), False)
        buf.add_online(np.array([4.0, 5.0, 6.0]), np.array([-0.2, 0.2]), 0.5, np.array([5.0, 6.0, 7.0]), True)
        state = buf.state_dict()
        buf2 = DualDemoReplayBuffer(16, 3, 2, discrete_action=False)
        buf2.load_state_dict(state)
        self.assertEqual(buf2.demo_buffer.size, 1)
        self.assertEqual(buf2.online_buffer.size, 1)

        rms = DictRunningMeanStd(infer_obs_spec(spaces.Box(-10, 10, shape=(3,), dtype=np.float32)))
        rms.update(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32))
        b_obs, _, _, _, _ = buf2.sample_symmetric(2, obs_rms=rms, rng=np.random.default_rng(0))
        self.assertTrue(bool(torch.all(torch.isfinite(b_obs))))

    def test_large_multidiscrete_masks_unbounded_guard_and_critic_scaling(self) -> None:
        vec = make_vec_env(_LargeMultiDiscreteMaskedEnv, num_envs=2)
        _, m0 = vec.reset_batch(seed=0)
        assert m0 is not None
        self.assertEqual(m0.shape, (2, 30))
        vec.close()

        res_ppo = train_ppo(_LargeMultiDiscreteMaskedEnv, PPOConfig(total_timesteps=40, n_steps=10, update_epochs=2, eval_episodes=2))
        self.assertIn("action_masking", res_ppo.optimizations_used)

        for fn, cfg in (
            (train_sac, SACConfig(total_timesteps=20)),
            (train_td3, TD3Config(total_timesteps=20)),
            (train_ddpg, DDPGConfig(total_timesteps=20)),
            (train_sac_lagrangian, SACLagrangianConfig(total_timesteps=20)),
            (train_hil_serl, HILSERLConfig(total_timesteps=20)),
        ):
            with self.assertRaises(ValueError):
                fn(_UnboundedBoxEnv, cfg)

        lo = np.array([0.0, 0.0], dtype=np.float32)
        hi = np.array([1000.0, 1000.0], dtype=np.float32)
        tq = TwinQCritic(4, 2, action_low=lo, action_high=hi)
        sq = SingleQCritic(4, 2, action_low=lo, action_high=hi)
        a_mid = torch.tensor([[500.0, 500.0]])
        self.assertTrue(torch.allclose(tq._norm_action(a_mid), torch.zeros(1, 2), atol=1e-6))
        self.assertTrue(torch.allclose(sq._norm_action(a_mid), torch.zeros(1, 2), atol=1e-6))

        res_sac = train_sac(
            lambda: _BoundedScaleContinuousEnv(0.0, 1000.0, act_dim=2),
            SACConfig(total_timesteps=40, learning_starts=10, batch_size=8, normalize_obs=True, eval_episodes=2),
        )
        self.assertTrue(np.isfinite(res_sac.train_diagnostics["mean_q_value"]))

    def test_probes_high_dim_sign_multi_kpi_aliases_cli_and_eval_episodes(self) -> None:
        # Probe B on obs_dim=80 (>64)
        pr_hd = run_mdp_probes(_HighDimShiftedLeakEnv, horizon_steps=140, n_permutations=15, seed=42)
        self.assertTrue(pr_hd.future_leakage_suspected, pr_hd.leakage_signals)

        # Probe C nonlinear sign lookahead
        pr_sign = run_mdp_probes(_SignInnovationLeakEnv, horizon_steps=150, n_permutations=15, seed=42)
        self.assertTrue(pr_sign.future_leakage_suspected, pr_sign.leakage_signals)

        # Multi-KPI degradation caught on second KPI (not just kpi_names[0])
        rand_base = EpisodeEvalStats(
            returns=[1.0],
            lengths=[10],
            mean_return=1.0,
            std_return=0.0,
            mean_length=10.0,
            task_metrics_mean={"kpi_ok": 0.5, "kpi_hacked": 0.9},
        )
        runs = [
            TrainResult(
                algo_id="ppo",
                seed=s,
                total_timesteps=100,
                elapsed_seconds=0.1,
                policy_fn=lambda _o, _i: 0,
                model=None,
                config={},
                learning_curve=[{"step": 100.0, "episode_return": 5.0}],
                train_diagnostics={"final_entropy": 0.5},
                eval_stats=EpisodeEvalStats(
                    returns=[5.0],
                    lengths=[10],
                    mean_return=5.0,
                    std_return=0.0,
                    mean_length=10.0,
                    task_metrics_mean={"kpi_ok": 0.8, "kpi_hacked": 0.2},
                ),
            )
            for s in range(3)
        ]
        audit = audit_drl_rigor(runs, random_baseline=rand_base, min_seeds=3)
        self.assertIn("suspected_reward_hacking_kpi_divergence", audit.red_flags)

        # Extractor aliases "impala" and "set_attention"
        img_spec = infer_obs_spec(spaces.Box(0, 255, shape=(3, 10, 10), dtype=np.uint8))
        ext_imp, _ = build_feature_extractor(img_spec, extractor_type="impala")
        self.assertIsInstance(ext_imp, ImpalaResNetExtractor)

        ent_spec = infer_obs_spec(spaces.Box(-1, 1, shape=(5, 4), dtype=np.float32), extractor_type="set_attention")
        ext_att, _ = build_feature_extractor(ent_spec, extractor_type="set_attention")
        self.assertIsInstance(ext_att, SetAttentionExtractor)

        # CLI intake supports finance_portfolio_risk, clinical_offline_safety, cyber_attack_defense
        with tempfile.TemporaryDirectory() as tmp:
            for arch in ("finance_portfolio_risk", "clinical_offline_safety", "cyber_attack_defense"):
                out_p = Path(tmp) / f"{arch}.json"
                rc = cli_main(["intake", "--archetype", arch, "--out", str(out_p), "--json"])
                self.assertEqual(rc, 0)
                self.assertTrue(out_p.exists())

        # evaluate_drl_suite forwards eval_episodes to default algo config
        rep = evaluate_drl_suite(
            _MaskedTrapDiscreteEnv,
            algos=["q_learning"],
            seeds=(0,),
            min_seeds=1,
            eval_episodes=3,
            run_preflight=False,
            run_audit=False,
        )
        self.assertEqual(int(rep.algo_summaries[0].hyperparameters["eval_episodes"]), 3)


if __name__ == "__main__":
    unittest.main()
