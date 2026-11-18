"""Comprehensive Phase A Engine Tests: Multimodal Encoders (A1), True VectorEnv (A2), Checkpoints & Resume (A3)."""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from wannado.drl import (
    A2CConfig,
    C51Config,
    CQLConfig,
    DDPGConfig,
    DictRunningMeanStd,
    DoubleDQNConfig,
    DQNConfig,
    DuelingDQNConfig,
    HILSERLConfig,
    ImpalaResNetExtractor,
    IQLConfig,
    NatureCNNExtractor,
    OfflineTransitionDataset,
    PolicyBundle,
    PPOConfig,
    PPOGRUConfig,
    ReinforceConfig,
    SACConfig,
    SetAttentionExtractor,
    TD3Config,
    TRPOConfig,
    evaluate_drl_suite,
    evaluate_offline_fqe,
    evaluate_offline_wis,
    evaluate_policy_episodes,
    infer_obs_spec,
    load_policy,
    make_vec_env,
    train_a2c,
    train_c51,
    train_cql,
    train_ddpg,
    train_double_dqn,
    train_dqn,
    train_dueling_dqn,
    train_hil_serl,
    train_iql_offline,
    train_ppo,
    train_ppo_gru,
    train_reinforce,
    train_sac,
    train_td3,
    train_trpo,
)
from wannado.drl.core.buffers import ReplayBuffer, RolloutBuffer


class _PixelNavEnv(gym.Env[np.ndarray, int]):
    """Lightweight HWC uint8 pixel environment rewarding rightward move toward target column."""

    def __init__(self, height: int = 84, width: int = 84, channels: int = 3, max_steps: int = 12) -> None:
        super().__init__()
        self.h = height
        self.w = width
        self.c = channels
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, channels), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.pos = 1
        self.step_idx = 0

    def _render_obs(self) -> np.ndarray:
        img = np.zeros((self.h, self.w, self.c), dtype=np.uint8)
        col = min(self.w - 8, max(0, int((self.pos / 5.0) * (self.w - 16))))
        img[self.h // 2 - 4 : self.h // 2 + 4, col : col + 8, :] = 220
        return img

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.pos = 1
        self.step_idx = 0
        return self._render_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_idx += 1
        if int(action) == 1:
            self.pos = min(5, self.pos + 1)
            rew = 1.0
        else:
            self.pos = max(0, self.pos - 1)
            rew = -0.2
        term = self.pos >= 5
        trunc = (self.step_idx >= self.max_steps) and not term
        return self._render_obs(), float(rew), term, trunc, {}


class _MinAtarLikeGridEnv(gym.Env[np.ndarray, int]):
    """Small (10, 10, 4) float32 [0..1] visual grid verifying adaptive NatureCNN & Impala."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10, 10, 4), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0
        obs = np.zeros((10, 10, 4), dtype=np.float32)
        obs[self.t % 10, 2, 0] = 1.0
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.t += 1
        obs = np.zeros((10, 10, 4), dtype=np.float32)
        obs[self.t % 10, int(action), min(3, int(action))] = 1.0
        rew = 1.0 if int(action) == 2 else 0.1
        return obs, float(rew), False, self.t >= 8, {}


class _MultimodalEntityEnv(gym.Env[dict[str, Any], np.ndarray]):
    """Dict(image, vec, entities, entities_mask) continuous environment with occasional all-masked rows."""

    def __init__(self, continuous: bool = True, all_masked_even: bool = True) -> None:
        super().__init__()
        self.continuous = continuous
        self.all_masked_even = all_masked_even
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
                "vec": spaces.Box(low=-5.0, high=5.0, shape=(4,), dtype=np.float32),
                "entities": spaces.Box(low=-5.0, high=5.0, shape=(5, 4), dtype=np.float32),
                "entities_mask": spaces.MultiBinary(5),
            }
        )
        if continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)
        self.t = 0
        self.state = np.zeros(4, dtype=np.float32)

    def _obs(self) -> dict[str, Any]:
        img = np.full((32, 32, 3), int((self.t * 30) % 255), dtype=np.uint8)
        ents = np.tile(self.state[None, :], (5, 1)).astype(np.float32)
        if self.all_masked_even and self.t % 2 == 0:
            mask = np.zeros((5,), dtype=np.int8)
        else:
            mask = np.array([1, 1, 1, 0, 0], dtype=np.int8)
        return {"image": img, "vec": self.state.copy(), "entities": ents, "entities_mask": mask}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0
        self.state = self.np_random.uniform(-0.5, 0.5, size=(4,)).astype(np.float32)
        return self._obs(), {}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.t += 1
        if self.continuous:
            a = np.asarray(action, dtype=np.float32).reshape(2)
            self.state[:2] = 0.8 * self.state[:2] + 0.2 * a
            rew = float(-np.sum(np.square(self.state[:2] - 0.5)))
        else:
            act_i = int(action)
            self.state[0] += (act_i - 1) * 0.2
            rew = 1.0 if act_i == 2 else 0.0
        return self._obs(), rew, False, self.t >= 6, {}


class _BoundaryCounterEnv(gym.Env[np.ndarray, int]):
    """Deterministic horizon-5 counter verifying AutoresetMode SAME_STEP vs NEXT_STEP."""

    def __init__(self, horizon: int = 5) -> None:
        super().__init__()
        self.horizon = horizon
        self.observation_space = spaces.Box(low=0.0, high=20.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.count = 0
        return np.array([0.0, 1.0], dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.count += 1
        done = self.count >= self.horizon
        rew = 10.0 if done else 1.0
        return np.array([float(self.count), float(action)], dtype=np.float32), rew, done, False, {}


class TestPhaseAEngine(unittest.TestCase):
    def test_a1_pixel_uint8_storage_and_adaptive_cnn_impala(self) -> None:
        pix_env = _PixelNavEnv()
        spec = infer_obs_spec(pix_env.observation_space)
        self.assertEqual(spec.kind, "image")
        self.assertEqual(spec.shape, (3, 84, 84))
        self.assertEqual(spec.dtype, "uint8")
        self.assertTrue(spec.normalize_pixel)

        # ReplayBuffer & RolloutBuffer keep uint8 storage (zero 4x float32 bloat)
        rb = ReplayBuffer(32, spec, 2, discrete_action=True)
        self.assertEqual(rb.obs.dtype, np.uint8)
        self.assertEqual(rb.obs.shape, (32, 3, 84, 84))
        ro = RolloutBuffer(8, spec, (), np.dtype(np.int64), num_envs=2, batched=True)
        self.assertEqual(ro.obs.dtype, np.uint8)
        self.assertEqual(ro.obs.shape, (8, 2, 3, 84, 84))

        # Adaptive NatureCNN & Impala handle both 84x84 and 10x10 MinAtar-size grids
        small_spec = infer_obs_spec(_MinAtarLikeGridEnv().observation_space)
        self.assertEqual(small_spec.kind, "image")
        self.assertEqual(small_spec.shape, (4, 10, 10))
        self.assertFalse(small_spec.normalize_pixel)
        cnn_small = NatureCNNExtractor(small_spec.shape, features_dim=32)  # type: ignore[arg-type]
        imp_small = ImpalaResNetExtractor(small_spec.shape, features_dim=32)  # type: ignore[arg-type]
        x_small = torch.rand(4, 4, 10, 10)
        self.assertEqual(cnn_small(x_small).shape, (4, 32))
        self.assertEqual(imp_small(x_small).shape, (4, 32))

        with tempfile.TemporaryDirectory() as td:
            res_ppo = train_ppo(
                _PixelNavEnv,
                config=PPOConfig(
                    total_timesteps=256,
                    n_steps=64,
                    minibatch_size=32,
                    update_epochs=2,
                    num_envs=2,
                    extractor_type="impala",
                    normalize_obs=True,
                    eval_episodes=2,
                    checkpoint_dir=td,
                    eval_freq=128,
                    seed=0,
                ),
            )
            self.assertIsNotNone(res_ppo.best_model_path)
            loaded = load_policy(res_ppo.best_model_path)
            raw_obs, _ = pix_env.reset(seed=1)
            a_orig = res_ppo.policy_fn(raw_obs, {})
            a_load = loaded(raw_obs, {})
            self.assertIn(a_orig, (0, 1))
            self.assertIn(a_load, (0, 1))

        # DQN on 10x10 MinAtar-like float grid
        res_dqn = train_double_dqn(
            _MinAtarLikeGridEnv,
            config=DoubleDQNConfig(
                total_timesteps=200,
                learning_starts=40,
                batch_size=16,
                num_envs=2,
                extractor_type="nature_cnn",
                eval_episodes=2,
                seed=0,
            ),
        )
        self.assertIsNotNone(res_dqn.eval_stats)
        self.assertEqual(res_dqn.train_diagnostics["num_envs"], 2.0)

    def test_a1_multimodal_dict_set_attention_all_masked_and_offline(self) -> None:
        env = _MultimodalEntityEnv(continuous=True)
        spec = infer_obs_spec(env.observation_space)
        self.assertEqual(spec.kind, "dict")
        self.assertEqual(spec.leaves["entities"].kind, "entities")
        self.assertEqual(spec.leaves["entities"].mask_key, "entities_mask")

        # SetAttention all-masked row must return finite non-NaN output
        sa = SetAttentionExtractor(spec.leaves["entities"].shape[-1], features_dim=16)
        ents = torch.randn(3, 5, 4)
        mask = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        out = sa(ents, mask)
        self.assertEqual(out.shape, (3, 16))
        self.assertTrue(torch.all(torch.isfinite(out)).item())

        # Permutation invariance on unmasked elements
        perm = torch.tensor([2, 0, 1, 3, 4])
        out_perm = sa(ents[:, perm, :], mask[:, perm])
        self.assertTrue(torch.allclose(out, out_perm, atol=1e-5))

        # DictRunningMeanStd skips uint8 image and mask keys
        rms = DictRunningMeanStd(spec)
        o0, _ = env.reset(seed=0)
        from wannado.drl.core.obs_spec import format_obs_single

        fmt0 = format_obs_single(o0, spec)
        rms.update(fmt0)
        norm0 = rms.normalize(fmt0)
        self.assertEqual(norm0["image"].dtype, np.uint8)
        self.assertTrue(np.array_equal(norm0["entities_mask"], fmt0["entities_mask"]))

        # Train continuous SAC + offline CQL + IQL on Dict+Entity multimodal env
        res_sac = train_sac(
            lambda: _MultimodalEntityEnv(continuous=True),
            config=SACConfig(
                total_timesteps=160,
                learning_starts=40,
                batch_size=16,
                num_envs=2,
                normalize_obs=True,
                eval_episodes=2,
                seed=0,
            ),
        )
        raw_d, _ = env.reset(seed=7)
        act_pred = res_sac.policy_bundle(raw_d, {})
        self.assertEqual(np.asarray(act_pred).shape, (2,))
        ev_sac = evaluate_policy_episodes(env, res_sac.policy_bundle, num_episodes=2, seed=11)
        self.assertEqual(len(ev_sac.returns), 2)

        # Offline Dict dataset + CQL + IQL + WIS + FQE
        disc_ds = OfflineTransitionDataset.collect_from_env(
            lambda: _MultimodalEntityEnv(continuous=False), num_steps=80, seed=0
        )
        res_cql = train_cql(
            disc_ds,
            config=CQLConfig(num_grad_steps=30, batch_size=16, normalize_obs=True, eval_episodes=2, seed=0),
            eval_env_or_factory=lambda: _MultimodalEntityEnv(continuous=False),
        )
        self.assertIsNotNone(res_cql.eval_stats)
        wis = evaluate_offline_wis(disc_ds, res_cql.policy_bundle)
        fqe = evaluate_offline_fqe(disc_ds, res_cql.policy_bundle, num_steps=10)
        self.assertTrue(np.isfinite(wis["wis_return"]))
        self.assertTrue(np.isfinite(fqe["fqe_mean_q"]))

        cont_ds = OfflineTransitionDataset.collect_from_env(
            lambda: _MultimodalEntityEnv(continuous=True), num_steps=80, seed=1
        )
        res_iql = train_iql_offline(
            cont_ds,
            config=IQLConfig(num_grad_steps=30, batch_size=16, normalize_obs=True, eval_episodes=2, seed=1),
        )
        self.assertIn("ope_wis_return", res_iql.train_diagnostics)

    def test_a2_vectorenv_same_step_vs_next_step_autoreset_and_algorithms(self) -> None:
        # 1. SAME_STEP boundary check
        vec_same = make_vec_env(_BoundaryCounterEnv, num_envs=3)
        self.assertIn("same", vec_same.autoreset_mode.lower())
        obs, _ = vec_same.reset_batch(seed=0)
        self.assertEqual(obs.shape, (3, 2))
        completed_all: list[tuple[float, int]] = []
        valid_total = 0
        for step_i in range(7):
            sb = vec_same.step_batch(np.zeros((3,), dtype=np.int64))
            valid_total += int(np.sum(sb.valid_transition))
            completed_all.extend(sb.completed_episodes)
            if step_i == 4:
                self.assertEqual(len(sb.completed_episodes), 3)
                # final_obs holds step 5.0 while obs has already reset to 0.0
                self.assertTrue(np.allclose(sb.final_obs[:, 0], 5.0))
                self.assertTrue(np.allclose(sb.obs[:, 0], 0.0))
        self.assertEqual(valid_total, 21)
        for ret, ep_len in completed_all:
            self.assertEqual(ep_len, 5)
            self.assertAlmostEqual(ret, 14.0, places=5)
        vec_same.close()

        # 2. External SyncVectorEnv under AutoresetMode.NEXT_STEP
        raw_next = gym.vector.SyncVectorEnv(
            [_BoundaryCounterEnv for _ in range(2)],
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        )
        vec_next = make_vec_env(raw_next)
        self.assertIn("next", vec_next.autoreset_mode.lower())
        rb_next = ReplayBuffer(64, vec_next.obs_spec, vec_next.action_dim, discrete_action=True)
        cur_obs, _ = vec_next.reset_batch(seed=0)
        done_returns: list[float] = []
        invalid_steps = 0
        for _ in range(12):
            acts = np.zeros((2,), dtype=np.int64)
            sb = vec_next.step_batch(acts)
            rb_next.add_batch(cur_obs, acts, sb.rewards, sb.final_obs, sb.terminateds, valid_mask=sb.valid_transition)
            invalid_steps += int(np.sum(~sb.valid_transition))
            for ret, ep_len in sb.completed_episodes:
                done_returns.append(ret)
                self.assertEqual(ep_len, 5)
            cur_obs = sb.obs
        raw_next.close()
        # Out of 12 steps across 2 envs: steps 5 & 11 trigger NEXT_STEP reset dummy transitions -> 2*2=4 invalid transitions
        self.assertEqual(invalid_steps, 4)
        self.assertEqual(rb_next.size, 20)
        self.assertEqual(len(done_returns), 4)
        for r in done_returns:
            self.assertAlmostEqual(r, 14.0, places=5)

        # 3. Smoke-test vectorized training across PG / VB / Off-Policy families with num_envs > 1
        res_gru = train_ppo_gru(
            lambda: gym.make("CartPole-v1"),
            config=PPOGRUConfig(total_timesteps=256, n_steps=64, num_envs=2, normalize_reward=True, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_gru.train_diagnostics["num_envs"], 2.0)

        res_a2c = train_a2c(
            lambda: gym.make("CartPole-v1"),
            config=A2CConfig(total_timesteps=256, n_steps=32, num_envs=2, normalize_reward=True, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_a2c.train_diagnostics["num_envs"], 2.0)

        res_c51 = train_c51(
            lambda: gym.make("CartPole-v1"),
            config=C51Config(total_timesteps=200, learning_starts=40, batch_size=16, num_envs=2, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_c51.train_diagnostics["num_envs"], 2.0)

        res_duel = train_dueling_dqn(
            lambda: gym.make("CartPole-v1"),
            config=DuelingDQNConfig(total_timesteps=200, learning_starts=40, batch_size=16, num_envs=2, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_duel.train_diagnostics["num_envs"], 2.0)

        res_dqn = train_dqn(
            lambda: gym.make("CartPole-v1"),
            config=DQNConfig(total_timesteps=200, learning_starts=40, batch_size=16, num_envs=2, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_dqn.train_diagnostics["num_envs"], 2.0)

        res_trpo = train_trpo(
            lambda: gym.make("CartPole-v1"),
            config=TRPOConfig(total_timesteps=256, n_steps=64, num_envs=2, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_trpo.train_diagnostics["num_envs"], 2.0)

        res_reinf = train_reinforce(
            lambda: gym.make("CartPole-v1"),
            config=ReinforceConfig(total_timesteps=200, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_reinf.train_diagnostics["num_envs"], 1.0)

        res_td3 = train_td3(
            lambda: gym.make("Pendulum-v1"),
            config=TD3Config(total_timesteps=220, learning_starts=50, batch_size=16, num_envs=2, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_td3.train_diagnostics["num_envs"], 2.0)

        res_ddpg = train_ddpg(
            lambda: gym.make("Pendulum-v1"),
            config=DDPGConfig(total_timesteps=220, learning_starts=50, batch_size=16, num_envs=2, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_ddpg.train_diagnostics["num_envs"], 2.0)

        res_hil = train_hil_serl(
            lambda: gym.make("Pendulum-v1"),
            config=HILSERLConfig(total_timesteps=180, learning_starts=40, batch_size=16, num_envs=2, eval_episodes=2, seed=0),
        )
        self.assertEqual(res_hil.train_diagnostics["num_envs"], 2.0)

    def test_a3_checkpoint_best_model_keep_k_warning_and_deterministic_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            # 1. Bare non-factory env + eval_freq>0 must emit RuntimeWarning cleanly without mutating train env
            bare = gym.make("CartPole-v1")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res_warn = train_ppo(
                    bare,
                    config=PPOConfig(total_timesteps=128, n_steps=64, minibatch_size=32, eval_freq=64, eval_episodes=2, seed=0),
                )
            self.assertTrue(any("Periodic mid-training evaluation" in str(w.message) for w in caught))
            self.assertIsNotNone(res_warn.eval_stats)
            bare.close()

            # 2. Periodic eval + best_model + keep_last_k=2 + progress logs + standalone load_policy
            ckpt_dir = root / "ppo_ckpt"
            res = train_ppo(
                lambda: gym.make("CartPole-v1"),
                config=PPOConfig(
                    total_timesteps=768,
                    n_steps=64,
                    minibatch_size=32,
                    update_epochs=2,
                    num_envs=2,
                    normalize_obs=True,
                    checkpoint_dir=str(ckpt_dir),
                    eval_freq=256,
                    checkpoint_freq=256,
                    keep_last_k=2,
                    eval_episodes=3,
                    seed=7,
                ),
            )
            self.assertIsNotNone(res.best_model_path)
            self.assertIsNotNone(res.last_checkpoint_path)
            self.assertTrue(Path(res.best_model_path).exists())
            self.assertTrue(Path(res.last_checkpoint_path).exists())
            self.assertTrue((ckpt_dir / "progress.jsonl").exists())
            self.assertTrue((ckpt_dir / "progress.csv").exists())
            step_files = sorted(ckpt_dir.glob("step_*.pt"))
            self.assertLessEqual(len(step_files), 2)
            self.assertGreaterEqual(len(res.eval_history), 3)

            lines = [json.loads(line) for line in (ckpt_dir / "progress.jsonl").read_text().splitlines() if line.strip()]
            self.assertTrue(any("eval/mean_return" in r for r in lines))

            # Standalone export + load_policy parity
            exported = res.save_policy(root / "standalone_policy.pt")
            bundle_loaded = load_policy(exported)
            best_loaded = load_policy(res.best_model_path)
            self.assertIsInstance(bundle_loaded, PolicyBundle)
            probe_obs = np.array([0.05, -0.1, 0.02, 0.3], dtype=np.float32)
            self.assertEqual(res.policy_bundle(probe_obs, {}), bundle_loaded(probe_obs, {}))
            self.assertIn(best_loaded(probe_obs, {}), (0, 1))
            logits_t = best_loaded.discrete_logits(probe_obs)
            self.assertEqual(logits_t.shape, (1, 2))

            # 3. Exact deterministic resume on offline CQL (zero env stochasticity)
            ds = OfflineTransitionDataset.collect_from_env(lambda: gym.make("CartPole-v1"), num_steps=120, seed=5)
            uninterrupted = train_cql(
                ds,
                config=CQLConfig(num_grad_steps=60, batch_size=16, eval_episodes=3, seed=42),
                eval_env_or_factory=lambda: gym.make("CartPole-v1"),
            )
            cql_ckpt = root / "cql_ckpt"
            _ = train_cql(
                ds,
                config=CQLConfig(
                    num_grad_steps=30,
                    batch_size=16,
                    checkpoint_dir=str(cql_ckpt),
                    checkpoint_freq=30,
                    eval_episodes=3,
                    seed=42,
                ),
                eval_env_or_factory=lambda: gym.make("CartPole-v1"),
            )
            resumed = train_cql(
                ds,
                config=CQLConfig(
                    num_grad_steps=60,
                    batch_size=16,
                    resume_from=str(cql_ckpt / "last_checkpoint.pt"),
                    eval_episodes=3,
                    seed=42,
                ),
                eval_env_or_factory=lambda: gym.make("CartPole-v1"),
            )
            self.assertAlmostEqual(
                uninterrupted.eval_stats.mean_return,
                resumed.eval_stats.mean_return,
                places=5,
            )
            self.assertTrue(
                torch.allclose(
                    uninterrupted.policy_bundle.discrete_logits(probe_obs),
                    resumed.policy_bundle.discrete_logits(probe_obs),
                    atol=1e-6,
                )
            )

            # 4. Off-policy SAC resume with save_replay_buffer=True (restores log_alpha + replay + weights)
            sac_ckpt = root / "sac_ckpt"
            sac_half = train_sac(
                lambda: gym.make("Pendulum-v1"),
                config=SACConfig(
                    total_timesteps=150,
                    learning_starts=40,
                    batch_size=16,
                    num_envs=2,
                    checkpoint_dir=str(sac_ckpt),
                    save_replay_buffer=True,
                    eval_episodes=2,
                    seed=9,
                ),
            )
            self.assertTrue((sac_ckpt / "last_replay_buffer.pt").exists())
            sac_resumed = train_sac(
                lambda: gym.make("Pendulum-v1"),
                config=SACConfig(
                    total_timesteps=250,
                    learning_starts=40,
                    batch_size=16,
                    num_envs=2,
                    resume_from=str(sac_ckpt / "last_checkpoint.pt"),
                    eval_episodes=2,
                    seed=9,
                ),
            )
            self.assertEqual(sac_resumed.total_timesteps, 250)
            self.assertIsNotNone(sac_half.policy_bundle)

            # 5. End-to-end evaluate_drl_suite with num_envs > 1 + isolated per-seed checkpoints
            suite_ckpt = root / "suite_ckpt"
            rep = evaluate_drl_suite(
                lambda: gym.make("CartPole-v1"),
                algos=["ppo", "double_dqn"],
                seeds=[0, 1],
                min_seeds=2,
                algo_configs={
                    "ppo": PPOConfig(
                        total_timesteps=256,
                        n_steps=64,
                        num_envs=2,
                        checkpoint_dir=str(suite_ckpt),
                        eval_episodes=2,
                    ),
                    "double_dqn": DoubleDQNConfig(
                        total_timesteps=200,
                        learning_starts=40,
                        batch_size=16,
                        num_envs=2,
                        checkpoint_dir=str(suite_ckpt),
                        eval_episodes=2,
                    ),
                },
                task_name="PhaseASuite",
                run_audit=True,
            )
            self.assertEqual(len(rep.algo_summaries), 2)
            self.assertTrue((suite_ckpt / "ppo" / "seed_0" / "last_checkpoint.pt").exists())
            self.assertTrue((suite_ckpt / "double_dqn" / "seed_1" / "last_checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
