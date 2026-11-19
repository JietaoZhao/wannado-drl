"""A2C: Synchronous Advantage Actor-Critic (Mnih et al., ICML 2016).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / B (Bridge from REINFORCE to PPO)
- Parent        : `reinforce` (Williams, 1992)
- Key Diff      : Replaces full-episode Monte Carlo returns with fixed-horizon n-step
                  bootstrapped TD/GAE advantages via a learned Critic V(s) plus entropy bonus.
- Extension Hook [AI-Prompt Hook]:
    1. Replace single-step policy gradient with clipped importance ratio -> `ppo`.
    2. Replace hand-crafted TD/policy objective with meta-learned rule -> `discorl`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import (
    build_actor_critic_for_env,
    finalize_online_run,
    maybe_periodic_hooks,
    maybe_resume_online_run,
    setup_online_run,
)
from wannado.drl.core.buffers import RolloutBuffer
from wannado.drl.core.networks import ContinuousActorCritic
from wannado.drl.core.obs_spec import obs_to_tensor, slice_obs
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import explained_variance, resolve_device, set_global_seed

META = AlgoMeta(
    id="a2c",
    name="Advantage Actor-Critic (A2C)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="B",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="reinforce",
    diff_from_parent="Uses n-step bootstrapped GAE advantages from a Critic V(s) so updates happen mid-episode with much lower variance.",
    summary_plain="Fast, lightweight synchronous Actor-Critic updating every n steps without waiting for episodes to finish.",
    when_to_use="Fast environments needing an immediate upgrade over REINFORCE or lightweight on-policy baseline.",
    when_not_to_use="Complex tasks where policy steps overshoot and collapse (prefer `ppo` with clipped ratios).",
    hyperparam_intuition={
        "n_steps": "Rollout horizon before each Actor-Critic update (longer = less bias, more variance).",
        "ent_coef": "Bonus discouraging premature greedy determinism.",
        "vf_coef": "Weight balancing Critic value loss against Actor policy loss.",
    },
    extension_hooks=(
        "Add PPO clipped surrogate objective over multi-epoch minibatches -> `ppo`",
        "Replace hand-written advantage target with learned meta-update network -> `discorl`",
    ),
    module_path="wannado.drl.algos.policy_gradient.a2c_mnih2016",
    citation=AlgoCitation(
        key="mnih2016asynchronous",
        title="Asynchronous Methods for Deep Reinforcement Learning",
        authors="Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu",
        venue="Proceedings of the 33rd International Conference on Machine Learning (ICML), PMLR 48",
        year=2016,
        url="http://proceedings.mlr.press/v48/mnih16.html",
        bibtex=(
            "@inproceedings{mnih2016asynchronous,\n"
            "  title={Asynchronous Methods for Deep Reinforcement Learning},\n"
            "  author={Mnih, Volodymyr and Badia, Adri{\\`a} Puigdom{\\`e}nech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy P. and Harley, Tim and Silver, David and Kavukcuoglu, Koray},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={48},\n"
            "  year={2016},\n"
            "  url={http://proceedings.mlr.press/v48/mnih16.html}\n"
            "}"
        ),
        ieee='V. Mnih et al., "Asynchronous methods for deep reinforcement learning," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 48, 2016.',
        apa="Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. ICML (PMLR 48).",
    ),
)


@dataclass
class A2CConfig:
    total_timesteps: int = 12_000
    n_steps: int = 32
    lr: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dims: tuple[int, ...] = (64, 64)
    normalize_obs: bool = False
    normalize_reward: bool = False
    num_envs: int = 1
    vec_mode: str = "auto"
    extractor_type: str = "auto"
    extractor_kwargs: dict[str, Any] | None = None
    checkpoint_dir: str | None = None
    checkpoint_freq: int = 0
    eval_freq: int = 0
    save_best_model: bool = True
    keep_last_k: int = 3
    resume_from: str | None = None
    tb_log_dir: str | None = None
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_a2c(
    env_or_factory: Any,
    config: A2CConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train synchronous Advantage Actor-Critic (A2C) with VectorEnv & multimodal encoders."""
    cfg = config or A2CConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    ctx = setup_online_run(
        env_or_factory,
        num_envs=cfg.num_envs,
        vec_mode=cfg.vec_mode,
        extractor_type=cfg.extractor_type,
        normalize_obs=cfg.normalize_obs,
        normalize_reward=cfg.normalize_reward,
        gamma=cfg.gamma,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_freq=cfg.checkpoint_freq,
        eval_freq=cfg.eval_freq,
        save_best_model=cfg.save_best_model,
        keep_last_k=cfg.keep_last_k,
        tb_log_dir=cfg.tb_log_dir,
        eval_env_or_factory=eval_env_or_factory,
        device=dev,
    )
    env, obs_rms, ret_rms = ctx.env, ctx.obs_rms, ctx.ret_rms
    net, policy_spec, act_shape, act_dtype = build_actor_critic_for_env(
        env,
        hidden_dims=cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
        device=dev,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    buffer = RolloutBuffer(
        cfg.n_steps,
        env.obs_spec,
        act_shape,
        action_dtype=act_dtype,
        mask_dim=env.action_dim if (env.is_discrete or env.action_nvec is not None) else None,
        num_envs=env.num_envs,
        batched=True,
    )
    cfg_dict = asdict(cfg)
    models_map: dict[str, Any] = {"policy": net}
    opts_map = {"policy": optimizer}

    steps, ep_idx, learning_curve = maybe_resume_online_run(
        ctx, resume_from=cfg.resume_from, models=models_map, optimizers=opts_map, rng=rng
    )
    t0 = time.perf_counter()
    obs, mask_batch = env.reset_batch(seed=cfg.seed)
    saw_action_mask = mask_batch is not None
    entropies, ev_list = [], []

    while steps < cfg.total_timesteps:
        buffer.reset()
        last_dones = np.zeros((env.num_envs,), dtype=np.bool_)
        for _ in range(cfg.n_steps):
            if steps >= cfg.total_timesteps:
                break
            if obs_rms is not None:
                obs_rms.update(obs)
                obs_proc = obs_rms.normalize(obs)
            else:
                obs_proc = obs

            obs_t = obs_to_tensor(obs_proc, env.obs_spec, dev, add_batch_dim=False)
            if mask_batch is not None:
                saw_action_mask = True
            mask_t = torch.as_tensor(mask_batch, device=dev, dtype=torch.bool) if mask_batch is not None else None
            with torch.no_grad():
                if env.is_discrete or env.action_nvec is not None:
                    act_t, lp_t, _, val_t = net.get_action_and_value(obs_t, action_mask=mask_t)
                else:
                    act_t, lp_t, _, val_t = net.get_action_and_value(obs_t)
                act_env = act_t.cpu().numpy()

            sb = env.step_batch(act_env)
            valid_count = int(np.sum(sb.valid_transition))
            step_rewards = ret_rms.normalize_batch(sb.rewards, sb.dones) if ret_rms is not None else sb.rewards.copy()
            trunc_mask = sb.valid_transition & sb.truncateds & (~sb.terminateds)
            if np.any(trunc_mask):
                trunc_raw = slice_obs(sb.final_obs, trunc_mask)
                trunc_proc = obs_rms.normalize(trunc_raw) if obs_rms is not None else trunc_raw
                with torch.no_grad():
                    trunc_vals = net.get_value(obs_to_tensor(trunc_proc, env.obs_spec, dev)).cpu().numpy()
                step_rewards[trunc_mask] += cfg.gamma * trunc_vals

            buffer.add_batch(
                obs_proc,
                act_env,
                lp_t.cpu().numpy(),
                step_rewards,
                sb.terminateds,
                val_t.cpu().numpy(),
                action_masks=mask_batch
                if mask_batch is not None
                else (
                    np.ones((env.num_envs, env.action_dim), dtype=np.bool_)
                    if (env.is_discrete or env.action_nvec is not None)
                    else None
                ),
                dones=sb.dones,
                valid_mask=sb.valid_transition,
            )
            steps += valid_count
            last_dones = sb.dones
            for ep_ret, _ in sb.completed_episodes:
                ep_idx += 1
                if ep_idx % 5 == 0 or len(learning_curve) == 0 or steps >= cfg.total_timesteps:
                    learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
                if ctx.ckpt_mgr is not None:
                    ctx.ckpt_mgr.log_scalars(steps, {"train/episode_return": float(ep_ret)})
            obs, mask_batch = sb.obs, sb.action_masks

        if buffer.ptr == 0:
            break
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            next_v = net.get_value(obs_to_tensor(next_proc, env.obs_spec, dev)).cpu().numpy()
        buffer.compute_returns_and_advantage(next_v, last_dones, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        b_obs, b_act, _, b_adv, b_ret, b_mask, b_vals = buffer.flatten_batch(dev)
        if b_adv.numel() == 0:
            break
        if env.is_discrete or env.action_nvec is not None:
            _, new_lp, entropy, new_val = net.get_action_and_value(b_obs, action=b_act, action_mask=b_mask)
        else:
            _, new_lp, entropy, new_val = net.get_action_and_value(b_obs, action=b_act)

        # [AI-Prompt Hook]: Single-step policy gradient loss; replace with clipped surrogate for PPO.
        if b_adv.numel() > 1:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        pg_loss = -(new_lp * b_adv).mean()
        v_loss = 0.5 * F.mse_loss(new_val, b_ret)
        loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
        optimizer.step()

        entropies.append(float(entropy.mean().item()))
        ev_list.append(explained_variance(b_vals.cpu().numpy(), b_ret.cpu().numpy()))
        maybe_periodic_hooks(
            ctx,
            steps,
            algo_id=META.id,
            policy_kind="actor_critic",
            policy_spec=policy_spec,
            primary_model_key="policy",
            models=models_map,
            optimizers=opts_map,
            config_dict=cfg_dict,
            eval_episodes=cfg.eval_episodes,
            seed=cfg.seed,
            ep_idx=ep_idx,
            learning_curve=learning_curve,
            rng=rng,
        )

    diagnostics: dict[str, Any] = {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "final_entropy": float(entropies[-1]) if entropies else 0.0,
        "mean_explained_variance": float(np.mean(ev_list)) if ev_list else 0.0,
        "is_continuous": 0.0 if (env.is_discrete or env.action_nvec is not None) else 1.0,
    }
    if isinstance(net, ContinuousActorCritic):
        diagnostics["final_logstd_mean"] = float(net.actor_logstd.detach().mean().item())

    return finalize_online_run(
        ctx,
        algo_id=META.id,
        policy_kind="actor_critic",
        policy_spec=policy_spec,
        primary_model_key="policy",
        models=models_map,
        optimizers=opts_map,
        result_model=net,
        config_dict=cfg_dict,
        steps=steps,
        t0=t0,
        ep_idx=ep_idx,
        learning_curve=learning_curve,
        train_diagnostics=diagnostics,
        eval_episodes=cfg.eval_episodes,
        seed=cfg.seed,
        saw_action_mask=saw_action_mask,
        rng=rng,
    )
