"""PPO: Proximal Policy Optimization (Schulman et al., arXiv:1707.06347).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / S (Default First-Try Benchmark Across Discrete & Continuous DRL)
- Parent        : `trpo` (Schulman et al., ICML 2015)
- Key Diff      : Replaces TRPO's expensive second-order Conjugate Gradient KL solver with
                  a simple first-order clipped surrogate objective min(r * A, clip(r, 1-e, 1+e) * A),
                  multi-epoch minibatch SGD, GAE advantages, and optional invalid-action masking.
- Extension Hook [AI-Prompt Hook]:
    1. Inject GRU hidden memory across time steps for POMDPs -> `ppo_gru`.
    2. Share global state in Critic across N decentralized actors -> `mappo`.
    3. Drop Critic completely and normalize across K rollouts per prompt -> `grpo`.
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
    id="ppo",
    name="Proximal Policy Optimization (PPO-Clip)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="S",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="trpo",
    diff_from_parent="Replaces second-order Fisher KL trust region with first-order clipped surrogate objective + minibatch epochs.",
    summary_plain="Reliable default workhorse for both discrete (with action masking) and continuous control.",
    when_to_use="First-try online DRL benchmark when simulator speed is reasonable and stability matters most.",
    when_not_to_use="Expensive slow simulators needing maximal sample reuse (`sac`/`td3`), partial observability (`ppo_gru`), or offline logs (`iql_offline`).",
    hyperparam_intuition={
        "clip_coef": "Trust-region clip width (0.2 keeps policy changes conservative per batch).",
        "n_steps": "Steps collected before each policy update batch.",
        "ent_coef": "Exploration curiosity weight preventing premature greedy collapse.",
        "gamma": "Future reward discount horizon.",
    },
    extension_hooks=(
        "Add GRU state recurrence across episode steps -> `ppo_gru`",
        "Add asymmetric Clip-Higher (eps_low vs eps_high) -> `dapo`-style exploration preservation",
        "Use centralized global state in critic for N agents -> `mappo`",
    ),
    module_path="wannado.drl.algos.policy_gradient.ppo_schulman2017",
    citation=AlgoCitation(
        key="schulman2017proximal",
        title="Proximal Policy Optimization Algorithms",
        authors="John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov",
        venue="arXiv preprint arXiv:1707.06347",
        year=2017,
        url="https://arxiv.org/abs/1707.06347",
        bibtex=(
            "@article{schulman2017proximal,\n"
            "  title={Proximal Policy Optimization Algorithms},\n"
            "  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},\n"
            "  journal={arXiv preprint arXiv:1707.06347},\n"
            "  year={2017},\n"
            "  url={https://arxiv.org/abs/1707.06347}\n"
            "}"
        ),
        ieee='J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.',
        apa="Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.",
    ),
)


@dataclass
class PPOConfig:
    total_timesteps: int = 16_000
    n_steps: int = 256
    update_epochs: int = 6
    minibatch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dims: tuple[int, ...] = (64, 64)
    normalize_obs: bool = False
    normalize_reward: bool = False
    anneal_lr: bool = True
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


def train_ppo(
    env_or_factory: Any,
    config: PPOConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train PPO-Clip with true VectorEnv rollouts, GAE, CNN/Dict/Attention encoders, and checkpoint resume."""
    cfg = config or PPOConfig()
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
        mask_dim=env.action_dim if env.is_discrete else None,
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
    ent_hist, kl_hist, ev_hist = [], [], []

    while steps < cfg.total_timesteps:
        if cfg.anneal_lr:
            frac = 1.0 - (steps / max(1, cfg.total_timesteps))
            optimizer.param_groups[0]["lr"] = max(1e-6, cfg.lr * frac)

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
                if env.is_discrete:
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
                else (np.ones((env.num_envs, env.action_dim), dtype=np.bool_) if env.is_discrete else None),
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

        if buffer.ptr == 0 or buffer.ptr * env.num_envs < 2:
            break
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            next_v = net.get_value(obs_to_tensor(next_proc, env.obs_spec, dev)).cpu().numpy()
        buffer.compute_returns_and_advantage(next_v, last_dones, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        for _ in range(cfg.update_epochs):
            for mb_obs, mb_act, mb_old_lp, mb_adv, mb_ret, mb_mask in buffer.iter_minibatches(cfg.minibatch_size, rng, dev):
                if env.is_discrete:
                    _, new_lp, entropy, new_val = net.get_action_and_value(mb_obs, action=mb_act, action_mask=mb_mask)
                else:
                    _, new_lp, entropy, new_val = net.get_action_and_value(mb_obs, action=mb_act)

                log_ratio = new_lp - mb_old_lp
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kl_hist.append(float(((ratio - 1.0) - log_ratio).mean().item()))

                mb_adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                # [AI-Prompt Hook]: PPO-Clip surrogate; customize asymmetric Clip-Higher or KL penalty here.
                pg_loss1 = -mb_adv_n * ratio
                pg_loss2 = -mb_adv_n * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * F.mse_loss(new_val, mb_ret)
                loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()
                ent_hist.append(float(entropy.mean().item()))

        valid_flat = buffer.valid_masks[: buffer.ptr].reshape(-1) > 0.5
        if np.any(valid_flat):
            ev_hist.append(
                explained_variance(
                    buffer.values[: buffer.ptr].reshape(-1)[valid_flat],
                    buffer.returns[: buffer.ptr].reshape(-1)[valid_flat],
                )
            )
        if ctx.ckpt_mgr is not None and kl_hist:
            ctx.ckpt_mgr.log_scalars(steps, {"train/approx_kl": kl_hist[-1], "train/entropy": ent_hist[-1]})

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
        "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
        "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
        "mean_approx_kl": float(np.mean(kl_hist)) if kl_hist else 0.0,
        "mean_explained_variance": float(np.mean(ev_hist)) if ev_hist else 0.0,
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
