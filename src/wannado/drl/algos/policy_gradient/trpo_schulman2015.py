"""TRPO: Trust Region Policy Optimization (Schulman et al., ICML 2015).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / A (Theoretical ancestor of PPO)
- Parent        : `reinforce` (Williams, 1992)
- Key Diff      : Prevents destructive policy collapses by constraining each policy step
                  inside a KL-divergence trust region using Conjugate Gradient + Fisher
                  Hessian-vector products + backtracking line search.
- Extension Hook [AI-Prompt Hook]:
    1. Replace expensive second-order CG + KL constraint with first-order clipped ratio -> `ppo`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence

from wannado.drl.core.algo_helpers import (
    build_actor_critic_for_env,
    finalize_online_run,
    maybe_periodic_hooks,
    maybe_resume_online_run,
    setup_online_run,
)
from wannado.drl.core.buffers import RolloutBuffer
from wannado.drl.core.networks import ContinuousActorCritic, MultiCategorical
from wannado.drl.core.obs_spec import obs_to_tensor, slice_obs
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import explained_variance, resolve_device, set_global_seed

META = AlgoMeta(
    id="trpo",
    name="Trust Region Policy Optimization (TRPO)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="A",
    on_policy=True,
    sample_efficiency="low",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="reinforce",
    diff_from_parent="Enforces monotonic KL trust-region constraints via Conjugate Gradient Fisher-vector products and line search.",
    summary_plain="Second-order policy gradient ensuring every update stays inside a mathematically safe KL trust region.",
    when_to_use="Studying policy trust-region theory or sensitive control tasks needing strict KL guarantees.",
    when_not_to_use="Large vision/sequence networks where Hessian-vector products are too slow (`ppo` achieves similar stability with first-order updates).",
    hyperparam_intuition={
        "max_kl": "Hard KL divergence radius allowed per policy step (0.01 = conservative trust region).",
        "cg_iters": "Conjugate Gradient iterations approximating F^{-1} g.",
        "n_steps": "Batch size per trust-region update; larger batches give cleaner Fisher curvature.",
    },
    extension_hooks=(
        "Replace CG + backtracking line search with clipped surrogate -> `ppo`",
    ),
    module_path="wannado.drl.algos.policy_gradient.trpo_schulman2015",
    citation=AlgoCitation(
        key="schulman2015trust",
        title="Trust Region Policy Optimization",
        authors="John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel",
        venue="Proceedings of the 32nd International Conference on Machine Learning (ICML), PMLR 37",
        year=2015,
        url="http://proceedings.mlr.press/v37/schulman15.html",
        bibtex=(
            "@inproceedings{schulman2015trust,\n"
            "  title={Trust Region Policy Optimization},\n"
            "  author={Schulman, John and Levine, Sergey and Moritz, Philipp and Jordan, Michael I. and Abbeel, Pieter},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={37},\n"
            "  year={2015},\n"
            "  url={http://proceedings.mlr.press/v37/schulman15.html}\n"
            "}"
        ),
        ieee='J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel, "Trust region policy optimization," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 37, 2015.',
        apa="Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). Trust region policy optimization. ICML (PMLR 37).",
    ),
)


@dataclass
class TRPOConfig:
    total_timesteps: int = 12_000
    n_steps: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_kl: float = 0.015
    cg_damping: float = 0.1
    cg_iters: int = 10
    vf_lr: float = 1e-3
    vf_iters: int = 5
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


def _flat_params(params: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def _set_flat_params(params: list[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(flat[offset : offset + n].view_as(p))
            offset += n


def _conjugate_gradient(avp_fn: Any, b: torch.Tensor, iters: int = 10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iters):
        if rdotr < 1e-10:
            break
        avp = avp_fn(p)
        alpha = rdotr / (torch.dot(p, avp) + 1e-8)
        x = x + alpha * p
        r = r - alpha * avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def train_trpo(
    env_or_factory: Any,
    config: TRPOConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Trust Region Policy Optimization (TRPO) with VectorEnv & multimodal encoders."""
    cfg = config or TRPOConfig()
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
    actor_params = net.actor_parameters()
    critic_opt = torch.optim.Adam(net.critic.parameters(), lr=cfg.vf_lr)
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
    opts_map = {"critic": critic_opt}

    steps, ep_idx, learning_curve = maybe_resume_online_run(
        ctx, resume_from=cfg.resume_from, models=models_map, optimizers=opts_map, rng=rng
    )
    t0 = time.perf_counter()
    obs, mask_batch = env.reset_batch(seed=cfg.seed)
    saw_action_mask = mask_batch is not None
    kl_hist, ent_hist, ev_hist = [], [], []

    def _get_dist(b_o: Any, b_m: torch.Tensor | None) -> Any:
        if env.is_discrete:
            return net.get_dist(b_o, action_mask=b_m)
        return net.get_dist(b_o)

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
            obs, mask_batch = sb.obs, sb.action_masks

        if buffer.ptr == 0 or buffer.ptr * env.num_envs < 2:
            break
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            next_v = net.get_value(obs_to_tensor(next_proc, env.obs_spec, dev)).cpu().numpy()
        buffer.compute_returns_and_advantage(next_v, last_dones, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        b_obs, b_act, b_old_lp, b_adv, b_ret, b_mask, b_vals = buffer.flatten_batch(dev)
        if b_adv.numel() < 2:
            break
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        with torch.no_grad():
            old_dist = _get_dist(b_obs, b_mask)
            if isinstance(old_dist, MultiCategorical):
                old_dist = old_dist.detached_copy()
            elif env.is_discrete:
                old_dist = type(old_dist)(logits=old_dist.logits.detach())
            else:
                old_dist = type(old_dist)(old_dist.mean.detach(), old_dist.stddev.detach())

        # [AI-Prompt Hook]: Fisher-vector product + Conjugate Gradient natural policy step.
        def _surrogate(
            obs_b: Any = b_obs,
            mask_b: torch.Tensor | None = b_mask,
            act_b: torch.Tensor = b_act,
            old_lp_b: torch.Tensor = b_old_lp,
            adv_b: torch.Tensor = b_adv,
        ) -> torch.Tensor:
            dist = _get_dist(obs_b, mask_b)
            lp = dist.log_prob(act_b) if (env.is_discrete or env.action_nvec is not None) else dist.log_prob(act_b).sum(-1)
            return (torch.exp(lp - old_lp_b) * adv_b).mean()

        loss_old = _surrogate()
        grads = torch.autograd.grad(loss_old, actor_params)
        loss_grad = torch.cat([g.reshape(-1) for g in grads]).detach()

        def _fvp(
            v: torch.Tensor,
            obs_b: Any = b_obs,
            mask_b: torch.Tensor | None = b_mask,
            ref_dist: Any = old_dist,
        ) -> torch.Tensor:
            dist = _get_dist(obs_b, mask_b)
            kl = kl_divergence(ref_dist, dist).mean()
            kl_grads = torch.autograd.grad(kl, actor_params, create_graph=True)
            flat_kl_grad = torch.cat([g.reshape(-1) for g in kl_grads])
            kl_v = torch.dot(flat_kl_grad, v)
            hvp = torch.autograd.grad(kl_v, actor_params)
            return torch.cat([h.contiguous().reshape(-1) for h in hvp]).detach() + cfg.cg_damping * v

        step_dir = _conjugate_gradient(_fvp, loss_grad, iters=cfg.cg_iters)
        shs = 0.5 * torch.dot(step_dir, _fvp(step_dir))
        if shs > 1e-10:
            step_scale = torch.sqrt(cfg.max_kl / shs)
            full_step = step_scale * step_dir
            old_theta = _flat_params(actor_params)
            accepted_kl = 0.0
            for j in range(10):
                frac = 0.5**j
                _set_flat_params(actor_params, old_theta + frac * full_step)
                with torch.no_grad():
                    new_loss = _surrogate()
                    new_dist = _get_dist(b_obs, b_mask)
                    kl_val = float(kl_divergence(old_dist, new_dist).mean().item())
                if new_loss > loss_old and kl_val <= cfg.max_kl * 1.5:
                    accepted_kl = kl_val
                    break
            else:
                _set_flat_params(actor_params, old_theta)
            kl_hist.append(accepted_kl)

        with torch.no_grad():
            d_now = _get_dist(b_obs, b_mask)
            ent_vec = d_now.entropy() if (env.is_discrete or env.action_nvec is not None) else d_now.entropy().sum(-1)
            ent_hist.append(float(ent_vec.mean().item()))

        for _ in range(cfg.vf_iters):
            v_pred = net.get_value(b_obs)
            v_loss = 0.5 * F.mse_loss(v_pred, b_ret)
            critic_opt.zero_grad()
            v_loss.backward()
            critic_opt.step()
        ev_hist.append(explained_variance(b_vals.cpu().numpy(), b_ret.cpu().numpy()))
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
        "mean_kl": float(np.mean(kl_hist)) if kl_hist else 0.0,
        "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
        "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
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
