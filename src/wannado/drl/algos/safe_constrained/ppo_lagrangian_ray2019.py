"""PPO-Lagrangian: Primal-Dual Constrained MDP PPO (Ray, Achiam, & Amodei, 2019).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : safe_constrained / S (Primary Constrained On-Policy Workhorse)
- Parent        : `ppo` (Schulman et al., 2017)
- Key Diff      : Adds a dedicated safety Cost Value Critic V_C(s), cost GAE advantage A_C,
                  and online dual ascent on Lagrange multiplier lambda >= 0 to maximize
                  return while enforcing E[cumulative cost] <= cost_limit.
- Extension Hook [AI-Prompt Hook]:
    1. Replace linear dual ascent with PID damping on continuous tasks -> `sac_lagrangian`.
    2. Replace hard cost expectations with tail-quantile CVaR optimization -> `cvar_ppo`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import (
    build_actor_critic_for_env,
    evaluate_safety_episodes,
    extract_batch_costs,
    finalize_online_run,
    maybe_periodic_hooks,
    maybe_resume_online_run,
    setup_online_run,
)
from wannado.drl.core.buffers import RolloutBuffer
from wannado.drl.core.networks import ContinuousActorCritic, ValueNetwork
from wannado.drl.core.obs_spec import obs_to_tensor, slice_obs
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import compute_gae, explained_variance, resolve_device, set_global_seed

META = AlgoMeta(
    id="ppo_lagrangian",
    name="PPO-Lagrangian (Primal-Dual CMDP)",
    family="safe_constrained",
    paradigm="on_policy",
    tier="S",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="ppo",
    diff_from_parent="Adds separate Cost Critic V_C(s), cost GAE advantage, and primal-dual Lagrange multiplier ascent for safety constraints.",
    summary_plain="Constrained PPO that maximizes reward subject to hard safety, power, or violation budgets via adaptive Lagrange penalties.",
    when_to_use="Discrete or continuous control with explicit per-step `info['cost']` or violation limits where penalties cannot be tuned once statically.",
    when_not_to_use="Slow continuous simulators needing off-policy replay (`sac_lagrangian`) or unconstrained reward maximization (`ppo`).",
    hyperparam_intuition={
        "cost_limit": "Target maximum expected cumulative constraint cost per episode (budget d).",
        "lagrange_lr": "Dual ascent step size raising penalty weight lambda whenever episode cost > cost_limit.",
        "init_lambda": "Initial safety multiplier balancing reward vs constraint satisfaction.",
        "cost_gamma": "Discount horizon for future constraint violations.",
    },
    extension_hooks=(
        "Swap to off-policy replay + PID dual control for continuous Box tasks -> `sac_lagrangian`",
        "Shape tail-risk quantiles instead of mean constraint cost -> `cvar_ppo`",
    ),
    module_path="wannado.drl.algos.safe_constrained.ppo_lagrangian_ray2019",
    citation=AlgoCitation(
        key="ray2019benchmarking",
        title="Benchmarking Safe Exploration in Deep Reinforcement Learning",
        authors="Alex Ray, Joshua Achiam, and Dario Amodei",
        venue="OpenAI Technical Report",
        year=2019,
        url="https://cdn.openai.com/safexp-short.pdf",
        bibtex=(
            "@article{ray2019benchmarking,\n"
            "  title={Benchmarking Safe Exploration in Deep Reinforcement Learning},\n"
            "  author={Ray, Alex and Achiam, Joshua and Amodei, Dario},\n"
            "  journal={OpenAI Technical Report},\n"
            "  year={2019},\n"
            "  url={https://cdn.openai.com/safexp-short.pdf}\n"
            "}"
        ),
        ieee='A. Ray, J. Achiam, and D. Amodei, "Benchmarking safe exploration in deep reinforcement learning," OpenAI Technical Report, 2019.',
        apa="Ray, A., Achiam, J., & Amodei, D. (2019). Benchmarking safe exploration in deep reinforcement learning. OpenAI Technical Report.",
    ),
)


@dataclass
class PPOLagrangianConfig:
    total_timesteps: int = 16_000
    n_steps: int = 256
    update_epochs: int = 6
    minibatch_size: int = 64
    lr: float = 3e-4
    cost_critic_lr: float = 3e-4
    gamma: float = 0.99
    cost_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    cost_limit: float = 1.0
    init_lambda: float = 0.1
    lagrange_lr: float = 0.05
    max_lambda: float = 50.0
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


def train_ppo_lagrangian(
    env_or_factory: Any,
    config: PPOLagrangianConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
    cost_fn: Callable[[Any, Any, Any, dict[str, Any]], float] | None = None,
) -> TrainResult:
    """Train PPO-Lagrangian with separate reward & safety cost critics and online dual ascent."""
    cfg = config or PPOLagrangianConfig()
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
    cost_critic = ValueNetwork(
        env.obs_spec,
        cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    cost_optimizer = torch.optim.Adam(cost_critic.parameters(), lr=cfg.cost_critic_lr, eps=1e-5)

    buffer = RolloutBuffer(
        cfg.n_steps,
        env.obs_spec,
        act_shape,
        action_dtype=act_dtype,
        mask_dim=env.action_dim if (env.is_discrete or env.action_nvec is not None) else None,
        num_envs=env.num_envs,
        batched=True,
    )
    costs_buf = np.zeros((cfg.n_steps, env.num_envs), dtype=np.float32)
    cost_vals_buf = np.zeros((cfg.n_steps, env.num_envs), dtype=np.float32)

    cfg_dict = asdict(cfg)
    models_map: dict[str, Any] = {"policy": net, "cost_critic": cost_critic}
    opts_map: dict[str, torch.optim.Optimizer] = {"policy": optimizer, "cost_critic": cost_optimizer}
    resumed_extra: dict[str, Any] = {}
    steps, ep_idx, learning_curve = maybe_resume_online_run(
        ctx, resume_from=cfg.resume_from, models=models_map, optimizers=opts_map, rng=rng, extra_out=resumed_extra
    )
    lagrange_lambda = float(resumed_extra.get("lagrange_lambda", max(0.0, cfg.init_lambda)))

    t0 = time.perf_counter()
    obs, mask_batch = env.reset_batch(seed=cfg.seed)
    saw_action_mask = mask_batch is not None
    saw_cost_signal = False
    running_ep_cost = np.zeros((env.num_envs,), dtype=np.float64)
    recent_ep_costs: list[float] = []
    ent_hist, kl_hist, ev_hist, lambda_hist = [], [], [], [lagrange_lambda]

    while steps < cfg.total_timesteps:
        if cfg.anneal_lr:
            frac = 1.0 - (steps / max(1, cfg.total_timesteps))
            optimizer.param_groups[0]["lr"] = max(1e-6, cfg.lr * frac)
            cost_optimizer.param_groups[0]["lr"] = max(1e-6, cfg.cost_critic_lr * frac)

        buffer.reset()
        costs_buf.fill(0.0)
        cost_vals_buf.fill(0.0)
        batch_ep_costs: list[float] = []
        last_dones = np.zeros((env.num_envs,), dtype=np.bool_)

        for t in range(cfg.n_steps):
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
                cval_t = cost_critic(obs_t).squeeze(-1)
                act_env = act_t.cpu().numpy()

            sb = env.step_batch(act_env)
            valid_count = int(np.sum(sb.valid_transition))
            step_rewards = ret_rms.normalize_batch(sb.rewards, sb.dones) if ret_rms is not None else sb.rewards.copy()
            step_costs_raw, step_has_cost = extract_batch_costs(
                sb.infos,
                env.num_envs,
                obs=obs,
                acts=act_env,
                next_obs=sb.final_obs,
                cost_fn=cost_fn,
            )
            step_costs = step_costs_raw.copy()
            if step_has_cost:
                saw_cost_signal = True

            trunc_mask = sb.valid_transition & sb.truncateds & (~sb.terminateds)
            if np.any(trunc_mask):
                trunc_raw = slice_obs(sb.final_obs, trunc_mask)
                trunc_proc = obs_rms.normalize(trunc_raw) if obs_rms is not None else trunc_raw
                with torch.no_grad():
                    trunc_ot = obs_to_tensor(trunc_proc, env.obs_spec, dev)
                    trunc_vals = net.get_value(trunc_ot).cpu().numpy()
                    trunc_cvals = cost_critic(trunc_ot).squeeze(-1).cpu().numpy()
                step_rewards[trunc_mask] += cfg.gamma * trunc_vals
                step_costs[trunc_mask] += cfg.cost_gamma * trunc_cvals

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
            costs_buf[t] = step_costs * sb.valid_transition.astype(np.float32)
            cost_vals_buf[t] = cval_t.cpu().numpy()

            for slot in range(env.num_envs):
                if not sb.valid_transition[slot]:
                    continue
                running_ep_cost[slot] += float(step_costs_raw[slot])
                if sb.dones[slot]:
                    ep_c = float(running_ep_cost[slot])
                    batch_ep_costs.append(ep_c)
                    recent_ep_costs.append(ep_c)
                    running_ep_cost[slot] = 0.0

            steps += valid_count
            last_dones = sb.dones
            for ep_ret, _ in sb.completed_episodes:
                ep_idx += 1
                if ep_idx % 5 == 0 or len(learning_curve) == 0 or steps >= cfg.total_timesteps:
                    learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
                if ctx.ckpt_mgr is not None:
                    ctx.ckpt_mgr.log_scalars(steps, {"train/episode_return": float(ep_ret), "train/lambda": lagrange_lambda})
            obs, mask_batch = sb.obs, sb.action_masks

        if buffer.ptr == 0 or buffer.ptr * env.num_envs < 2:
            break
        n_roll = buffer.ptr
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            next_ot = obs_to_tensor(next_proc, env.obs_spec, dev)
            next_v = net.get_value(next_ot).cpu().numpy()
            next_cv = cost_critic(next_ot).squeeze(-1).cpu().numpy()
        buffer.compute_returns_and_advantage(next_v, last_dones, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
        c_adv_np, c_ret_np = compute_gae(
            costs_buf[:n_roll],
            cost_vals_buf[:n_roll],
            buffer.terminateds[:n_roll],
            next_cv,
            last_dones.astype(np.float32),
            gamma=cfg.cost_gamma,
            gae_lambda=cfg.gae_lambda,
            dones=buffer.dones[:n_roll],
        )

        # Dual ascent step on Lagrange multiplier using realized episode constraint violations
        if batch_ep_costs:
            j_cost = float(np.mean(batch_ep_costs))
        else:
            valid_c = costs_buf[:n_roll].reshape(-1)[buffer.valid_masks[:n_roll].reshape(-1) > 0.5]
            j_cost = float(np.mean(valid_c) * cfg.n_steps) if valid_c.size else 0.0
        # [AI-Prompt Hook]: Online primal-dual Lagrange multiplier update enforcing E[cost] <= cost_limit.
        lagrange_lambda = float(
            np.clip(lagrange_lambda + cfg.lagrange_lr * (j_cost - cfg.cost_limit), 0.0, cfg.max_lambda)
        )
        lambda_hist.append(lagrange_lambda)

        obs_t, act_t, lp_t, adv_t, ret_t, mask_t, _ = buffer.flatten_batch(dev)
        valid_flat = buffer.valid_masks[:n_roll].reshape(-1) > 0.5
        c_adv_t = torch.as_tensor(c_adv_np.reshape(-1)[valid_flat], device=dev, dtype=torch.float32)
        c_ret_t = torch.as_tensor(c_ret_np.reshape(-1)[valid_flat], device=dev, dtype=torch.float32)

        total = int(lp_t.shape[0])
        mb_size = min(int(cfg.minibatch_size), total)
        for _ in range(cfg.update_epochs):
            indices = np.arange(total)
            rng.shuffle(indices)
            for start in range(0, total, mb_size):
                idx = indices[start : start + mb_size]
                if len(idx) < 2:
                    continue
                idx_t = torch.as_tensor(idx, device=dev, dtype=torch.long)
                mb_obs = slice_obs(obs_t, idx_t)
                mb_act = act_t[idx_t]
                mb_old_lp = lp_t[idx_t]
                mb_adv = adv_t[idx_t]
                mb_ret = ret_t[idx_t]
                mb_c_adv = c_adv_t[idx_t]
                mb_c_ret = c_ret_t[idx_t]
                mb_mask = mask_t[idx_t] if mask_t is not None else None

                if env.is_discrete or env.action_nvec is not None:
                    _, new_lp, entropy, new_val = net.get_action_and_value(mb_obs, action=mb_act, action_mask=mb_mask)
                else:
                    _, new_lp, entropy, new_val = net.get_action_and_value(mb_obs, action=mb_act)

                log_ratio = new_lp - mb_old_lp
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kl_hist.append(float(((ratio - 1.0) - log_ratio).mean().item()))

                adv_r_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                c_std = float(mb_c_adv.std().item())
                adv_c_n = (mb_c_adv - mb_c_adv.mean()) / max(c_std, 1e-2) if c_std > 1e-4 else torch.zeros_like(mb_c_adv)
                # Normalized primal-dual advantage keeps gradient step size stable as lambda grows
                comb_adv = (adv_r_n - lagrange_lambda * adv_c_n) / (1.0 + lagrange_lambda)

                pg_loss1 = -comb_adv * ratio
                pg_loss2 = -comb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * F.mse_loss(new_val, mb_ret)
                loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pred_c_val = cost_critic(mb_obs).squeeze(-1)
                c_loss = 0.5 * F.mse_loss(pred_c_val, mb_c_ret)
                cost_optimizer.zero_grad()
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(cost_critic.parameters(), cfg.max_grad_norm)
                cost_optimizer.step()

                ent_hist.append(float(entropy.mean().item()))

        if np.any(valid_flat):
            ev_hist.append(
                explained_variance(
                    buffer.values[:n_roll].reshape(-1)[valid_flat],
                    buffer.returns[:n_roll].reshape(-1)[valid_flat],
                )
            )
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
            extra_state_fields={"lagrange_lambda": lagrange_lambda},
        )

    diagnostics: dict[str, Any] = {
        "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
        "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
        "mean_approx_kl": float(np.mean(kl_hist)) if kl_hist else 0.0,
        "mean_explained_variance": float(np.mean(ev_hist)) if ev_hist else 0.0,
        "final_lagrange_lambda": float(lagrange_lambda),
        "cost_limit": float(cfg.cost_limit),
        "mean_train_ep_cost": float(np.mean(recent_ep_costs[-20:])) if recent_ep_costs else 0.0,
        "is_continuous": 0.0 if (env.is_discrete or env.action_nvec is not None) else 1.0,
    }
    if isinstance(net, ContinuousActorCritic):
        diagnostics["final_logstd_mean"] = float(net.actor_logstd.detach().mean().item())

    result = finalize_online_run(
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
        extra_state_fields={"lagrange_lambda": lagrange_lambda},
    )
    eval_target = eval_env_or_factory if eval_env_or_factory is not None else env_or_factory
    if not callable(eval_target) and hasattr(eval_target, "raw_env"):
        eval_target = eval_target.raw_env
    safety_metrics = evaluate_safety_episodes(
        eval_target,
        result.policy_bundle,
        num_episodes=cfg.eval_episodes,
        seed=cfg.seed + 10_000,
        cost_limit=cfg.cost_limit,
        cost_fn=cost_fn,
    )
    result.train_diagnostics.update(safety_metrics)
    result.train_diagnostics["cost_budget_exceeded"] = bool(
        (saw_cost_signal or safety_metrics["saw_cost_signal"] > 0.5)
        and safety_metrics["eval_mean_cost"] > max(cfg.cost_limit * 1.15, cfg.cost_limit + 1e-4)
    )
    return result
