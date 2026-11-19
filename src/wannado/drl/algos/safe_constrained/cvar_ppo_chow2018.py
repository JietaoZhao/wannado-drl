"""CVaR-PPO: Tail-Risk Sensitive PPO via Percentile CVaR Optimization (Chow et al., JMLR 2018).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : safe_constrained / S (Risk-Averse Tail-Return & Heavy-Tail Control)
- Parent        : `ppo` (Schulman et al., 2017)
- Key Diff      : Optimizes a smooth Rockafellar-Uryasev lower-alpha Conditional Value-at-Risk
                  (CVaR) tail utility nu - (nu - G)_+ / alpha blended with risk-neutral reward,
                  protecting against catastrophic lower-tail crashes that mean-reward PPO ignores.
- Extension Hook [AI-Prompt Hook]:
    1. Enforce explicit `info['cost']` constraint budget instead of return-tail CVaR -> `ppo_lagrangian`.
    2. Use recurrent GRU state for partial observability -> `ppo_gru`.
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
from wannado.drl.core.offline_risk import compute_cvar
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import explained_variance, resolve_device, set_global_seed

META = AlgoMeta(
    id="cvar_ppo",
    name="CVaR-PPO (Percentile Tail-Risk PPO)",
    family="safe_constrained",
    paradigm="on_policy",
    tier="S",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "finance", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="ppo",
    diff_from_parent="Replaces pure expected-return advantage with Rockafellar-Uryasev CVaR tail-risk utility shaping across the worst alpha outcomes.",
    summary_plain="Risk-averse PPO optimizing the worst-alpha% outcomes (CVaR) rather than risk-neutral average return — suppresses heavy-tail crashes.",
    when_to_use="Finance, portfolio drawdowns, reliability/QoS SLA tails, or any task where occasional catastrophic crashes outweigh slightly higher average rewards.",
    when_not_to_use="Pure deterministic reward maximization (`ppo`) or hard cumulative cost budgets `E[cost] <= d` (`ppo_lagrangian`).",
    hyperparam_intuition={
        "cvar_alpha": "Worst-case quantile fraction focused by CVaR (e.g., 0.20 focuses on the worst 20% trajectories).",
        "mean_mix": "Trade-off between risk-neutral mean return (1.0) and pure worst-alpha CVaR (0.0).",
        "nu_ema": "Exponential moving average rate updating the empirical alpha-quantile Value-at-Risk threshold nu.",
    },
    extension_hooks=(
        "Add explicit per-step `info['cost']` dual budget constraint -> `ppo_lagrangian`",
        "Lower `cvar_alpha` (e.g. 0.1) and set `mean_mix=0.05` for strict worst-case drawdown defense",
    ),
    module_path="wannado.drl.algos.safe_constrained.cvar_ppo_chow2018",
    citation=AlgoCitation(
        key="chow2018risk",
        title="Risk-Constrained Reinforcement Learning with Percentile Risk Criteria",
        authors="Yinlam Chow, Mohammad Ghavamzadeh, Lucas Janson, and Marco Pavone",
        venue="Journal of Machine Learning Research (JMLR), Vol. 18, No. 167, pp. 1-51",
        year=2018,
        url="https://arxiv.org/abs/1512.01629",
        bibtex=(
            "@article{chow2018risk,\n"
            "  title={Risk-Constrained Reinforcement Learning with Percentile Risk Criteria},\n"
            "  author={Chow, Yinlam and Ghavamzadeh, Mohammad and Janson, Lucas and Pavone, Marco},\n"
            "  journal={Journal of Machine Learning Research},\n"
            "  volume={18},\n"
            "  number={167},\n"
            "  pages={1--51},\n"
            "  year={2018},\n"
            "  url={https://arxiv.org/abs/1512.01629}\n"
            "}"
        ),
        ieee='Y. Chow, M. Ghavamzadeh, L. Janson, and M. Pavone, "Risk-constrained reinforcement learning with percentile risk criteria," JMLR, vol. 18, no. 167, pp. 1-51, 2018.',
        apa="Chow, Y., Ghavamzadeh, M., Janson, L., & Pavone, M. (2018). Risk-constrained reinforcement learning with percentile risk criteria. JMLR, 18(167).",
    ),
)


@dataclass
class CVaRPPOConfig:
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
    cvar_alpha: float = 0.25
    mean_mix: float = 0.15
    nu_ema: float = 0.25
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


def train_cvar_ppo(
    env_or_factory: Any,
    config: CVaRPPOConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train CVaR-PPO optimizing lower-alpha tail return utility via Rockafellar-Uryasev shortfall shaping."""
    cfg = config or CVaRPPOConfig()
    if not (0.0 < cfg.cvar_alpha <= 1.0):
        raise ValueError(f"cvar_alpha must be in (0, 1], got {cfg.cvar_alpha}.")
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
    resumed_extra: dict[str, Any] = {}
    steps, ep_idx, learning_curve = maybe_resume_online_run(
        ctx, resume_from=cfg.resume_from, models=models_map, optimizers=opts_map, rng=rng, extra_out=resumed_extra
    )
    cvar_nu: float | None = resumed_extra.get("cvar_nu")

    t0 = time.perf_counter()
    obs, mask_batch = env.reset_batch(seed=cfg.seed)
    saw_action_mask = mask_batch is not None
    recent_ep_returns: list[float] = []
    running_ep_ret = np.zeros((env.num_envs,), dtype=np.float64)
    ent_hist, kl_hist, ev_hist = [], [], []

    while steps < cfg.total_timesteps:
        if cfg.anneal_lr:
            frac = 1.0 - (steps / max(1, cfg.total_timesteps))
            optimizer.param_groups[0]["lr"] = max(1e-6, cfg.lr * frac)

        buffer.reset()
        ep_step_traces: list[list[int]] = [[] for _ in range(env.num_envs)]
        finished_episodes_in_rollout: list[tuple[int, list[int], float]] = []
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

            for slot in range(env.num_envs):
                if not sb.valid_transition[slot]:
                    continue
                ep_step_traces[slot].append(t)
                running_ep_ret[slot] += float(sb.rewards[slot])
                if sb.dones[slot]:
                    ep_r = float(running_ep_ret[slot])
                    recent_ep_returns.append(ep_r)
                    if len(recent_ep_returns) > 64:
                        recent_ep_returns.pop(0)
                    finished_episodes_in_rollout.append((slot, list(ep_step_traces[slot]), ep_r))
                    ep_step_traces[slot].clear()
                    running_ep_ret[slot] = 0.0

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
        n_roll = buffer.ptr
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            next_v = net.get_value(obs_to_tensor(next_proc, env.obs_spec, dev)).cpu().numpy()

        # [AI-Prompt Hook]: Smooth downside CVaR shortfall shaping (Rockafellar-Uryasev / Mean-Semideviation hybrid).
        # Using EMA of rolling mean/quantile prevents discrete-mode collapse of nu onto a single tail atom.
        if len(recent_ep_returns) >= 3:
            batch_q = float(np.quantile(recent_ep_returns, cfg.cvar_alpha))
            batch_mean = float(np.mean(recent_ep_returns))
            batch_med = float(np.median(recent_ep_returns))
            batch_ref = max(batch_q, min(batch_mean, batch_med))
            cvar_nu = batch_ref if cvar_nu is None else float((1.0 - cfg.nu_ema) * cvar_nu + cfg.nu_ema * batch_ref)
        if cvar_nu is not None and finished_episodes_in_rollout:
            alpha_safe = max(float(cfg.cvar_alpha), 0.05)
            mix = float(np.clip(cfg.mean_mix, 0.0, 1.0))
            tail_coef = (1.0 - mix) * max(1.0, (1.0 / alpha_safe) - 1.0)
            for slot, t_list, ep_r in finished_episodes_in_rollout:
                if not t_list:
                    continue
                shortfall = max(0.0, cvar_nu - ep_r)
                if shortfall > 0.0:
                    step_penalty = (tail_coef * shortfall) / len(t_list)
                    for t_idx in t_list:
                        buffer.rewards[t_idx, slot] -= np.float32(step_penalty)

        buffer.compute_returns_and_advantage(next_v, last_dones, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        for _ in range(cfg.update_epochs):
            for mb_obs, mb_act, mb_old_lp, mb_adv, mb_ret, mb_mask in buffer.iter_minibatches(cfg.minibatch_size, rng, dev):
                if env.is_discrete or env.action_nvec is not None:
                    _, new_lp, entropy, new_val = net.get_action_and_value(mb_obs, action=mb_act, action_mask=mb_mask)
                else:
                    _, new_lp, entropy, new_val = net.get_action_and_value(mb_obs, action=mb_act)

                log_ratio = new_lp - mb_old_lp
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kl_hist.append(float(((ratio - 1.0) - log_ratio).mean().item()))

                mb_adv_n = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
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

        valid_flat = buffer.valid_masks[:n_roll].reshape(-1) > 0.5
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
            extra_state_fields={"cvar_nu": cvar_nu},
        )

    diagnostics: dict[str, Any] = {
        "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
        "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
        "mean_approx_kl": float(np.mean(kl_hist)) if kl_hist else 0.0,
        "mean_explained_variance": float(np.mean(ev_hist)) if ev_hist else 0.0,
        "cvar_alpha": float(cfg.cvar_alpha),
        "final_cvar_nu": float(cvar_nu) if cvar_nu is not None else 0.0,
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
        extra_state_fields={"cvar_nu": cvar_nu},
    )
    if result.eval_stats is not None:
        result.train_diagnostics["eval_cvar_alpha"] = compute_cvar(result.eval_stats.returns, alpha=cfg.cvar_alpha)
    return result
