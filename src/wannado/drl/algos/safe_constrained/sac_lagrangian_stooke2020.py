"""SAC-Lagrangian: Off-Policy Safe RL with PID Lagrangian Control (Stooke, Achiam, & Abbeel, ICML 2020).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : safe_constrained / S (Sample-Efficient Continuous Constrained Control)
- Parent        : `sac` (Haarnoja et al., ICML 2018)
- Key Diff      : Augments Soft Actor-Critic with a twin Safety Cost Q-Critic Q_C(s,a)
                  and Stooke et al. (ICML 2020) PID Lagrangian feedback (with anti-windup
                  integral clamping and non-negative derivative damping) to curb constraint
                  oscillation and overshoot on continuous control.
- Extension Hook [AI-Prompt Hook]:
    1. Use on-policy discrete/continuous CMDP dual ascent -> `ppo_lagrangian`.
    2. Optimize coherent tail-quantile CVaR directly without explicit `info['cost']` -> `cvar_ppo`.
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
    evaluate_safety_episodes,
    extract_batch_costs,
    finalize_online_run,
    maybe_periodic_hooks,
    maybe_resume_online_run,
    setup_online_run,
)
from wannado.drl.core.buffers import ReplayBuffer
from wannado.drl.core.networks import SquashedGaussianActor, TwinQCritic
from wannado.drl.core.obs_spec import obs_to_tensor
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import hard_update, polyak_update, resolve_device, set_global_seed

META = AlgoMeta(
    id="sac_lagrangian",
    name="SAC-Lagrangian (PID Safe SAC)",
    family="safe_constrained",
    paradigm="off_policy",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("continuous",),
    env_modalities=("continuous_control", "custom_sim"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="sac",
    diff_from_parent="Adds twin Safety Cost Q-Critic Q_C(s,a) + Stooke et al. (2020) PID Lagrangian dual controller.",
    summary_plain="High-sample-efficiency continuous Safe RL using PID feedback on safety constraint violations to eliminate oscillatory penalty tuning.",
    when_to_use="Continuous Box control with safety budgets (`info['cost']`) and costly simulators requiring off-policy sample efficiency.",
    when_not_to_use="Discrete/masked action spaces (`ppo_lagrangian`) or unconstrained continuous tasks (`sac`).",
    hyperparam_intuition={
        "cost_limit": "Episode constraint budget d (E[sum cost] <= cost_limit).",
        "pid_kp": "Proportional reaction gain to immediate violation (J_C - cost_limit).",
        "pid_ki": "Integral accumulation gain (clamped >= 0) tracking steady-state shadow price.",
        "pid_kd": "Derivative overshoot damping reacting only to rising cost slopes.",
    },
    extension_hooks=(
        "Add discrete/masked support + on-policy GAE cost critic -> `ppo_lagrangian`",
        "Add offline demo symmetric sampling (`DualDemoReplayBuffer`) for zero-violation startup",
    ),
    module_path="wannado.drl.algos.safe_constrained.sac_lagrangian_stooke2020",
    citation=AlgoCitation(
        key="stooke2020responsive",
        title="Responsive Safety in Reinforcement Learning by PID Lagrangian Methods",
        authors="Adam Stooke, Joshua Achiam, and Pieter Abbeel",
        venue="Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR Vol. 119, pp. 9133-9143",
        year=2020,
        url="https://arxiv.org/abs/2007.03964",
        bibtex=(
            "@inproceedings{stooke2020responsive,\n"
            "  title={Responsive Safety in Reinforcement Learning by PID Lagrangian Methods},\n"
            "  author={Stooke, Adam and Achiam, Joshua and Abbeel, Pieter},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={119},\n"
            "  pages={9133--9143},\n"
            "  year={2020},\n"
            "  url={https://arxiv.org/abs/2007.03964}\n"
            "}"
        ),
        ieee='A. Stooke, J. Achiam, and P. Abbeel, "Responsive safety in reinforcement learning by PID Lagrangian methods," in Proc. ICML, PMLR 119, pp. 9133-9143, 2020.',
        apa="Stooke, A., Achiam, J., & Abbeel, P. (2020). Responsive safety in reinforcement learning by PID Lagrangian methods. ICML (PMLR 119).",
    ),
)


@dataclass
class SACLagrangianConfig:
    total_timesteps: int = 8_000
    learning_starts: int = 400
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    cost_critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    cost_gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    autotune_alpha: bool = True
    cost_limit: float = 1.0
    pid_kp: float = 0.05
    pid_ki: float = 0.02
    pid_kd: float = 0.01
    init_lambda: float = 0.1
    max_lambda: float = 50.0
    buffer_size: int = 50_000
    batch_size: int = 64
    train_freq: int = 1
    gradient_steps: int = 0
    hidden_dims: tuple[int, ...] = (128, 128)
    num_envs: int = 1
    vec_mode: str = "auto"
    extractor_type: str = "auto"
    extractor_kwargs: dict[str, Any] | None = None
    normalize_obs: bool = False
    normalize_reward: bool = False
    eval_episodes: int = 10
    checkpoint_dir: str | None = None
    checkpoint_freq: int = 0
    eval_freq: int = 0
    save_best_model: bool = True
    keep_last_k: int = 3
    save_replay_buffer: bool = False
    resume_from: str | None = None
    tb_log_dir: str | None = None
    device: str = "auto"
    seed: int = 42


def train_sac_lagrangian(
    env_or_factory: Any,
    config: SACLagrangianConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
    cost_fn: Callable[[Any, Any, Any, dict[str, Any]], float] | None = None,
) -> TrainResult:
    """Train SAC-Lagrangian with twin safety Q-critic and PID dual control on continuous tasks."""
    cfg = config or SACLagrangianConfig()
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
        save_replay_buffer=cfg.save_replay_buffer,
        tb_log_dir=cfg.tb_log_dir,
        eval_env_or_factory=eval_env_or_factory,
        device=dev,
    )
    env, obs_rms, ret_rms = ctx.env, ctx.obs_rms, ctx.ret_rms
    if env.is_discrete or env.action_nvec is not None or env.action_low is None or env.action_high is None:
        ctx.env.close()
        raise ValueError("sac_lagrangian requires bounded continuous Box action spaces.")

    actor = SquashedGaussianActor(
        env.obs_spec,
        env.action_dim,
        env.action_low,
        env.action_high,
        cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    critic = TwinQCritic(
        env.obs_spec,
        env.action_dim,
        cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    target_critic = TwinQCritic(
        env.obs_spec,
        env.action_dim,
        cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    hard_update(critic, target_critic)

    cost_critic = TwinQCritic(
        env.obs_spec,
        env.action_dim,
        cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    target_cost_critic = TwinQCritic(
        env.obs_spec,
        env.action_dim,
        cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    hard_update(cost_critic, target_cost_critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    cost_critic_opt = torch.optim.Adam(cost_critic.parameters(), lr=cfg.cost_critic_lr)
    log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), device=dev, requires_grad=cfg.autotune_alpha)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None
    target_entropy = -float(env.action_dim)

    replay = ReplayBuffer(cfg.buffer_size, env.obs_spec, env.action_dim, discrete_action=False, device=dev, track_costs=True)

    policy_spec = {
        "net_class": "SquashedGaussianActor",
        "init_kwargs": {
            "action_dim": int(env.action_dim),
            "action_low": env.action_low.tolist(),
            "action_high": env.action_high.tolist(),
            "hidden_dims": list(cfg.hidden_dims),
            "extractor_type": cfg.extractor_type,
            "extractor_kwargs": cfg.extractor_kwargs,
        },
    }
    models_map: dict[str, Any] = {
        "actor": actor,
        "critic": critic,
        "target_critic": target_critic,
        "cost_critic": cost_critic,
        "target_cost_critic": target_cost_critic,
        "log_alpha": log_alpha,
    }
    opts_map: dict[str, torch.optim.Optimizer] = {
        "actor": actor_opt,
        "critic": critic_opt,
        "cost_critic": cost_critic_opt,
    }
    if alpha_opt is not None:
        opts_map["alpha"] = alpha_opt

    cfg_dict = asdict(cfg)
    resumed_extra: dict[str, Any] = {}
    steps, ep_idx, learning_curve = maybe_resume_online_run(
        ctx,
        resume_from=cfg.resume_from,
        models=models_map,
        optimizers=opts_map,
        rng=rng,
        replay_buffer=replay if cfg.resume_from else None,
        extra_out=resumed_extra,
    )
    lagrange_lambda = float(resumed_extra.get("lagrange_lambda", max(0.0, cfg.init_lambda)))
    pid_i = float(resumed_extra.get("pid_i", max(0.0, lagrange_lambda / max(cfg.pid_ki, 1e-4))))
    prev_cost_err: float | None = resumed_extra.get("prev_cost_err")

    t0 = time.perf_counter()
    obs, _ = env.reset_batch(seed=cfg.seed)
    train_freq = max(1, int(cfg.train_freq))
    grad_steps = int(cfg.gradient_steps) if cfg.gradient_steps > 0 else max(1, env.num_envs)
    next_train_step = max(cfg.learning_starts, ((steps // train_freq) + 1) * train_freq)

    saw_cost_signal = False
    running_ep_cost = np.zeros((env.num_envs,), dtype=np.float64)
    recent_ep_costs: list[float] = []
    q_hist, c_q_hist, ent_hist = [], [], []

    def _pid_update(measured_ep_cost: float) -> None:
        nonlocal lagrange_lambda, pid_i, prev_cost_err
        err = float(measured_ep_cost) - float(cfg.cost_limit)
        pid_p = cfg.pid_kp * err
        pid_i = max(0.0, pid_i + cfg.pid_ki * err)
        pid_d = cfg.pid_kd * max(0.0, err - prev_cost_err) if prev_cost_err is not None else 0.0
        prev_cost_err = err
        lagrange_lambda = float(np.clip(pid_p + pid_i + pid_d, 0.0, cfg.max_lambda))

    while steps < cfg.total_timesteps:
        if obs_rms is not None:
            obs_rms.update(obs)
            obs_proc = obs_rms.normalize(obs)
        else:
            obs_proc = obs

        if steps < cfg.learning_starts:
            acts = rng.uniform(env.action_low, env.action_high, size=(env.num_envs, env.action_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs_proc, env.obs_spec, dev, add_batch_dim=False)
                a_t, _ = actor.get_action(obs_t)
                acts = a_t.cpu().numpy().astype(np.float32)

        sb = env.step_batch(acts)
        valid_count = int(np.sum(sb.valid_transition))
        train_rewards = ret_rms.normalize_batch(sb.rewards, sb.dones) if ret_rms is not None else sb.rewards
        step_costs, step_has_cost = extract_batch_costs(
            sb.infos,
            env.num_envs,
            obs=obs,
            acts=acts,
            next_obs=sb.final_obs,
            cost_fn=cost_fn,
        )
        if step_has_cost:
            saw_cost_signal = True

        next_proc = obs_rms.normalize(sb.final_obs) if obs_rms is not None else sb.final_obs
        replay.add_batch(obs_proc, acts, train_rewards, next_proc, sb.terminateds, valid_mask=sb.valid_transition, costs=step_costs)

        for slot in range(env.num_envs):
            if not sb.valid_transition[slot]:
                continue
            running_ep_cost[slot] += float(step_costs[slot])
            if sb.dones[slot]:
                ep_c = float(running_ep_cost[slot])
                recent_ep_costs.append(ep_c)
                running_ep_cost[slot] = 0.0
                if steps >= cfg.learning_starts:
                    _pid_update(ep_c)

        steps += valid_count
        for ep_ret, _ in sb.completed_episodes:
            ep_idx += 1
            if ep_idx % 5 == 0 or len(learning_curve) == 0 or steps >= cfg.total_timesteps:
                learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
            if ctx.ckpt_mgr is not None:
                ctx.ckpt_mgr.log_scalars(steps, {"train/episode_return": float(ep_ret), "train/lambda": lagrange_lambda})
        obs = sb.obs

        if steps >= next_train_step and replay.size >= cfg.batch_size:
            for _ in range(grad_steps):
                b_obs, b_act, b_rew, b_next, b_term, b_cost = replay.sample_with_costs(cfg.batch_size, rng=rng)
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    next_a, next_lp = actor.get_action(b_next)
                    tq1, tq2 = target_critic(b_next, next_a)
                    min_tq = torch.min(tq1, tq2) - alpha * next_lp
                    td_target = b_rew.unsqueeze(-1) + cfg.gamma * (1.0 - b_term.unsqueeze(-1)) * min_tq

                    tc1, tc2 = target_cost_critic(b_next, next_a)
                    avg_tc = 0.5 * (tc1 + tc2)
                    cost_target = b_cost.unsqueeze(-1) + cfg.cost_gamma * (1.0 - b_term.unsqueeze(-1)) * avg_tc

                q1, q2 = critic(b_obs, b_act)
                critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                q_hist.append(float(q1.mean().item()))

                c1, c2 = cost_critic(b_obs, b_act)
                cost_loss = F.mse_loss(c1, cost_target) + F.mse_loss(c2, cost_target)
                cost_critic_opt.zero_grad()
                cost_loss.backward()
                cost_critic_opt.step()
                c_q_hist.append(float(c1.mean().item()))

                pi_a, log_pi = actor.get_action(b_obs)
                q1_pi, q2_pi = critic(b_obs, pi_a)
                c1_pi, c2_pi = cost_critic(b_obs, pi_a)
                avg_c_pi = 0.5 * (c1_pi + c2_pi)
                # [AI-Prompt Hook]: Constrained soft actor objective penalizing expected safety Q_C(s,a).
                raw_actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi) + lagrange_lambda * avg_c_pi).mean()
                actor_loss = raw_actor_loss / max(1.0, 1.0 + 0.05 * lagrange_lambda)
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                ent_hist.append(float(-log_pi.mean().item()))

                if cfg.autotune_alpha and alpha_opt is not None:
                    alpha_loss = (-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()

                polyak_update(critic, target_critic, cfg.tau)
                polyak_update(cost_critic, target_cost_critic, cfg.tau)

            while next_train_step <= steps:
                next_train_step += train_freq

        maybe_periodic_hooks(
            ctx,
            steps,
            algo_id=META.id,
            policy_kind="squashed_actor",
            policy_spec=policy_spec,
            primary_model_key="actor",
            models=models_map,
            optimizers=opts_map,
            config_dict=cfg_dict,
            eval_episodes=cfg.eval_episodes,
            seed=cfg.seed,
            ep_idx=ep_idx,
            learning_curve=learning_curve,
            rng=rng,
            replay_buffer=replay if ctx.save_replay_buffer else None,
            extra_state_fields={"lagrange_lambda": lagrange_lambda, "pid_i": pid_i, "prev_cost_err": prev_cost_err},
        )

    diagnostics: dict[str, Any] = {
        "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
        "max_q_value": float(max(q_hist, key=abs)) if q_hist else 0.0,
        "mean_cost_q_value": float(np.mean(c_q_hist)) if c_q_hist else 0.0,
        "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
        "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
        "final_alpha": float(log_alpha.exp().item()),
        "final_lagrange_lambda": float(lagrange_lambda),
        "cost_limit": float(cfg.cost_limit),
        "mean_train_ep_cost": float(np.mean(recent_ep_costs[-20:])) if recent_ep_costs else 0.0,
        "is_continuous": 1.0,
    }
    result = finalize_online_run(
        ctx,
        algo_id=META.id,
        policy_kind="squashed_actor",
        policy_spec=policy_spec,
        primary_model_key="actor",
        models=models_map,
        optimizers=opts_map,
        result_model={"actor": actor, "critic": critic, "cost_critic": cost_critic},
        config_dict=cfg_dict,
        steps=steps,
        t0=t0,
        ep_idx=ep_idx,
        learning_curve=learning_curve,
        train_diagnostics=diagnostics,
        eval_episodes=cfg.eval_episodes,
        seed=cfg.seed,
        saw_action_mask=False,
        rng=rng,
        replay_buffer=replay if ctx.save_replay_buffer else None,
        extra_state_fields={"lagrange_lambda": lagrange_lambda, "pid_i": pid_i, "prev_cost_err": prev_cost_err},
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
