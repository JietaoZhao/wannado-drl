"""Shared lifecycle helpers for vectorized rollout/replay setup, periodic eval, checkpointing, and PolicyBundle export."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from wannado.drl.core.checkpoint import CheckpointManager, PolicyBundle, action_meta_from_env
from wannado.drl.core.provenance import resolve_run_optimizations
from wannado.drl.core.types import TrainResult
from wannado.drl.core.wrappers import (
    DictRunningMeanStd,
    RewardNormalizer,
    UniversalGymAdapter,
    VectorizedEnvAdapter,
    make_vec_env,
)


@dataclass
class OnlineRunContext:
    """Runtime container holding vectorized training env, isolated eval env, normalizers, and checkpointer."""

    env: VectorizedEnvAdapter
    eval_env: UniversalGymAdapter
    can_mid_eval: bool
    obs_rms: DictRunningMeanStd | None
    ret_rms: RewardNormalizer | None
    ckpt_mgr: CheckpointManager | None
    action_meta: dict[str, Any]
    device: torch.device
    next_eval_step: int = 0
    next_ckpt_step: int = 0
    eval_freq: int = 0
    checkpoint_freq: int = 0
    save_replay_buffer: bool = False
    episode_returns_history: list[float] = field(default_factory=list)
    episode_lengths_history: list[int] = field(default_factory=list)


def setup_online_run(
    env_or_factory: Any,
    *,
    num_envs: int = 1,
    vec_mode: str = "auto",
    extractor_type: str = "auto",
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    gamma: float = 0.99,
    checkpoint_dir: str | None = None,
    checkpoint_freq: int = 0,
    eval_freq: int = 0,
    save_best_model: bool = True,
    keep_last_k: int = 3,
    save_replay_buffer: bool = False,
    tb_log_dir: str | None = None,
    eval_env_or_factory: Any | None = None,
    device: torch.device,
) -> OnlineRunContext:
    """Create a ``VectorizedEnvAdapter``, isolated ``eval_env``, normalizers, and ``CheckpointManager``."""
    env = make_vec_env(
        env_or_factory,
        num_envs=num_envs,
        vec_mode=vec_mode,
        extractor_type=extractor_type,
        structured_obs=True,
    )
    can_mid_eval = eval_env_or_factory is not None or env._eval_factory is not None
    if eval_freq > 0 and not can_mid_eval:
        warnings.warn(
            "Periodic mid-training evaluation (`eval_freq > 0`) skipped because `env_or_factory` is a raw env instance "
            "without a factory or `eval_env_or_factory`; pass a callable factory (`lambda: ...`) for isolated periodic eval.",
            RuntimeWarning,
            stacklevel=3,
        )
    eval_env = env.build_eval_env(eval_env_or_factory)
    obs_rms = DictRunningMeanStd(env.obs_spec) if normalize_obs else None
    ret_rms = RewardNormalizer(num_envs=env.num_envs, gamma=gamma) if normalize_reward else None

    use_ckpt_mgr = (
        checkpoint_dir is not None
        or tb_log_dir is not None
        or (eval_freq > 0 and can_mid_eval)
        or checkpoint_freq > 0
    )
    ckpt_mgr = (
        CheckpointManager(
            checkpoint_dir,
            keep_last_k=keep_last_k,
            save_best_model=save_best_model,
            tb_log_dir=tb_log_dir,
        )
        if use_ckpt_mgr
        else None
    )
    return OnlineRunContext(
        env=env,
        eval_env=eval_env,
        can_mid_eval=can_mid_eval,
        obs_rms=obs_rms,
        ret_rms=ret_rms,
        ckpt_mgr=ckpt_mgr,
        action_meta=action_meta_from_env(env),
        device=device,
        next_eval_step=int(eval_freq) if eval_freq > 0 else 0,
        next_ckpt_step=int(checkpoint_freq) if checkpoint_freq > 0 else 0,
        eval_freq=int(eval_freq),
        checkpoint_freq=int(checkpoint_freq),
        save_replay_buffer=bool(save_replay_buffer),
    )


def build_bundle_for_model(
    ctx: OnlineRunContext,
    *,
    algo_id: str,
    policy_kind: str,
    primary_model: nn.Module,
    policy_spec: dict[str, Any],
    config_dict: dict[str, Any],
    freeze_obs_rms: bool = True,
) -> PolicyBundle:
    """Create an inference-ready ``PolicyBundle`` tied to ``obs_rms`` statistics."""
    return PolicyBundle(
        algo_id=algo_id,
        policy_kind=policy_kind,
        model=primary_model,
        obs_spec=ctx.env.obs_spec,
        action_meta=ctx.action_meta,
        obs_rms=ctx.obs_rms,
        policy_spec=policy_spec,
        config=config_dict,
        device=ctx.device,
        freeze_obs_rms=freeze_obs_rms,
    )


def maybe_resume_online_run(
    ctx: OnlineRunContext,
    *,
    resume_from: str | None,
    models: dict[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    rng: np.random.Generator | None = None,
    replay_buffer: Any | None = None,
    extra_out: dict[str, Any] | None = None,
) -> tuple[int, int, list[dict[str, float]]]:
    """Restore model, optimizer, RMS, RNG, and progress if ``resume_from`` is set."""
    if not resume_from:
        return 0, 0, []
    mgr = ctx.ckpt_mgr if ctx.ckpt_mgr is not None else CheckpointManager()
    info = mgr.load_checkpoint(
        resume_from,
        models=models,
        optimizers=optimizers,
        obs_rms=ctx.obs_rms,
        ret_rms=ctx.ret_rms,
        rng=rng,
        replay_buffer=replay_buffer,
        device=ctx.device,
    )
    if ctx.ckpt_mgr is not None and ctx.ckpt_mgr is not mgr:
        ctx.ckpt_mgr.best_eval_mean = mgr.best_eval_mean
        ctx.ckpt_mgr.eval_history = list(mgr.eval_history)
    step = int(info["step"])
    extra_state = info.get("extra_state", {})
    if extra_out is not None:
        extra_out.update(extra_state)
    ep_idx = int(extra_state.get("ep_idx", 0))
    curve = list(extra_state.get("learning_curve", []))
    if ctx.eval_freq > 0:
        while ctx.next_eval_step <= step:
            ctx.next_eval_step += ctx.eval_freq
    if ctx.checkpoint_freq > 0:
        while ctx.next_ckpt_step <= step:
            ctx.next_ckpt_step += ctx.checkpoint_freq
    return step, ep_idx, curve


def maybe_periodic_hooks(
    ctx: OnlineRunContext,
    step: int,
    *,
    algo_id: str,
    policy_kind: str,
    policy_spec: dict[str, Any],
    primary_model_key: str,
    models: dict[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    config_dict: dict[str, Any],
    eval_episodes: int,
    seed: int,
    ep_idx: int,
    learning_curve: list[dict[str, float]],
    rng: np.random.Generator | None = None,
    replay_buffer: Any | None = None,
    extra_state_fields: dict[str, Any] | None = None,
) -> None:
    """Run threshold-scheduled periodic eval & checkpoint saving cleanly under VectorEnv step jumps."""
    if ctx.ckpt_mgr is None:
        return
    extra_state: dict[str, Any] = {"ep_idx": int(ep_idx), "learning_curve": learning_curve}
    if extra_state_fields:
        extra_state.update(extra_state_fields)

    if ctx.eval_freq > 0 and ctx.can_mid_eval and step >= ctx.next_eval_step:
        prev_frozen = ctx.obs_rms.frozen if ctx.obs_rms is not None else True
        bundle = build_bundle_for_model(
            ctx,
            algo_id=algo_id,
            policy_kind=policy_kind,
            primary_model=models[primary_model_key],
            policy_spec=policy_spec,
            config_dict=config_dict,
            freeze_obs_rms=False,
        )
        ctx.ckpt_mgr.run_periodic_eval(
            step,
            eval_env=ctx.eval_env,
            policy_fn=bundle,
            obs_rms=ctx.obs_rms,
            num_episodes=eval_episodes,
            seed=seed + 10_000,
            algo_id=algo_id,
            policy_kind=policy_kind,
            policy_spec=policy_spec,
            primary_model_key=primary_model_key,
            models=models,
            optimizers=optimizers,
            obs_spec=ctx.env.obs_spec,
            action_meta=ctx.action_meta,
            config=config_dict,
            ret_rms=ctx.ret_rms,
            rng=rng,
            extra_state=extra_state,
        )
        if ctx.obs_rms is not None:
            ctx.obs_rms.freeze(prev_frozen)
        while ctx.next_eval_step <= step:
            ctx.next_eval_step += ctx.eval_freq

    if ctx.checkpoint_freq > 0 and ctx.ckpt_mgr.root is not None and step >= ctx.next_ckpt_step:
        ctx.ckpt_mgr.save_checkpoint(
            step,
            algo_id=algo_id,
            policy_kind=policy_kind,
            policy_spec=policy_spec,
            primary_model_key=primary_model_key,
            models=models,
            optimizers=optimizers,
            obs_spec=ctx.env.obs_spec,
            action_meta=ctx.action_meta,
            config=config_dict,
            obs_rms=ctx.obs_rms,
            ret_rms=ctx.ret_rms,
            rng=rng,
            replay_buffer=replay_buffer,
            save_replay_buffer=ctx.save_replay_buffer,
            extra_state=extra_state,
            tag="step",
        )
        while ctx.next_ckpt_step <= step:
            ctx.next_ckpt_step += ctx.checkpoint_freq


def finalize_online_run(
    ctx: OnlineRunContext,
    *,
    algo_id: str,
    policy_kind: str,
    policy_spec: dict[str, Any],
    primary_model_key: str,
    models: dict[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    result_model: Any,
    config_dict: dict[str, Any],
    steps: int,
    t0: float,
    ep_idx: int,
    learning_curve: list[dict[str, float]],
    train_diagnostics: dict[str, Any],
    eval_episodes: int,
    seed: int,
    saw_action_mask: bool = False,
    rng: np.random.Generator | None = None,
    replay_buffer: Any | None = None,
    extra: dict[str, Any] | None = None,
    extra_state_fields: dict[str, Any] | None = None,
) -> TrainResult:
    """Freeze RMS stats, run deterministic final evaluation, persist final checkpoints, and emit TrainResult."""
    if ctx.obs_rms is not None:
        ctx.obs_rms.freeze(True)

    bundle = build_bundle_for_model(
        ctx,
        algo_id=algo_id,
        policy_kind=policy_kind,
        primary_model=models[primary_model_key],
        policy_spec=policy_spec,
        config_dict=config_dict,
    )
    from wannado.drl.mdp.baselines import evaluate_policy_episodes

    eval_stats = evaluate_policy_episodes(
        ctx.eval_env, bundle, num_episodes=eval_episodes, seed=seed + 10_000
    )

    best_path: str | None = None
    last_path: str | None = None
    eval_history: list[dict[str, float]] = []
    if ctx.ckpt_mgr is not None:
        extra_state: dict[str, Any] = {"ep_idx": int(ep_idx), "learning_curve": learning_curve}
        if extra_state_fields:
            extra_state.update(extra_state_fields)
        if ctx.ckpt_mgr.root is not None:
            saved_last = ctx.ckpt_mgr.save_checkpoint(
                steps,
                algo_id=algo_id,
                policy_kind=policy_kind,
                policy_spec=policy_spec,
                primary_model_key=primary_model_key,
                models=models,
                optimizers=optimizers,
                obs_spec=ctx.env.obs_spec,
                action_meta=ctx.action_meta,
                config=config_dict,
                obs_rms=ctx.obs_rms,
                ret_rms=ctx.ret_rms,
                rng=rng,
                replay_buffer=replay_buffer,
                save_replay_buffer=ctx.save_replay_buffer,
                extra_state=extra_state,
                tag="last",
            )
            last_path = str(saved_last) if saved_last is not None else None
            if ctx.ckpt_mgr.save_best_model and (
                ctx.ckpt_mgr.best_model_path is None or eval_stats.mean_return >= ctx.ckpt_mgr.best_eval_mean
            ):
                ctx.ckpt_mgr.best_eval_mean = float(eval_stats.mean_return)
                saved_best = ctx.ckpt_mgr.save_checkpoint(
                    steps,
                    algo_id=algo_id,
                    policy_kind=policy_kind,
                    policy_spec=policy_spec,
                    primary_model_key=primary_model_key,
                    models=models,
                    optimizers=optimizers,
                    obs_spec=ctx.env.obs_spec,
                    action_meta=ctx.action_meta,
                    config=config_dict,
                    obs_rms=ctx.obs_rms,
                    ret_rms=ctx.ret_rms,
                    rng=rng,
                    extra_state=extra_state,
                    tag="best",
                )
                best_path = str(saved_best) if saved_best is not None else None
            elif ctx.ckpt_mgr.best_model_path is not None:
                best_path = str(ctx.ckpt_mgr.best_model_path)
        eval_history = list(ctx.ckpt_mgr.eval_history)
        ctx.ckpt_mgr.close()

    ctx.env.close()
    if ctx.eval_env.raw_env is not ctx.env.raw_env:
        ctx.eval_env.close()
    opts = resolve_run_optimizations(algo_id, config=config_dict, saw_action_mask=saw_action_mask)
    return TrainResult(
        algo_id=algo_id,
        seed=seed,
        total_timesteps=steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=bundle,
        model=result_model,
        config=config_dict,
        learning_curve=learning_curve,
        train_diagnostics={
            **train_diagnostics,
            "num_envs": float(ctx.env.num_envs),
        },
        eval_stats=eval_stats,
        obs_rms_state=ctx.obs_rms.state_dict() if ctx.obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if ctx.obs_rms is None else ctx.obs_rms.frozen,
        optimizations_used=[o.id for o in opts],
        extra=dict(extra or {}),
        eval_history=eval_history,
        best_model_path=best_path,
        last_checkpoint_path=last_path,
        policy_bundle=bundle,
    )


def build_actor_critic_for_env(
    env: Any,
    *,
    hidden_dims: tuple[int, ...],
    extractor_type: str,
    extractor_kwargs: dict[str, Any] | None,
    device: torch.device,
) -> tuple[Any, dict[str, Any], tuple[int, ...], np.dtype[Any]]:
    """Construct Discrete/MultiDiscrete/Continuous ActorCritic plus checkpoint spec and action buffer layout."""
    from wannado.drl.core.networks import (
        ContinuousActorCritic,
        DiscreteActorCritic,
        MultiDiscreteActorCritic,
    )

    if env.is_discrete:
        net: Any = DiscreteActorCritic(
            env.obs_spec,
            env.action_dim,
            hidden_dims=hidden_dims,
            extractor_type=extractor_type,
            extractor_kwargs=extractor_kwargs,
        ).to(device)
        spec = {
            "net_class": "DiscreteActorCritic",
            "init_kwargs": {
                "action_dim": int(env.action_dim),
                "hidden_dims": list(hidden_dims),
                "extractor_type": extractor_type,
                "extractor_kwargs": extractor_kwargs,
            },
        }
        return net, spec, (), np.dtype(np.int64)
    if env.action_nvec is not None:
        net = MultiDiscreteActorCritic(
            env.obs_spec,
            env.action_nvec,
            hidden_dims=hidden_dims,
            extractor_type=extractor_type,
            extractor_kwargs=extractor_kwargs,
        ).to(device)
        spec = {
            "net_class": "MultiDiscreteActorCritic",
            "init_kwargs": {
                "action_nvec": [int(v) for v in env.action_nvec],
                "hidden_dims": list(hidden_dims),
                "extractor_type": extractor_type,
                "extractor_kwargs": extractor_kwargs,
            },
        }
        return net, spec, (len(env.action_nvec),), np.dtype(np.int64)
    net = ContinuousActorCritic(
        env.obs_spec,
        env.action_dim,
        hidden_dims=hidden_dims,
        action_low=env.action_low,
        action_high=env.action_high,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
    ).to(device)
    spec = {
        "net_class": "ContinuousActorCritic",
        "init_kwargs": {
            "action_dim": int(env.action_dim),
            "hidden_dims": list(hidden_dims),
            "extractor_type": extractor_type,
            "extractor_kwargs": extractor_kwargs,
        },
    }
    return net, spec, (int(env.action_dim),), np.dtype(np.float32)


def extract_step_cost(
    info: dict[str, Any] | None,
    *,
    obs: Any = None,
    action: Any = None,
    next_obs: Any = None,
    cost_fn: Callable[[Any, Any, Any, dict[str, Any]], float] | None = None,
) -> tuple[float, bool]:
    """Extract scalar safety constraint cost from a step transition info dict or custom cost_fn."""
    info_d = info if isinstance(info, dict) else {}
    if cost_fn is not None:
        val = float(cost_fn(obs, action, next_obs, info_d))
        return val, True
    for key in ("cost", "constraint_cost"):
        if key in info_d and isinstance(info_d[key], (int, float)):
            return float(info_d[key]), True
    r_terms = info_d.get("reward_terms")
    if isinstance(r_terms, dict) and isinstance(r_terms.get("cost"), (int, float)):
        return float(r_terms["cost"]), True
    return 0.0, False


def extract_batch_costs(
    infos: list[dict[str, Any]],
    num_envs: int,
    *,
    obs: Any = None,
    acts: np.ndarray | None = None,
    next_obs: Any = None,
    cost_fn: Callable[[Any, Any, Any, dict[str, Any]], float] | None = None,
) -> tuple[np.ndarray, bool]:
    """Extract per-slot constraint costs from VectorStepBatch infos or callable cost_fn."""
    from wannado.drl.core.obs_spec import slice_obs

    costs = np.zeros((num_envs,), dtype=np.float32)
    saw_cost = False
    for i in range(num_envs):
        info_i = infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}
        o_i = slice_obs(obs, i) if obs is not None and cost_fn is not None else None
        no_i = slice_obs(next_obs, i) if next_obs is not None and cost_fn is not None else None
        a_i = acts[i] if acts is not None and cost_fn is not None else None
        c_val, present = extract_step_cost(info_i, obs=o_i, action=a_i, next_obs=no_i, cost_fn=cost_fn)
        costs[i] = np.float32(c_val)
        if present or abs(c_val) > 1e-12:
            saw_cost = True
    return costs, saw_cost


def evaluate_safety_episodes(
    env_or_factory: Any,
    policy_fn: Callable[..., Any],
    *,
    num_episodes: int = 10,
    seed: int = 10_000,
    cost_limit: float = 0.0,
    cost_fn: Callable[[Any, Any, Any, dict[str, Any]], float] | None = None,
) -> dict[str, float]:
    """Evaluate episode cumulative safety costs and tail returns deterministically."""
    from wannado.drl.core.offline_risk import compute_cvar
    from wannado.drl.core.wrappers import ensure_gym_env

    policy_spec = getattr(policy_fn, "obs_spec", None)
    needs_structured = bool(policy_spec is not None and getattr(policy_spec, "kind", "vector") != "vector")
    env = ensure_gym_env(env_or_factory, structured_obs=needs_structured)
    ep_costs: list[float] = []
    ep_rets: list[float] = []
    saw_any_cost = False
    try:
        for ep in range(max(1, int(num_episodes))):
            obs, info = env.reset(seed=seed + ep)
            reset_hook = getattr(policy_fn, "reset", None)
            if callable(reset_hook):
                reset_hook()
            ep_c = 0.0
            ep_r = 0.0
            done = False
            last_info = dict(info)
            while not done:
                act = policy_fn(obs, last_info)
                nxt, rew, term, trunc, last_info = env.step(act)
                c_val, present = extract_step_cost(last_info, obs=obs, action=act, next_obs=nxt, cost_fn=cost_fn)
                if present:
                    saw_any_cost = True
                ep_c += c_val
                ep_r += float(rew)
                obs = nxt
                done = bool(term or trunc)
            ep_costs.append(ep_c)
            ep_rets.append(ep_r)
    finally:
        if env.raw_env is not env_or_factory and hasattr(env.raw_env, "close"):
            env.raw_env.close()

    mean_c = float(np.mean(ep_costs)) if ep_costs else 0.0
    viol_rate = float(np.mean(np.asarray(ep_costs) > float(cost_limit) + 1e-6)) if (ep_costs and saw_any_cost) else 0.0
    return {
        "eval_mean_cost": mean_c,
        "eval_cvar20_cost": -compute_cvar([-c for c in ep_costs], alpha=0.2) if ep_costs else 0.0,
        "cost_violation_rate": viol_rate,
        "eval_cvar20_return": compute_cvar(ep_rets, alpha=0.2) if ep_rets else 0.0,
        "saw_cost_signal": float(saw_any_cost),
    }


def sample_batched_eps_greedy_q(
    q_net: Any,
    obs_proc: Any,
    mask_batch: np.ndarray | None,
    *,
    env: Any,
    eps: float,
    rng: np.random.Generator,
    device: torch.device,
    use_get_q_values: bool = False,
) -> np.ndarray:
    """Select batched epsilon-greedy discrete actions with per-slot action_mask support."""
    from wannado.drl.core.obs_spec import obs_to_tensor

    n = env.num_envs
    acts = np.zeros((n,), dtype=np.int64)
    rand_draw = rng.random(n)
    need_greedy = np.flatnonzero(rand_draw >= eps)
    need_rand = np.flatnonzero(rand_draw < eps)

    for i in need_rand:
        m_i = mask_batch[i] if mask_batch is not None else None
        valid = np.flatnonzero(m_i) if m_i is not None else np.arange(env.action_dim)
        acts[i] = int(rng.choice(valid if valid.size > 0 else np.arange(env.action_dim)))

    if need_greedy.size > 0:
        with torch.no_grad():
            obs_t = obs_to_tensor(obs_proc, env.obs_spec, device, add_batch_dim=False)
            q_all = q_net.get_q_values(obs_t) if use_get_q_values else q_net(obs_t)
            if mask_batch is not None:
                m_t = torch.as_tensor(mask_batch, device=device, dtype=torch.bool)
                has_valid = m_t.any(dim=1, keepdim=True)
                safe_m = torch.where(has_valid, m_t, torch.ones_like(m_t))
                q_all = q_all.masked_fill(~safe_m, -1e9)
            greedy = q_all.argmax(dim=1).cpu().numpy()
        acts[need_greedy] = greedy[need_greedy]
    return acts


def run_value_based_loop(
    env_or_factory: Any,
    *,
    algo_id: str,
    cfg: Any,
    eval_env_or_factory: Any | None,
    build_q_nets: Any,
    loss_fn: Any,
    use_get_q_values: bool = False,
    policy_kind: str = "q_network",
    loss_metric_key: str = "mean_td_loss",
    extra_diagnostics_fn: Any = None,
) -> TrainResult:
    """Shared vectorized, multimodal, checkpointed training loop for DQN / DoubleDQN / Dueling / C51."""
    from dataclasses import asdict

    from wannado.drl.core.buffers import ReplayBuffer
    from wannado.drl.core.utils import hard_update, resolve_device, set_global_seed

    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    ctx = setup_online_run(
        env_or_factory,
        num_envs=getattr(cfg, "num_envs", 1),
        vec_mode=getattr(cfg, "vec_mode", "auto"),
        extractor_type=getattr(cfg, "extractor_type", "auto"),
        normalize_obs=cfg.normalize_obs,
        normalize_reward=getattr(cfg, "normalize_reward", False),
        gamma=cfg.gamma,
        checkpoint_dir=getattr(cfg, "checkpoint_dir", None),
        checkpoint_freq=getattr(cfg, "checkpoint_freq", 0),
        eval_freq=getattr(cfg, "eval_freq", 0),
        save_best_model=getattr(cfg, "save_best_model", True),
        keep_last_k=getattr(cfg, "keep_last_k", 3),
        save_replay_buffer=getattr(cfg, "save_replay_buffer", False),
        tb_log_dir=getattr(cfg, "tb_log_dir", None),
        eval_env_or_factory=eval_env_or_factory,
        device=dev,
    )
    env, obs_rms, ret_rms = ctx.env, ctx.obs_rms, ctx.ret_rms
    if not env.is_discrete:
        ctx.env.close()
        raise ValueError(f"{algo_id} supports Discrete action spaces only.")

    q_net, target_net, policy_spec = build_q_nets(ctx, dev)
    hard_update(q_net, target_net)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(
        cfg.buffer_size,
        env.obs_spec,
        env.action_dim,
        discrete_action=True,
        device=dev,
        track_next_masks=True,
    )
    import inspect

    _loss_accepts_next_mask = len(inspect.signature(loss_fn).parameters) >= 8

    cfg_dict = asdict(cfg)
    models_map: dict[str, Any] = {"q_net": q_net, "target_net": target_net}
    opts_map = {"q_net": optimizer}
    steps, ep_idx, learning_curve = maybe_resume_online_run(
        ctx,
        resume_from=getattr(cfg, "resume_from", None),
        models=models_map,
        optimizers=opts_map,
        rng=rng,
        replay_buffer=replay if getattr(cfg, "resume_from", None) else None,
    )

    t0 = time.perf_counter()
    obs, mask_batch = env.reset_batch(seed=cfg.seed)
    saw_action_mask = mask_batch is not None
    losses_hist: list[float] = []
    q_hist: list[float] = []

    train_freq = max(1, int(getattr(cfg, "train_freq", 1)))
    grad_steps_cfg = int(getattr(cfg, "gradient_steps", 0))
    grad_steps = grad_steps_cfg if grad_steps_cfg > 0 else max(1, env.num_envs)
    next_train_step = max(cfg.learning_starts, ((steps // train_freq) + 1) * train_freq)
    next_target_update = ((steps // cfg.target_update_freq) + 1) * cfg.target_update_freq

    while steps < cfg.total_timesteps:
        decay_steps = max(1.0, cfg.exploration_fraction * cfg.total_timesteps)
        eps = max(cfg.eps_end, cfg.eps_start - (steps / decay_steps) * (cfg.eps_start - cfg.eps_end))

        if obs_rms is not None:
            obs_rms.update(obs)
            obs_proc = obs_rms.normalize(obs)
        else:
            obs_proc = obs

        if mask_batch is not None:
            saw_action_mask = True
        acts = sample_batched_eps_greedy_q(
            q_net,
            obs_proc,
            mask_batch,
            env=env,
            eps=eps,
            rng=rng,
            device=dev,
            use_get_q_values=use_get_q_values,
        )
        sb = env.step_batch(acts)
        valid_count = int(np.sum(sb.valid_transition))
        train_rewards = ret_rms.normalize_batch(sb.rewards, sb.dones) if ret_rms is not None else sb.rewards
        replay.add_batch(
            obs,
            acts,
            train_rewards,
            sb.final_obs,
            sb.terminateds,
            valid_mask=sb.valid_transition,
            next_action_masks=sb.action_masks,
        )
        steps += valid_count

        for ep_ret, _ in sb.completed_episodes:
            ep_idx += 1
            if ep_idx % 5 == 0 or len(learning_curve) == 0 or steps >= cfg.total_timesteps:
                learning_curve.append({"step": float(steps), "episode_return": float(ep_ret), "epsilon": float(eps)})
            if ctx.ckpt_mgr is not None:
                ctx.ckpt_mgr.log_scalars(steps, {"train/episode_return": float(ep_ret), "train/epsilon": float(eps)})
        obs, mask_batch = sb.obs, sb.action_masks
        if mask_batch is not None:
            saw_action_mask = True

        if steps >= next_train_step and replay.size >= cfg.batch_size:
            for _ in range(grad_steps):
                b_obs, b_act, b_rew, b_next, b_term, b_nmask = replay.sample_with_next_masks(
                    cfg.batch_size, rng=rng, obs_rms=obs_rms
                )
                eff_nmask = b_nmask if saw_action_mask else None
                if _loss_accepts_next_mask:
                    loss, q_mean = loss_fn(q_net, target_net, b_obs, b_act, b_rew, b_next, b_term, eff_nmask)
                else:
                    loss, q_mean = loss_fn(q_net, target_net, b_obs, b_act, b_rew, b_next, b_term)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                losses_hist.append(float(loss.item()))
                q_hist.append(float(q_mean))
            while next_train_step <= steps:
                next_train_step += train_freq

        if steps >= next_target_update:
            hard_update(q_net, target_net)
            while next_target_update <= steps:
                next_target_update += cfg.target_update_freq

        maybe_periodic_hooks(
            ctx,
            steps,
            algo_id=algo_id,
            policy_kind=policy_kind,
            policy_spec=policy_spec,
            primary_model_key="q_net",
            models=models_map,
            optimizers=opts_map,
            config_dict=cfg_dict,
            eval_episodes=cfg.eval_episodes,
            seed=cfg.seed,
            ep_idx=ep_idx,
            learning_curve=learning_curve,
            rng=rng,
            replay_buffer=replay if ctx.save_replay_buffer else None,
        )

    diagnostics: dict[str, Any] = {
        loss_metric_key: float(np.mean(losses_hist)) if losses_hist else 0.0,
        "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
    }
    if extra_diagnostics_fn is not None:
        diagnostics.update(extra_diagnostics_fn(losses_hist, q_hist))

    return finalize_online_run(
        ctx,
        algo_id=algo_id,
        policy_kind=policy_kind,
        policy_spec=policy_spec,
        primary_model_key="q_net",
        models=models_map,
        optimizers=opts_map,
        result_model=q_net,
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
        replay_buffer=replay if ctx.save_replay_buffer else None,
    )


def run_offpolicy_continuous_loop(
    env_or_factory: Any,
    *,
    algo_id: str,
    cfg: Any,
    eval_env_or_factory: Any | None,
    build_models_fn: Any,
    sample_action_fn: Any,
    update_fn: Any,
    diagnostics_fn: Any,
) -> TrainResult:
    """Shared vectorized, multimodal, checkpointed training loop for SAC / TD3 / DDPG."""
    from dataclasses import asdict

    from wannado.drl.core.buffers import ReplayBuffer
    from wannado.drl.core.obs_spec import obs_to_tensor
    from wannado.drl.core.utils import resolve_device, set_global_seed

    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    ctx = setup_online_run(
        env_or_factory,
        num_envs=getattr(cfg, "num_envs", 1),
        vec_mode=getattr(cfg, "vec_mode", "auto"),
        extractor_type=getattr(cfg, "extractor_type", "auto"),
        normalize_obs=cfg.normalize_obs,
        normalize_reward=getattr(cfg, "normalize_reward", False),
        gamma=cfg.gamma,
        checkpoint_dir=getattr(cfg, "checkpoint_dir", None),
        checkpoint_freq=getattr(cfg, "checkpoint_freq", 0),
        eval_freq=getattr(cfg, "eval_freq", 0),
        save_best_model=getattr(cfg, "save_best_model", True),
        keep_last_k=getattr(cfg, "keep_last_k", 3),
        save_replay_buffer=getattr(cfg, "save_replay_buffer", False),
        tb_log_dir=getattr(cfg, "tb_log_dir", None),
        eval_env_or_factory=eval_env_or_factory,
        device=dev,
    )
    env, obs_rms, ret_rms = ctx.env, ctx.obs_rms, ctx.ret_rms
    if (
        env.is_discrete
        or env.action_nvec is not None
        or env.action_low is None
        or env.action_high is None
        or not np.all(np.isfinite(env.action_low))
        or not np.all(np.isfinite(env.action_high))
        or np.any(env.action_high <= env.action_low)
    ):
        ctx.env.close()
        raise ValueError(f"{algo_id} requires finite bounded continuous Box action spaces.")

    actor, result_model, policy_kind, policy_spec, models_map, opts_map = build_models_fn(ctx, dev)
    replay = ReplayBuffer(cfg.buffer_size, env.obs_spec, env.action_dim, discrete_action=False, device=dev)

    cfg_dict = asdict(cfg)
    resumed_extra: dict[str, Any] = {}
    steps, ep_idx, learning_curve = maybe_resume_online_run(
        ctx,
        resume_from=getattr(cfg, "resume_from", None),
        models=models_map,
        optimizers=opts_map,
        rng=rng,
        replay_buffer=replay if getattr(cfg, "resume_from", None) else None,
        extra_out=resumed_extra,
    )
    update_step = int(resumed_extra.get("update_step", 0))

    t0 = time.perf_counter()
    obs, _ = env.reset_batch(seed=cfg.seed)
    train_freq = max(1, int(getattr(cfg, "train_freq", 1)))
    grad_steps_cfg = int(getattr(cfg, "gradient_steps", 0))
    grad_steps = grad_steps_cfg if grad_steps_cfg > 0 else max(1, env.num_envs)
    next_train_step = max(cfg.learning_starts, ((steps // train_freq) + 1) * train_freq)

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
                acts = sample_action_fn(actor, obs_t, env, rng)

        sb = env.step_batch(acts)
        valid_count = int(np.sum(sb.valid_transition))
        train_rewards = ret_rms.normalize_batch(sb.rewards, sb.dones) if ret_rms is not None else sb.rewards
        replay.add_batch(obs, acts, train_rewards, sb.final_obs, sb.terminateds, valid_mask=sb.valid_transition)
        steps += valid_count

        for ep_ret, _ in sb.completed_episodes:
            ep_idx += 1
            if ep_idx % 5 == 0 or len(learning_curve) == 0 or steps >= cfg.total_timesteps:
                learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
            if ctx.ckpt_mgr is not None:
                ctx.ckpt_mgr.log_scalars(steps, {"train/episode_return": float(ep_ret)})
        obs = sb.obs

        if steps >= next_train_step and replay.size >= cfg.batch_size:
            for _ in range(grad_steps):
                update_step += 1
                b_obs, b_act, b_rew, b_next, b_term = replay.sample(cfg.batch_size, rng=rng, obs_rms=obs_rms)
                update_fn(update_step, b_obs, b_act, b_rew, b_next, b_term)
            while next_train_step <= steps:
                next_train_step += train_freq

        maybe_periodic_hooks(
            ctx,
            steps,
            algo_id=algo_id,
            policy_kind=policy_kind,
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
            extra_state_fields={"update_step": int(update_step)},
        )

    return finalize_online_run(
        ctx,
        algo_id=algo_id,
        policy_kind=policy_kind,
        policy_spec=policy_spec,
        primary_model_key="actor",
        models=models_map,
        optimizers=opts_map,
        result_model=result_model,
        config_dict=cfg_dict,
        steps=steps,
        t0=t0,
        ep_idx=ep_idx,
        learning_curve=learning_curve,
        train_diagnostics=diagnostics_fn(),
        eval_episodes=cfg.eval_episodes,
        seed=cfg.seed,
        rng=rng,
        replay_buffer=replay if ctx.save_replay_buffer else None,
    )


@dataclass
class OfflineRunContext:
    """Runtime container holding obs_spec, action_meta, RMS stats, optional eval_env, and checkpointer for offline RL."""

    obs_spec: Any
    action_meta: dict[str, Any]
    obs_rms: DictRunningMeanStd | None
    eval_env: UniversalGymAdapter | None
    owns_eval_env: bool
    ckpt_mgr: CheckpointManager | None
    device: torch.device
    next_eval_step: int = 0
    next_ckpt_step: int = 0
    eval_freq: int = 0
    checkpoint_freq: int = 0


def setup_offline_run(
    dataset: Any,
    *,
    extractor_type: str = "auto",
    normalize_obs: bool = True,
    checkpoint_dir: str | None = None,
    checkpoint_freq: int = 0,
    eval_freq: int = 0,
    save_best_model: bool = True,
    keep_last_k: int = 3,
    tb_log_dir: str | None = None,
    eval_env_or_factory: Any | None = None,
    device: torch.device,
) -> OfflineRunContext:
    """Create OfflineRunContext with inferred/stored ObsSpec, RMS stats, optional eval env, and CheckpointManager."""
    from wannado.drl.core.wrappers import ensure_gym_env

    obs_spec = dataset.resolved_obs_spec
    action_meta = {
        "is_discrete": bool(dataset.is_discrete),
        "raw_action_kind": "discrete" if dataset.is_discrete else "box",
        "action_dim": int(dataset.action_dim),
        "action_nvec": None,
        "action_low": dataset.action_low.tolist() if dataset.action_low is not None else None,
        "action_high": dataset.action_high.tolist() if dataset.action_high is not None else None,
    }
    obs_rms = DictRunningMeanStd(obs_spec) if normalize_obs else None
    eval_env: UniversalGymAdapter | None = None
    owns_eval_env = False
    if eval_env_or_factory is not None:
        owns_eval_env = not isinstance(eval_env_or_factory, UniversalGymAdapter)
        eval_env = ensure_gym_env(
            eval_env_or_factory,
            structured_obs=(obs_spec.kind != "vector"),
            extractor_type=extractor_type,
        )
    use_ckpt = (
        checkpoint_dir is not None
        or tb_log_dir is not None
        or (eval_freq > 0 and eval_env is not None)
        or checkpoint_freq > 0
    )
    ckpt_mgr = (
        CheckpointManager(checkpoint_dir, keep_last_k=keep_last_k, save_best_model=save_best_model, tb_log_dir=tb_log_dir)
        if use_ckpt
        else None
    )
    return OfflineRunContext(
        obs_spec=obs_spec,
        action_meta=action_meta,
        obs_rms=obs_rms,
        eval_env=eval_env,
        owns_eval_env=owns_eval_env,
        ckpt_mgr=ckpt_mgr,
        device=device,
        next_eval_step=int(eval_freq) if eval_freq > 0 else 0,
        next_ckpt_step=int(checkpoint_freq) if checkpoint_freq > 0 else 0,
        eval_freq=int(eval_freq),
        checkpoint_freq=int(checkpoint_freq),
    )


def prepare_offline_run(
    ctx: OfflineRunContext,
    dataset: Any,
    *,
    resume_from: str | None,
    models: dict[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    rng: np.random.Generator,
) -> tuple[Any, Any, int, list[dict[str, float]]]:
    """Restore offline checkpoint if requested, normalize dataset obs/next_obs, and return (obs_all, next_obs_all, start_step, curve)."""
    start_step = 0
    learning_curve: list[dict[str, float]] = []
    if resume_from:
        mgr = ctx.ckpt_mgr if ctx.ckpt_mgr is not None else CheckpointManager()
        info = mgr.load_checkpoint(
            resume_from,
            models=models,
            optimizers=optimizers,
            obs_rms=ctx.obs_rms,
            rng=rng,
            device=ctx.device,
        )
        if ctx.ckpt_mgr is not None and ctx.ckpt_mgr is not mgr:
            ctx.ckpt_mgr.best_eval_mean = mgr.best_eval_mean
            ctx.ckpt_mgr.eval_history = list(mgr.eval_history)
        start_step = int(info["step"])
        learning_curve = list(info.get("extra_state", {}).get("learning_curve", []))
        if ctx.eval_freq > 0:
            while ctx.next_eval_step <= start_step:
                ctx.next_eval_step += ctx.eval_freq
        if ctx.checkpoint_freq > 0:
            while ctx.next_ckpt_step <= start_step:
                ctx.next_ckpt_step += ctx.checkpoint_freq
    elif ctx.obs_rms is not None:
        ctx.obs_rms.update(dataset.obs)
        ctx.obs_rms.freeze(True)

    if ctx.obs_rms is not None:
        ctx.obs_rms.freeze(True)
        obs_all = ctx.obs_rms.normalize(dataset.obs)
        next_obs_all = ctx.obs_rms.normalize(dataset.next_obs)
    else:
        obs_all = dataset.obs
        next_obs_all = dataset.next_obs
    return obs_all, next_obs_all, start_step, learning_curve


def maybe_offline_periodic_hooks(
    ctx: OfflineRunContext,
    step: int,
    *,
    algo_id: str,
    policy_kind: str,
    policy_spec: dict[str, Any],
    primary_model_key: str,
    models: dict[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    config_dict: dict[str, Any],
    eval_episodes: int,
    seed: int,
    learning_curve: list[dict[str, float]],
    rng: np.random.Generator | None = None,
) -> None:
    """Run threshold-scheduled periodic eval & checkpoints for offline RL algorithms."""
    if ctx.ckpt_mgr is None:
        return
    extra_state = {"learning_curve": learning_curve}
    if ctx.eval_freq > 0 and ctx.eval_env is not None and step >= ctx.next_eval_step:
        bundle = PolicyBundle(
            algo_id=algo_id,
            policy_kind=policy_kind,
            model=models[primary_model_key],
            obs_spec=ctx.obs_spec,
            action_meta=ctx.action_meta,
            obs_rms=ctx.obs_rms,
            policy_spec=policy_spec,
            config=config_dict,
            device=ctx.device,
            freeze_obs_rms=True,
        )
        ctx.ckpt_mgr.run_periodic_eval(
            step,
            eval_env=ctx.eval_env,
            policy_fn=bundle,
            obs_rms=ctx.obs_rms,
            num_episodes=eval_episodes,
            seed=seed + 10_000,
            algo_id=algo_id,
            policy_kind=policy_kind,
            policy_spec=policy_spec,
            primary_model_key=primary_model_key,
            models=models,
            optimizers=optimizers,
            obs_spec=ctx.obs_spec,
            action_meta=ctx.action_meta,
            config=config_dict,
            rng=rng,
            extra_state=extra_state,
        )
        while ctx.next_eval_step <= step:
            ctx.next_eval_step += ctx.eval_freq

    if ctx.checkpoint_freq > 0 and ctx.ckpt_mgr.root is not None and step >= ctx.next_ckpt_step:
        ctx.ckpt_mgr.save_checkpoint(
            step,
            algo_id=algo_id,
            policy_kind=policy_kind,
            policy_spec=policy_spec,
            primary_model_key=primary_model_key,
            models=models,
            optimizers=optimizers,
            obs_spec=ctx.obs_spec,
            action_meta=ctx.action_meta,
            config=config_dict,
            obs_rms=ctx.obs_rms,
            rng=rng,
            extra_state=extra_state,
            tag="step",
        )
        while ctx.next_ckpt_step <= step:
            ctx.next_ckpt_step += ctx.checkpoint_freq


def finalize_offline_run(
    ctx: OfflineRunContext,
    dataset: Any,
    *,
    algo_id: str,
    policy_kind: str,
    policy_spec: dict[str, Any],
    primary_model_key: str,
    models: dict[str, Any],
    optimizers: Mapping[str, torch.optim.Optimizer],
    result_model: Any,
    config_dict: dict[str, Any],
    num_grad_steps: int,
    gamma: float,
    t0: float,
    learning_curve: list[dict[str, float]],
    train_diagnostics: dict[str, Any],
    eval_episodes: int,
    seed: int,
    rng: np.random.Generator | None = None,
    explicit_optimizations: list[str] | None = None,
) -> TrainResult:
    """Build PolicyBundle, evaluate via sim or WIS OPE, save checkpoints, and emit TrainResult."""
    from wannado.drl.core.offline_risk import evaluate_offline_wis
    from wannado.drl.mdp.baselines import evaluate_policy_episodes

    if ctx.obs_rms is not None:
        ctx.obs_rms.freeze(True)

    bundle = PolicyBundle(
        algo_id=algo_id,
        policy_kind=policy_kind,
        model=models[primary_model_key],
        obs_spec=ctx.obs_spec,
        action_meta=ctx.action_meta,
        obs_rms=ctx.obs_rms,
        policy_spec=policy_spec,
        config=config_dict,
        device=ctx.device,
        freeze_obs_rms=True,
    )
    eval_stats = (
        evaluate_policy_episodes(ctx.eval_env, bundle, num_episodes=eval_episodes, seed=seed + 10_000)
        if ctx.eval_env is not None
        else None
    )
    ope_stats = evaluate_offline_wis(dataset, bundle, gamma=gamma) if eval_stats is None else None

    best_path: str | None = None
    last_path: str | None = None
    eval_history: list[dict[str, float]] = []
    if ctx.ckpt_mgr is not None:
        extra_state = {"learning_curve": learning_curve}
        if ctx.ckpt_mgr.root is not None:
            saved_last = ctx.ckpt_mgr.save_checkpoint(
                num_grad_steps,
                algo_id=algo_id,
                policy_kind=policy_kind,
                policy_spec=policy_spec,
                primary_model_key=primary_model_key,
                models=models,
                optimizers=optimizers,
                obs_spec=ctx.obs_spec,
                action_meta=ctx.action_meta,
                config=config_dict,
                obs_rms=ctx.obs_rms,
                rng=rng,
                extra_state=extra_state,
                tag="last",
            )
            last_path = str(saved_last) if saved_last is not None else None
            score_metric = eval_stats.mean_return if eval_stats is not None else float((ope_stats or {}).get("wis_return", 0.0))
            if ctx.ckpt_mgr.save_best_model and (ctx.ckpt_mgr.best_model_path is None or score_metric >= ctx.ckpt_mgr.best_eval_mean):
                ctx.ckpt_mgr.best_eval_mean = score_metric
                saved_best = ctx.ckpt_mgr.save_checkpoint(
                    num_grad_steps,
                    algo_id=algo_id,
                    policy_kind=policy_kind,
                    policy_spec=policy_spec,
                    primary_model_key=primary_model_key,
                    models=models,
                    optimizers=optimizers,
                    obs_spec=ctx.obs_spec,
                    action_meta=ctx.action_meta,
                    config=config_dict,
                    obs_rms=ctx.obs_rms,
                    rng=rng,
                    extra_state=extra_state,
                    tag="best",
                )
                best_path = str(saved_best) if saved_best is not None else None
            elif ctx.ckpt_mgr.best_model_path is not None:
                best_path = str(ctx.ckpt_mgr.best_model_path)
        eval_history = list(ctx.ckpt_mgr.eval_history)
        ctx.ckpt_mgr.close()

    if ctx.eval_env is not None and ctx.owns_eval_env and hasattr(ctx.eval_env.raw_env, "close"):
        ctx.eval_env.raw_env.close()

    diag = dict(train_diagnostics)
    if ope_stats is not None:
        diag.update(
            {
                "ope_wis_return": float(ope_stats["wis_return"]),
                "ope_behavior_return": float(ope_stats["behavior_return"]),
                "ope_effective_sample_size": float(ope_stats["effective_sample_size"]),
            }
        )
    opts = resolve_run_optimizations(algo_id, config=config_dict, explicit_ids=explicit_optimizations)
    return TrainResult(
        algo_id=algo_id,
        seed=seed,
        total_timesteps=num_grad_steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=bundle,
        model=result_model,
        config=config_dict,
        learning_curve=learning_curve,
        train_diagnostics=diag,
        eval_stats=eval_stats,
        obs_rms_state=ctx.obs_rms.state_dict() if ctx.obs_rms is not None else None,
        obs_rms_frozen_on_eval=True,
        optimizations_used=[o.id for o in opts],
        eval_history=eval_history,
        best_model_path=best_path,
        last_checkpoint_path=last_path,
        policy_bundle=bundle,
    )

