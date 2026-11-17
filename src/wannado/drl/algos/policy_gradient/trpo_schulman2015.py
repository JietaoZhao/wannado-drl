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

from wannado.drl.core.buffers import RolloutBuffer
from wannado.drl.core.networks import ContinuousActorCritic, DiscreteActorCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, explained_variance, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

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
    """Train Trust Region Policy Optimization (TRPO) on Discrete or Continuous envs."""
    cfg = config or TRPOConfig()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)

    net: DiscreteActorCritic | ContinuousActorCritic
    act_dtype: np.dtype[Any]
    if env.is_discrete:
        net = DiscreteActorCritic(env.obs_dim, env.action_dim, hidden_dims=cfg.hidden_dims).to(dev)
        actor_params = list(net.actor.parameters())
        act_shape: tuple[int, ...] = ()
        act_dtype = np.dtype(np.int64)
    else:
        net = ContinuousActorCritic(
            env.obs_dim,
            env.action_dim,
            hidden_dims=cfg.hidden_dims,
            action_low=env.action_low,
            action_high=env.action_high,
        ).to(dev)
        actor_params = list(net.actor_mean.parameters()) + [net.actor_logstd]
        act_shape = (env.action_dim,)
        act_dtype = np.dtype(np.float32)

    critic_opt = torch.optim.Adam(net.critic.parameters(), lr=cfg.vf_lr)
    buffer = RolloutBuffer(
        cfg.n_steps,
        env.obs_dim,
        act_shape,
        action_dtype=act_dtype,
        mask_dim=env.action_dim if env.is_discrete else None,
    )
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    t0 = time.perf_counter()
    obs, info = env.reset(seed=cfg.seed)
    ep_ret, ep_idx, steps = 0.0, 0, 0
    learning_curve: list[dict[str, float]] = []
    kl_hist, ent_hist, ev_hist = [], [], []

    def _get_dist(b_o: torch.Tensor, b_m: torch.Tensor | None) -> Any:
        if env.is_discrete:
            return net.get_dist(b_o, action_mask=b_m)  # type: ignore[call-arg]
        return net.get_dist(b_o)

    while steps < cfg.total_timesteps:
        buffer.reset()
        last_done = False
        for _ in range(cfg.n_steps):
            if steps >= cfg.total_timesteps:
                break
            if obs_rms is not None:
                obs_rms.update(obs)
                obs_proc = obs_rms.normalize(obs)
            else:
                obs_proc = obs

            obs_t = torch.as_tensor(obs_proc, device=dev, dtype=torch.float32).unsqueeze(0)
            mask = info.get("action_mask")
            mask_t = torch.as_tensor(mask, device=dev, dtype=torch.bool).unsqueeze(0) if mask is not None else None
            with torch.no_grad():
                if env.is_discrete:
                    act_t, lp_t, _, val_t = net.get_action_and_value(obs_t, action_mask=mask_t)  # type: ignore[call-arg]
                    act_env: Any = int(act_t.item())
                else:
                    act_t, lp_t, _, val_t = net.get_action_and_value(obs_t)
                    act_env = act_t.squeeze(0).cpu().numpy()

            next_obs, reward, term, trunc, next_info = env.step(act_env)
            ep_ret += float(reward)
            steps += 1
            step_reward = float(reward)
            done_flag = bool(term or trunc)
            if trunc and not term:
                trunc_proc = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs
                with torch.no_grad():
                    trunc_v = float(
                        net.get_value(torch.as_tensor(trunc_proc, device=dev, dtype=torch.float32).unsqueeze(0)).item()
                    )
                step_reward += cfg.gamma * trunc_v

            buffer.add(
                obs_proc,
                act_env,
                float(lp_t.item()),
                step_reward,
                bool(term),
                float(val_t.item()),
                action_mask=mask if mask is not None else (np.ones(env.action_dim, dtype=np.bool_) if env.is_discrete else None),
                done=done_flag,
            )
            last_done = done_flag
            if done_flag:
                ep_idx += 1
                if ep_idx % 5 == 0:
                    learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
                obs, info = env.reset(seed=cfg.seed + ep_idx)
                ep_ret = 0.0
            else:
                obs, info = next_obs, next_info

        n = buffer.ptr
        if n < 2:
            break
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            next_v = float(net.get_value(torch.as_tensor(next_proc, device=dev, dtype=torch.float32).unsqueeze(0)).item())
        buffer.compute_returns_and_advantage(next_v, last_done, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        b_obs = torch.as_tensor(buffer.obs[:n], device=dev)
        b_act = torch.as_tensor(buffer.actions[:n], device=dev)
        b_old_lp = torch.as_tensor(buffer.logprobs[:n], device=dev)
        b_adv = torch.as_tensor(buffer.advantages[:n], device=dev)
        b_ret = torch.as_tensor(buffer.returns[:n], device=dev)
        b_mask = torch.as_tensor(buffer.action_masks[:n], device=dev) if buffer.action_masks is not None else None
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        with torch.no_grad():
            old_dist = _get_dist(b_obs, b_mask)
            if env.is_discrete:
                old_dist = type(old_dist)(logits=old_dist.logits.detach())
            else:
                old_dist = type(old_dist)(old_dist.mean.detach(), old_dist.stddev.detach())

        # [AI-Prompt Hook]: Fisher-vector product + Conjugate Gradient natural policy step.
        def _surrogate(
            obs_b: torch.Tensor = b_obs,
            mask_b: torch.Tensor | None = b_mask,
            act_b: torch.Tensor = b_act,
            old_lp_b: torch.Tensor = b_old_lp,
            adv_b: torch.Tensor = b_adv,
        ) -> torch.Tensor:
            dist = _get_dist(obs_b, mask_b)
            lp = dist.log_prob(act_b) if env.is_discrete else dist.log_prob(act_b).sum(-1)
            return (torch.exp(lp - old_lp_b) * adv_b).mean()

        loss_old = _surrogate()
        grads = torch.autograd.grad(loss_old, actor_params)
        loss_grad = torch.cat([g.reshape(-1) for g in grads]).detach()

        def _fvp(
            v: torch.Tensor,
            obs_b: torch.Tensor = b_obs,
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
            ent_vec = d_now.entropy() if env.is_discrete else d_now.entropy().sum(-1)
            ent_hist.append(float(ent_vec.mean().item()))

        for _ in range(cfg.vf_iters):
            v_pred = net.get_value(b_obs)
            v_loss = 0.5 * F.mse_loss(v_pred, b_ret)
            critic_opt.zero_grad()
            v_loss.backward()
            critic_opt.step()
        ev_hist.append(explained_variance(buffer.values[:n], buffer.returns[:n]))

    if obs_rms is not None:
        obs_rms.freeze(True)

    def policy_fn(obs_in: np.ndarray, info_in: dict[str, Any] | None = None) -> Any:
        o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
        obs_t = torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0)
        with eval_mode(net), torch.no_grad():
            if env.is_discrete:
                m = (info_in or {}).get("action_mask")
                m_t = torch.as_tensor(m, device=dev, dtype=torch.bool).unsqueeze(0) if m is not None else None
                act_t, _, _, _ = net.get_action_and_value(obs_t, action_mask=m_t, deterministic=True)  # type: ignore[call-arg]
                return int(act_t.item())
            act_t, _, _, _ = net.get_action_and_value(obs_t, deterministic=True)
            return act_t.squeeze(0).cpu().numpy()

    eval_target = eval_env_or_factory if eval_env_or_factory is not None else env
    eval_stats = evaluate_policy_episodes(eval_target, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=net,
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_kl": float(np.mean(kl_hist)) if kl_hist else 0.0,
            "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
            "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
            "mean_explained_variance": float(np.mean(ev_hist)) if ev_hist else 0.0,
            "is_continuous": 0.0 if env.is_discrete else 1.0,
            **(
                {"final_logstd_mean": float(net.actor_logstd.detach().mean().item())}
                if isinstance(net, ContinuousActorCritic)
                else {}
            ),
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
    )
