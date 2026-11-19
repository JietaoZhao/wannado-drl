"""PPO-GRU: Recurrent Proximal Policy Optimization for POMDPs (Schulman et al., 2017).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / A (POMDP & Partial Observability Extension)
- Parent        : `ppo` (Schulman et al., 2017)
- Key Diff      : Inserts a GRU memory cell between encoder and Actor/Critic heads,
                  resetting hidden states cleanly at episode boundaries so the agent can
                  infer hidden velocity, queue state, or latent channel dynamics over time.
- Extension Hook [AI-Prompt Hook]:
    1. Replace GRU with Causal Transformer / Attention block for longer context windows.
    2. Add auxiliary state-reconstruction loss on hidden state h_t.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from wannado.drl.core.algo_helpers import (
    finalize_online_run,
    maybe_periodic_hooks,
    maybe_resume_online_run,
    setup_online_run,
)
from wannado.drl.core.buffers import RolloutBuffer
from wannado.drl.core.networks import MultiCategorical, RecurrentActorCriticGRU
from wannado.drl.core.obs_spec import obs_to_tensor, slice_obs
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import explained_variance, resolve_device, set_global_seed

META = AlgoMeta(
    id="ppo_gru",
    name="Recurrent PPO (PPO-GRU for POMDP)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="A",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("pomdp", "discrete_control", "continuous_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=True,
    parent_id="ppo",
    diff_from_parent="Adds a GRU memory cell unrolled sequentially with episode-boundary hidden state resets for POMDPs.",
    summary_plain="PPO with short-term working memory (GRU) so agents infer hidden dynamics like velocity or noisy channels.",
    when_to_use="Stage 0 identifies partial observability (missing velocity, occluded sensors, delayed feedback).",
    when_not_to_use="Fully Markovian environments where single-frame observations already contain complete state (use `ppo`).",
    hyperparam_intuition={
        "n_steps": "Sequence rollout chunk length for truncated backpropagation through time.",
        "hidden_dim": "GRU memory capacity (64-128 for compact control/wireless tasks).",
        "clip_coef": "PPO trust-region clip width.",
    },
    extension_hooks=(
        "Swap GRU cell for causal Transformer block",
        "Add belief-state prediction loss to supervise latent GRU state h_t",
    ),
    module_path="wannado.drl.algos.policy_gradient.ppo_gru_schulman2017",
    citation=AlgoCitation(
        key="schulman2017proximal_gru",
        title="Proximal Policy Optimization Algorithms",
        authors="John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov",
        venue="arXiv preprint arXiv:1707.06347 (Recurrent GRU POMDP variant)",
        year=2017,
        url="https://arxiv.org/abs/1707.06347",
        bibtex=(
            "@article{schulman2017proximal_gru,\n"
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
class PPOGRUConfig:
    total_timesteps: int = 16_000
    n_steps: int = 128
    update_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
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


def train_ppo_gru(
    env_or_factory: Any,
    config: PPOGRUConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Recurrent PPO-GRU with vectorized POMDP rollouts, multimodal encoders, and checkpoints."""
    cfg = config or PPOGRUConfig()
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
    net = RecurrentActorCriticGRU(
        env.obs_spec,
        env.action_dim,
        hidden_dim=cfg.hidden_dim,
        is_discrete=env.is_discrete,
        action_low=env.action_low,
        action_high=env.action_high,
        action_nvec=env.action_nvec if not env.is_discrete else None,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    policy_spec = {
        "net_class": "RecurrentActorCriticGRU",
        "init_kwargs": {
            "action_dim": int(env.action_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "is_discrete": bool(env.is_discrete),
            "action_nvec": [int(v) for v in env.action_nvec] if (env.action_nvec is not None and not env.is_discrete) else None,
            "extractor_type": cfg.extractor_type,
            "extractor_kwargs": cfg.extractor_kwargs,
        },
    }
    act_shape: tuple[int, ...] = () if env.is_discrete else ((len(env.action_nvec),) if env.action_nvec is not None else (int(env.action_dim),))
    act_dtype = np.dtype(np.int64 if (env.is_discrete or env.action_nvec is not None) else np.float32)

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
    h_cur = torch.zeros(1, env.num_envs, cfg.hidden_dim, device=dev)
    saw_action_mask = mask_batch is not None
    ent_hist, ev_hist = [], []

    while steps < cfg.total_timesteps:
        buffer.reset()
        h_start = h_cur.detach().clone()
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
                act_t, lp_t, _, val_t, h_cur = net.step_action_value(obs_t, h_cur, action_mask=mask_t)
                act_env = act_t.cpu().numpy()

            sb = env.step_batch(act_env)
            valid_count = int(np.sum(sb.valid_transition))
            step_rewards = ret_rms.normalize_batch(sb.rewards, sb.dones) if ret_rms is not None else sb.rewards.copy()
            trunc_mask = sb.valid_transition & sb.truncateds & (~sb.terminateds)
            if np.any(trunc_mask):
                trunc_raw = slice_obs(sb.final_obs, trunc_mask)
                trunc_proc = obs_rms.normalize(trunc_raw) if obs_rms is not None else trunc_raw
                h_trunc = h_cur[:, torch.as_tensor(trunc_mask, device=dev, dtype=torch.bool), :]
                with torch.no_grad():
                    _, _, _, trunc_nv_t, _ = net.step_action_value(
                        obs_to_tensor(trunc_proc, env.obs_spec, dev), h_trunc
                    )
                step_rewards[trunc_mask] += cfg.gamma * trunc_nv_t.cpu().numpy()

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
            if np.any(sb.dones):
                done_m = torch.as_tensor(sb.dones, device=dev, dtype=torch.bool)
                h_cur[:, done_m, :] = 0.0
            steps += valid_count
            last_dones = sb.dones
            for ep_ret, _ in sb.completed_episodes:
                ep_idx += 1
                if ep_idx % 5 == 0 or len(learning_curve) == 0 or steps >= cfg.total_timesteps:
                    learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
                if ctx.ckpt_mgr is not None:
                    ctx.ckpt_mgr.log_scalars(steps, {"train/episode_return": float(ep_ret)})
            obs, mask_batch = sb.obs, sb.action_masks

        n = buffer.ptr
        if n < 2:
            break
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            _, _, _, nv_t, _ = net.step_action_value(obs_to_tensor(next_proc, env.obs_spec, dev), h_cur)
        buffer.compute_returns_and_advantage(nv_t.cpu().numpy(), last_dones, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        b_obs_raw = slice_obs(buffer.obs, slice(0, n))
        if isinstance(b_obs_raw, dict):
            b_obs: Any = {
                k: obs_to_tensor(v.reshape(n * env.num_envs, *v.shape[2:]), env.obs_spec.leaves[k], dev)
                for k, v in b_obs_raw.items()
            }
            # Restore (T, N, ...) leading dims per dict leaf
            for k in b_obs:
                t_val = b_obs[k]
                b_obs[k] = t_val.reshape(n, env.num_envs, *t_val.shape[1:])
        else:
            flat_t = obs_to_tensor(b_obs_raw.reshape(n * env.num_envs, *b_obs_raw.shape[2:]), env.obs_spec, dev)
            b_obs = flat_t.reshape(n, env.num_envs, *flat_t.shape[1:])

        b_done = torch.as_tensor(buffer.dones[:n], device=dev, dtype=torch.float32)
        b_valid = torch.as_tensor(buffer.valid_masks[:n], device=dev, dtype=torch.float32)
        b_act = torch.as_tensor(buffer.actions[:n], device=dev)
        b_old_lp = torch.as_tensor(buffer.logprobs[:n], device=dev, dtype=torch.float32)
        b_adv = torch.as_tensor(buffer.advantages[:n], device=dev, dtype=torch.float32)
        b_ret = torch.as_tensor(buffer.returns[:n], device=dev, dtype=torch.float32)
        b_mask = torch.as_tensor(buffer.action_masks[:n], device=dev, dtype=torch.bool) if buffer.action_masks is not None else None

        valid_bool = b_valid > 0.5
        if bool(valid_bool.sum().item()) > 1:
            adv_valid = b_adv[valid_bool]
            b_adv = (b_adv - adv_valid.mean()) / (adv_valid.std() + 1e-8)

        # [AI-Prompt Hook]: Unroll full (T, B) sequence through GRU with done_seq episode-boundary resets.
        for _ in range(cfg.update_epochs):
            feats, _ = net.forward_seq(b_obs, h_start, done_seq=b_done)
            new_val = net.critic(feats).squeeze(-1)
            if env.is_discrete:
                logits = net.actor(feats)
                if b_mask is not None:
                    logits = logits.masked_fill(~b_mask, -1e8)
                dist: Any = Categorical(logits=logits)
                new_lp = dist.log_prob(b_act)
                entropy = dist.entropy()
            elif net.action_nvec is not None:
                logits = net.actor(feats)
                if b_mask is not None and b_mask.shape[-1] == logits.shape[-1]:
                    logits = logits.masked_fill(~b_mask, -1e8)
                dist = MultiCategorical(torch.split(logits, list(net.action_nvec), dim=-1))
                new_lp = dist.log_prob(b_act)
                entropy = dist.entropy()
            else:
                mean = net.action_bias + net.action_scale * net.actor(feats)
                std = net.action_scale * torch.exp(net.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
                dist = Normal(mean, std)
                new_lp = dist.log_prob(b_act).sum(-1)
                entropy = dist.entropy().sum(-1)

            ratio = (new_lp - b_old_lp).exp()
            pg_obj = torch.max(-b_adv * ratio, -b_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef))
            v_err = 0.5 * F.mse_loss(new_val, b_ret, reduction="none")
            denom = b_valid.sum().clamp_min(1.0)
            loss = ((pg_obj + cfg.vf_coef * v_err - cfg.ent_coef * entropy) * b_valid).sum() / denom

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            ent_hist.append(float(((entropy * b_valid).sum() / denom).item()))

        valid_np = buffer.valid_masks[:n].reshape(-1) > 0.5
        if np.any(valid_np):
            ev_hist.append(
                explained_variance(
                    buffer.values[:n].reshape(-1)[valid_np],
                    buffer.returns[:n].reshape(-1)[valid_np],
                )
            )
        maybe_periodic_hooks(
            ctx,
            steps,
            algo_id=META.id,
            policy_kind="gru_actor_critic",
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
        "mean_explained_variance": float(np.mean(ev_hist)) if ev_hist else 0.0,
        "is_continuous": 0.0 if (env.is_discrete or env.action_nvec is not None) else 1.0,
    }
    if hasattr(net, "actor_logstd"):
        diagnostics["final_logstd_mean"] = float(net.actor_logstd.detach().mean().item())

    return finalize_online_run(
        ctx,
        algo_id=META.id,
        policy_kind="gru_actor_critic",
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
