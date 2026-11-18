"""REINFORCE: Monte Carlo Policy Gradient with Optional Baseline (Williams, 1992).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / C (Foundational Policy Gradient Baseline)
- Parent        : None (Origin of neural policy-gradient reinforcement learning)
- Plain Intuition:
    Instead of estimating Q-values for every move, REINFORCE adjusts the policy directly:
    after finishing an episode, moves that led to higher-than-average total returns get
    their probabilities boosted, while worse moves get suppressed.
- Extension Hook [AI-Prompt Hook]:
    1. Replace episodic Monte Carlo return with TD Advantage + Critic -> `a2c`.
    2. Constrain KL jump per step -> `trpo` or clipped surrogate -> `ppo`.
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
from wannado.drl.core.networks import ContinuousActorCritic
from wannado.drl.core.obs_spec import obs_to_tensor
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import resolve_device, set_global_seed

META = AlgoMeta(
    id="reinforce",
    name="REINFORCE (Monte Carlo Policy Gradient)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="C",
    on_policy=True,
    sample_efficiency="low",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id=None,
    diff_from_parent="Origin node of neural policy-gradient RL: optimizes log pi(a|s) weighted by episodic returns.",
    summary_plain="Classic policy-gradient baseline that boosts action probabilities along entire finished episodes.",
    when_to_use="Teaching policy gradients, simple episodic tasks, or verifying gradient flow before scaling to PPO.",
    when_not_to_use="Long horizons, noisy rewards, or sample-constrained tasks (high variance; upgrade to `a2c` or `ppo`).",
    hyperparam_intuition={
        "lr": "Policy optimizer step size; too high causes premature deterministic collapse.",
        "gamma": "Discount horizon weighting future rewards.",
        "use_baseline": "Subtracts a learned state-value baseline V(s) to slash gradient variance without adding bias.",
    },
    extension_hooks=(
        "Replace full return G_t with bootstrapped GAE -> `a2c` / `ppo`",
        "Add group-relative reward normalization across K rollouts -> `grpo`",
    ),
    module_path="wannado.drl.algos.policy_gradient.reinforce_williams1992",
    citation=AlgoCitation(
        key="williams1992simple",
        title="Simple statistical gradient-following algorithms for connectionist reinforcement learning",
        authors="Ronald J. Williams",
        venue="Machine Learning, vol. 8, no. 3-4, pp. 229-256",
        year=1992,
        url="https://doi.org/10.1007/BF00992696",
        bibtex=(
            "@article{williams1992simple,\n"
            "  title={Simple statistical gradient-following algorithms for connectionist reinforcement learning},\n"
            "  author={Williams, Ronald J.},\n"
            "  journal={Machine Learning},\n"
            "  volume={8},\n"
            "  number={3-4},\n"
            "  pages={229--256},\n"
            "  year={1992},\n"
            "  doi={10.1007/BF00992696}\n"
            "}"
        ),
        ieee='R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," Machine Learning, vol. 8, no. 3-4, pp. 229-256, 1992.',
        apa="Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4), 229-256.",
    ),
)


@dataclass
class ReinforceConfig:
    total_timesteps: int = 12_000
    lr: float = 1e-3
    gamma: float = 0.99
    ent_coef: float = 0.01
    use_baseline: bool = True
    hidden_dims: tuple[int, ...] = (64, 64)
    normalize_obs: bool = False
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


def train_reinforce(
    env_or_factory: Any,
    config: ReinforceConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train REINFORCE with optional state-value baseline, multimodal encoders, and checkpoints."""
    cfg = config or ReinforceConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    ctx = setup_online_run(
        env_or_factory,
        num_envs=1,
        extractor_type=cfg.extractor_type,
        normalize_obs=cfg.normalize_obs,
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
    env, obs_rms = ctx.env, ctx.obs_rms
    eval_single = env.build_eval_env(env_or_factory) if env._single_adapter is None else env._single_adapter
    net, policy_spec, _, _ = build_actor_critic_for_env(
        env,
        hidden_dims=cfg.hidden_dims,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
        device=dev,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    cfg_dict = asdict(cfg)
    models_map: dict[str, Any] = {"policy": net}
    opts_map = {"policy": optimizer}

    steps, ep_idx, learning_curve = maybe_resume_online_run(
        ctx, resume_from=cfg.resume_from, models=models_map, optimizers=opts_map, rng=rng
    )
    t0 = time.perf_counter()
    entropies: list[float] = []
    saw_action_mask = False

    while steps < cfg.total_timesteps:
        obs, info = eval_single.reset(seed=cfg.seed + ep_idx)
        ep_idx += 1
        ep_obs, ep_acts, ep_rews, ep_masks = [], [], [], []
        done = False
        last_trunc = False
        last_term = False

        while not done and steps < cfg.total_timesteps:
            if obs_rms is not None:
                obs_rms.update(obs)
                obs_proc = obs_rms.normalize(obs)
            else:
                obs_proc = obs

            obs_t = obs_to_tensor(obs_proc, env.obs_spec, dev, add_batch_dim=True)
            mask = info.get("action_mask")
            if mask is not None:
                saw_action_mask = True
            mask_t = torch.as_tensor(mask, device=dev, dtype=torch.bool).unsqueeze(0) if mask is not None else None
            with torch.no_grad():
                if env.is_discrete:
                    act_t, _, _, _ = net.get_action_and_value(obs_t, action_mask=mask_t)
                    act_env: Any = int(act_t.item())
                else:
                    act_t, _, _, _ = net.get_action_and_value(obs_t)
                    act_env = act_t.squeeze(0).cpu().numpy()

            next_obs, reward, term, trunc, info = eval_single.step(act_env)
            ep_obs.append(obs_proc)
            ep_acts.append(act_env)
            ep_rews.append(float(reward))
            if env.is_discrete:
                ep_masks.append(mask if mask is not None else np.ones(env.action_dim, dtype=np.bool_))
            steps += 1
            obs = next_obs
            last_term = bool(term)
            last_trunc = bool(trunc)
            done = bool(term or trunc)

        returns = np.zeros(len(ep_rews), dtype=np.float32)
        g = 0.0
        if cfg.use_baseline and (last_trunc or not last_term) and not last_term:
            tail_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
            with torch.no_grad():
                g = float(net.get_value(obs_to_tensor(tail_proc, env.obs_spec, dev, add_batch_dim=True)).item())
        for i in range(len(ep_rews) - 1, -1, -1):
            g = ep_rews[i] + cfg.gamma * g
            returns[i] = g

        if isinstance(ep_obs[0], dict):
            b_obs: Any = {
                k: obs_to_tensor(np.stack([o[k] for o in ep_obs], axis=0), env.obs_spec.leaves[k], dev)
                for k in env.obs_spec.leaves
                if k in ep_obs[0]
            }
        else:
            b_obs = obs_to_tensor(np.stack(ep_obs, axis=0), env.obs_spec, dev)
        b_ret = torch.as_tensor(returns, device=dev)
        b_act = torch.as_tensor(
            np.asarray(ep_acts, dtype=np.int64 if (env.is_discrete or env.action_nvec is not None) else np.float32), device=dev
        )
        b_mask = torch.as_tensor(np.asarray(ep_masks, dtype=np.bool_), device=dev) if env.is_discrete else None

        if env.is_discrete:
            _, logprob, entropy, values = net.get_action_and_value(b_obs, action=b_act, action_mask=b_mask)
        else:
            _, logprob, entropy, values = net.get_action_and_value(b_obs, action=b_act)

        # [AI-Prompt Hook]: Replace Monte Carlo G_t - V(s) below with GAE or group-normalized rewards.
        if cfg.use_baseline:
            adv = b_ret - values.detach()
            value_loss = 0.5 * F.mse_loss(values, b_ret)
        else:
            adv = b_ret
            value_loss = torch.tensor(0.0, device=dev)

        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(logprob * adv).mean()
        loss = policy_loss + value_loss - cfg.ent_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        entropies.append(float(entropy.mean().item()))

        if ep_idx % 5 == 0 or len(learning_curve) == 0 or steps >= cfg.total_timesteps:
            learning_curve.append({"step": float(steps), "episode_return": float(sum(ep_rews)), "entropy": entropies[-1]})
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
