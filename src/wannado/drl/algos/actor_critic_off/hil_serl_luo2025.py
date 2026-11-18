"""HIL-SERL: Human-in-the-Loop Sample-Efficient RL (Luo, Xu, Wu, & Levine, Science Robotics 2025).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : actor_critic_off / S (Frontier Real-World & Human-Guided Off-Policy SOTA)
- Parent        : `sac` (Haarnoja et al., 2018), incorporating RLPD (Ball et al., ICML 2023)
- Key Diff      :
    1. LayerNorm Twin Critics (`use_layer_norm=True`) to prevent Q-value divergence under
       high Update-to-Data (UTD) ratios and off-policy demo distributions.
    2. 50/50 Symmetric Demo + Online Dual Replay Buffer (`DualDemoReplayBuffer`).
    3. Live Human Intervention Takeover Hook (`intervention_fn`): human corrective overrides
       are injected directly into BOTH the online buffer and the prioritized demo buffer.
- Extension Hook [AI-Prompt Hook]:
    1. Plug in a learned binary success reward classifier in place of raw env rewards.
    2. Tune `utd_ratio` (e.g. 2-10) and `demo_fraction` (default 0.5) for ultra-scarce samples.
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
    finalize_online_run,
    maybe_periodic_hooks,
    maybe_resume_online_run,
    setup_online_run,
)
from wannado.drl.core.buffers import DualDemoReplayBuffer
from wannado.drl.core.networks import SquashedGaussianActor, TwinQCritic
from wannado.drl.core.obs_spec import obs_to_tensor, slice_obs
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import OfflineTransitionDataset

META = AlgoMeta(
    id="hil_serl",
    name="Human-in-the-Loop SERL (HIL-SERL / RLPD)",
    family="actor_critic_off",
    paradigm="off_policy",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("continuous",),
    env_modalities=("continuous_control", "robotics", "custom_sim"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="sac",
    diff_from_parent="Adds LayerNorm Q-critics, 50/50 symmetric offline-demo/online sampling (RLPD), high UTD updates, and live human intervention replay.",
    summary_plain="Ultra-sample-efficient SAC upgraded with prior demonstrations and live human corrective interventions.",
    when_to_use="Complex continuous/manipulation tasks where pure online exploration fails or you have human demos/takeovers.",
    when_not_to_use="Zero online interaction allowed (`cql`/`iql_offline`) or simple fast sims where standard `sac`/`ppo` solves easily.",
    hyperparam_intuition={
        "demo_fraction": "Fraction of each gradient batch drawn from expert demos / human interventions (0.5 canonical RLPD).",
        "utd_ratio": "Gradient steps taken per environment step (higher = faster bootstrapping, stabilized by LayerNorm).",
        "init_alpha": "Initial entropy temperature.",
    },
    extension_hooks=(
        "Supply `intervention_fn(obs, agent_action, info) -> (taken_over, human_action)` for real-time corrections",
        "Attach a learned binary success reward classifier in env_wrapper",
    ),
    module_path="wannado.drl.algos.actor_critic_off.hil_serl_luo2025",
    citation=AlgoCitation(
        key="luo2025hilserl",
        title="Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning",
        authors="Jianlan Luo, Charles Xu, Jeffrey Wu, and Sergey Levine",
        venue="Science Robotics, vol. 10, no. 105, eads5033",
        year=2025,
        url="https://arxiv.org/abs/2410.21845",
        bibtex=(
            "@article{luo2025hilserl,\n"
            "  title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},\n"
            "  author={Luo, Jianlan and Xu, Charles and Wu, Jeffrey and Levine, Sergey},\n"
            "  journal={Science Robotics},\n"
            "  volume={10},\n"
            "  number={105},\n"
            "  pages={eads5033},\n"
            "  year={2025},\n"
            "  url={https://arxiv.org/abs/2410.21845}\n"
            "}"
        ),
        ieee='J. Luo, C. Xu, J. Wu, and S. Levine, "Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning," Science Robotics, vol. 10, no. 105, eads5033, 2025.',
        apa="Luo, J., Xu, C., Wu, J., & Levine, S. (2025). Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning. Science Robotics, 10(105), eads5033.",
    ),
)


@dataclass
class HILSERLConfig:
    total_timesteps: int = 5_000
    learning_starts: int = 100
    utd_ratio: int = 2
    demo_fraction: float = 0.5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    autotune_alpha: bool = True
    buffer_size: int = 50_000
    batch_size: int = 64
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


def train_hil_serl(
    env_or_factory: Any,
    config: HILSERLConfig | None = None,
    *,
    demo_dataset: OfflineTransitionDataset | None = None,
    intervention_fn: Callable[[Any, np.ndarray, dict[str, Any]], tuple[bool, np.ndarray]] | None = None,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train HIL-SERL (RLPD-SAC + LayerNorm Critics + 50/50 Demo/Intervention Replay)."""
    cfg = config or HILSERLConfig()
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
        raise ValueError("hil_serl requires bounded continuous Box action spaces.")

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
        use_layer_norm=True,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    target_critic = TwinQCritic(
        env.obs_spec,
        env.action_dim,
        cfg.hidden_dims,
        use_layer_norm=True,
        extractor_type=cfg.extractor_type,
        extractor_kwargs=cfg.extractor_kwargs,
    ).to(dev)
    hard_update(critic, target_critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), device=dev, requires_grad=cfg.autotune_alpha)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None
    target_entropy = -float(env.action_dim)

    replay = DualDemoReplayBuffer(cfg.buffer_size, env.obs_spec, env.action_dim, discrete_action=False, device=dev)
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
        "log_alpha": log_alpha,
    }
    opts_map: dict[str, torch.optim.Optimizer] = {"actor": actor_opt, "critic": critic_opt}
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
    intervention_count = int(resumed_extra.get("human_interventions", 0))

    # Preload initial offline demonstrations into demo buffer if not already restored from replay sidecar
    if demo_dataset is not None and replay.demo_buffer.size == 0:
        for i in range(len(demo_dataset)):
            o = slice_obs(demo_dataset.obs, i)
            no = slice_obs(demo_dataset.next_obs, i)
            if obs_rms is not None:
                if steps == 0:
                    obs_rms.update(o)
                o = obs_rms.normalize(o)
                no = obs_rms.normalize(no)
            replay.add_demo(o, demo_dataset.actions[i], float(demo_dataset.rewards[i]), no, bool(demo_dataset.terminateds[i]))

    t0 = time.perf_counter()
    obs, _ = env.reset_batch(seed=cfg.seed)
    infos: list[dict[str, Any]] = [{} for _ in range(env.num_envs)]
    q_hist: list[float] = []

    while steps < cfg.total_timesteps:
        if obs_rms is not None:
            obs_rms.update(obs)
            obs_proc = obs_rms.normalize(obs)
        else:
            obs_proc = obs

        if steps < cfg.learning_starts and replay.demo_buffer.size == 0:
            proposed = rng.uniform(env.action_low, env.action_high, size=(env.num_envs, env.action_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = obs_to_tensor(obs_proc, env.obs_spec, dev, add_batch_dim=False)
                a_t, _ = actor.get_action(obs_t)
                proposed = a_t.cpu().numpy().astype(np.float32)

        # [AI-Prompt Hook]: Real-time human intervention takeover sends corrections into BOTH buffers.
        acts = proposed.copy()
        intervened_slots: list[int] = []
        if intervention_fn is not None:
            for i in range(env.num_envs):
                intervened, human_act = intervention_fn(slice_obs(obs, i), proposed[i], infos[i])
                if intervened:
                    acts[i] = np.clip(np.asarray(human_act, dtype=np.float32), env.action_low, env.action_high)
                    intervention_count += 1
                    intervened_slots.append(i)

        sb = env.step_batch(acts)
        valid_count = int(np.sum(sb.valid_transition))
        train_rewards = ret_rms.normalize_batch(sb.rewards, sb.dones) if ret_rms is not None else sb.rewards
        next_proc = obs_rms.normalize(sb.final_obs) if obs_rms is not None else sb.final_obs
        replay.add_online_batch(obs_proc, acts, train_rewards, next_proc, sb.terminateds, valid_mask=sb.valid_transition)
        for i in intervened_slots:
            if bool(sb.valid_transition[i]):
                replay.add_demo(
                    slice_obs(obs_proc, i),
                    acts[i],
                    float(train_rewards[i]),
                    slice_obs(next_proc, i),
                    bool(sb.terminateds[i]),
                )
        steps += valid_count

        for ep_ret, _ in sb.completed_episodes:
            ep_idx += 1
            if ep_idx % 5 == 0 or len(learning_curve) == 0 or steps >= cfg.total_timesteps:
                learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
            if ctx.ckpt_mgr is not None:
                ctx.ckpt_mgr.log_scalars(steps, {"train/episode_return": float(ep_ret)})
        obs = sb.obs
        infos = sb.infos

        total_stored = replay.demo_buffer.size + replay.online_buffer.size
        if steps >= cfg.learning_starts and total_stored >= cfg.batch_size:
            n_updates = max(1, cfg.utd_ratio * max(1, valid_count))
            for u in range(n_updates):
                b_obs, b_act, b_rew, b_next, b_term = replay.sample_symmetric(
                    cfg.batch_size, demo_fraction=cfg.demo_fraction, rng=rng
                )
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    next_a, next_lp = actor.get_action(b_next)
                    tq1, tq2 = target_critic(b_next, next_a)
                    min_tq = torch.min(tq1, tq2) - alpha * next_lp
                    td_target = b_rew.unsqueeze(-1) + cfg.gamma * (1.0 - b_term.unsqueeze(-1)) * min_tq

                q1, q2 = critic(b_obs, b_act)
                critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                q_hist.append(float(q1.mean().item()))
                polyak_update(critic, target_critic, cfg.tau)

                if (u + 1) % cfg.utd_ratio == 0 or u == n_updates - 1:
                    pi_a, log_pi = actor.get_action(b_obs)
                    q1_pi, q2_pi = critic(b_obs, pi_a)
                    actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    if cfg.autotune_alpha and alpha_opt is not None:
                        alpha_loss = (-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()
                        alpha_opt.zero_grad()
                        alpha_loss.backward()
                        alpha_opt.step()

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
            extra_state_fields={"human_interventions": int(intervention_count)},
        )

    return finalize_online_run(
        ctx,
        algo_id=META.id,
        policy_kind="squashed_actor",
        policy_spec=policy_spec,
        primary_model_key="actor",
        models=models_map,
        optimizers=opts_map,
        result_model={"actor": actor, "critic": critic},
        config_dict=cfg_dict,
        steps=steps,
        t0=t0,
        ep_idx=ep_idx,
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
            "max_q_value": float(max(q_hist, key=abs)) if q_hist else 0.0,
            "human_interventions": intervention_count,
            "demo_buffer_size": replay.demo_buffer.size,
            "online_buffer_size": replay.online_buffer.size,
        },
        eval_episodes=cfg.eval_episodes,
        seed=cfg.seed,
        rng=rng,
        replay_buffer=replay if ctx.save_replay_buffer else None,
    )
