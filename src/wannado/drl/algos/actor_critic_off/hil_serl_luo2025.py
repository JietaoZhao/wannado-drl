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

from wannado.drl.core.buffers import DualDemoReplayBuffer
from wannado.drl.core.networks import SquashedGaussianActor, TwinQCritic
from wannado.drl.core.provenance import resolve_run_optimizations
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import OfflineTransitionDataset, RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

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
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_hil_serl(
    env_or_factory: Any,
    config: HILSERLConfig | None = None,
    *,
    demo_dataset: OfflineTransitionDataset | None = None,
    intervention_fn: Callable[[np.ndarray, np.ndarray, dict[str, Any]], tuple[bool, np.ndarray]] | None = None,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train HIL-SERL (RLPD-SAC + LayerNorm Critics + 50/50 Demo/Intervention Replay)."""
    cfg = config or HILSERLConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)
    if env.is_discrete or env.action_low is None or env.action_high is None:
        raise ValueError("hil_serl requires bounded continuous Box action spaces.")

    actor = SquashedGaussianActor(env.obs_dim, env.action_dim, env.action_low, env.action_high, cfg.hidden_dims).to(dev)
    critic = TwinQCritic(env.obs_dim, env.action_dim, cfg.hidden_dims, use_layer_norm=True).to(dev)
    target_critic = TwinQCritic(env.obs_dim, env.action_dim, cfg.hidden_dims, use_layer_norm=True).to(dev)
    hard_update(critic, target_critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), device=dev, requires_grad=cfg.autotune_alpha)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None
    target_entropy = -float(env.action_dim)

    replay = DualDemoReplayBuffer(cfg.buffer_size, env.obs_dim, env.action_dim, discrete_action=False, device=dev)
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    # Preload initial offline demonstrations into demo buffer if supplied
    if demo_dataset is not None:
        for i in range(len(demo_dataset)):
            o = demo_dataset.obs[i]
            no = demo_dataset.next_obs[i]
            if obs_rms is not None:
                obs_rms.update(o)
                o = obs_rms.normalize(o)
                no = obs_rms.normalize(no)
            replay.add_demo(o, demo_dataset.actions[i], float(demo_dataset.rewards[i]), no, bool(demo_dataset.terminateds[i]))

    t0 = time.perf_counter()
    obs, info = env.reset(seed=cfg.seed)
    ep_ret, ep_idx = 0.0, 0
    intervention_count = 0
    learning_curve: list[dict[str, float]] = []
    q_hist: list[float] = []

    for step in range(1, cfg.total_timesteps + 1):
        if obs_rms is not None:
            obs_rms.update(obs)
            obs_proc = obs_rms.normalize(obs)
        else:
            obs_proc = obs

        if step < cfg.learning_starts and replay.demo_buffer.size == 0:
            proposed = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        else:
            with torch.no_grad():
                a_t, _ = actor.get_action(torch.as_tensor(obs_proc, device=dev, dtype=torch.float32).unsqueeze(0))
                proposed = a_t.squeeze(0).cpu().numpy()

        # [AI-Prompt Hook]: Real-time human intervention takeover sends corrections into BOTH buffers.
        intervened = False
        act = proposed
        if intervention_fn is not None:
            intervened, human_act = intervention_fn(obs, proposed, info)
            if intervened:
                act = np.clip(np.asarray(human_act, dtype=np.float32), env.action_low, env.action_high)
                intervention_count += 1

        next_obs, reward, term, trunc, next_info = env.step(act)
        ep_ret += float(reward)
        next_proc = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs
        replay.add_online(obs_proc, act, float(reward), next_proc, bool(term))
        if intervened:
            replay.add_demo(obs_proc, act, float(reward), next_proc, bool(term))

        if term or trunc:
            ep_idx += 1
            if ep_idx % 5 == 0 or step == cfg.total_timesteps:
                learning_curve.append({"step": float(step), "episode_return": float(ep_ret)})
            obs, info = env.reset(seed=cfg.seed + ep_idx)
            ep_ret = 0.0
        else:
            obs, info = next_obs, next_info

        total_stored = replay.demo_buffer.size + replay.online_buffer.size
        if step >= cfg.learning_starts and total_stored >= cfg.batch_size:
            for u in range(cfg.utd_ratio):
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

                if u == cfg.utd_ratio - 1:
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

    if obs_rms is not None:
        obs_rms.freeze(True)

    def policy_fn(obs_in: np.ndarray, _info: dict[str, Any] | None = None) -> np.ndarray:
        o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
        with eval_mode(actor), torch.no_grad():
            a_t, _ = actor.get_action(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0), deterministic=True)
            return a_t.squeeze(0).cpu().numpy()

    eval_target = eval_env_or_factory if eval_env_or_factory is not None else env
    eval_stats = evaluate_policy_episodes(eval_target, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)
    cfg_dict = asdict(cfg)
    opts = resolve_run_optimizations(META.id, config=cfg_dict)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.total_timesteps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model={"actor": actor, "critic": critic},
        config=cfg_dict,
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
            "human_interventions": intervention_count,
            "demo_buffer_size": replay.demo_buffer.size,
            "online_buffer_size": replay.online_buffer.size,
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
        optimizations_used=[o.id for o in opts],
    )
