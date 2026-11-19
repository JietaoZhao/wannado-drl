"""TD3: Twin Delayed Deep Deterministic Policy Gradient (Fujimoto, van Hoof, & Meger, ICML 2018).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : actor_critic_off / S (Continuous Deterministic SOTA Benchmark)
- Parent        : `ddpg` (Lillicrap et al., 2016)
- Key Diff      : Fixes DDPG's brittle Q-value explosion via three stabilizing upgrades:
                  1. Clipped Double-Q Learning (min(Q1, Q2) critic target)
                  2. Delayed Policy Updates (update actor every `policy_delay` critic steps)
                  3. Target Policy Smoothing (jitter target actions inside `noise_clip`)
- Extension Hook [AI-Prompt Hook]:
    1. Replace deterministic actor + fixed noise with stochastic max-entropy policy -> `sac`.
    2. Use LayerNorm twin critics + 50/50 demo-online sampling -> `hil_serl`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import OnlineRunContext, run_offpolicy_continuous_loop
from wannado.drl.core.networks import DeterministicActor, TwinQCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import hard_update, polyak_update

META = AlgoMeta(
    id="td3",
    name="Twin Delayed DDPG (TD3)",
    family="actor_critic_off",
    paradigm="off_policy",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("continuous",),
    env_modalities=("continuous_control", "custom_sim"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="ddpg",
    diff_from_parent="Adds Twin Clipped Double-Q critics, delayed actor/target updates, and target action smoothing.",
    summary_plain="Gold-standard deterministic continuous control algorithm; far more stable than DDPG.",
    when_to_use="Sample-efficient continuous control benchmarks where deterministic smooth control is preferred.",
    when_not_to_use="Discrete actions (`dqn`/`ppo`) or multimodal exploration tasks needing entropy maximization (`sac`).",
    hyperparam_intuition={
        "policy_delay": "Number of critic updates per actor update (default 2 lets Q-values settle first).",
        "policy_noise": "Smoothing jitter added to target actions to prevent narrow Q-function spikes.",
        "tau": "Polyak soft update rate for target networks.",
    },
    extension_hooks=(
        "Swap deterministic actor for entropy-tuned squashed Gaussian -> `sac`",
        "Add TD3+BC behavior cloning regularizer for offline continuous control",
    ),
    module_path="wannado.drl.algos.actor_critic_off.td3_fujimoto2018",
    citation=AlgoCitation(
        key="fujimoto2018addressing",
        title="Addressing Function Approximation Error in Actor-Critic Methods",
        authors="Scott Fujimoto, Herke van Hoof, and David Meger",
        venue="Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80",
        year=2018,
        url="https://proceedings.mlr.press/v80/fujimoto18a.html",
        bibtex=(
            "@inproceedings{fujimoto2018addressing,\n"
            "  title={Addressing Function Approximation Error in Actor-Critic Methods},\n"
            "  author={Fujimoto, Scott and van Hoof, Herke and Meger, David},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={80},\n"
            "  year={2018},\n"
            "  url={https://proceedings.mlr.press/v80/fujimoto18a.html}\n"
            "}"
        ),
        ieee='S. Fujimoto, H. van Hoof, and D. Meger, "Addressing function approximation error in actor-critic methods," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018.',
        apa="Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. ICML (PMLR 80).",
    ),
)


@dataclass
class TD3Config:
    total_timesteps: int = 8_000
    learning_starts: int = 500
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.15
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


def train_td3(
    env_or_factory: Any,
    config: TD3Config | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Twin Delayed DDPG (TD3) on bounded continuous Box action spaces."""
    cfg = config or TD3Config()
    state: dict[str, Any] = {}
    q_hist: list[float] = []

    def _build(
        ctx: OnlineRunContext, dev: torch.device
    ) -> tuple[Any, Any, str, dict[str, Any], dict[str, Any], dict[str, torch.optim.Optimizer]]:
        env = ctx.env
        assert env.action_low is not None and env.action_high is not None
        actor = DeterministicActor(
            env.obs_spec,
            env.action_dim,
            env.action_low,
            env.action_high,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_actor = DeterministicActor(
            env.obs_spec,
            env.action_dim,
            env.action_low,
            env.action_high,
            cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        hard_update(actor, target_actor)

        critic = TwinQCritic(
            env.obs_spec,
            env.action_dim,
            cfg.hidden_dims,
            action_low=env.action_low,
            action_high=env.action_high,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_critic = TwinQCritic(
            env.obs_spec,
            env.action_dim,
            cfg.hidden_dims,
            action_low=env.action_low,
            action_high=env.action_high,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        hard_update(critic, target_critic)

        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        act_low_t = torch.as_tensor(env.action_low, device=dev, dtype=torch.float32)
        act_high_t = torch.as_tensor(env.action_high, device=dev, dtype=torch.float32)
        act_scale_t = (act_high_t - act_low_t) / 2.0
        act_scale_np = (env.action_high - env.action_low) / 2.0

        state.update(
            {
                "actor": actor,
                "target_actor": target_actor,
                "critic": critic,
                "target_critic": target_critic,
                "actor_opt": actor_opt,
                "critic_opt": critic_opt,
                "act_low_t": act_low_t,
                "act_high_t": act_high_t,
                "act_scale_t": act_scale_t,
                "act_scale_np": act_scale_np,
            }
        )
        policy_spec = {
            "net_class": "DeterministicActor",
            "init_kwargs": {
                "action_dim": int(env.action_dim),
                "action_low": env.action_low.tolist(),
                "action_high": env.action_high.tolist(),
                "hidden_dims": list(cfg.hidden_dims),
                "extractor_type": cfg.extractor_type,
                "extractor_kwargs": cfg.extractor_kwargs,
            },
        }
        models_map = {
            "actor": actor,
            "target_actor": target_actor,
            "critic": critic,
            "target_critic": target_critic,
        }
        opts_map: dict[str, torch.optim.Optimizer] = {"actor": actor_opt, "critic": critic_opt}
        return actor, {"actor": actor, "critic": critic}, "deterministic_actor", policy_spec, models_map, opts_map

    def _sample(actor: Any, obs_t: Any, env: Any, rng: np.random.Generator) -> np.ndarray:
        raw = actor(obs_t).cpu().numpy()
        noise = rng.normal(0.0, cfg.exploration_noise * state["act_scale_np"], size=raw.shape).astype(np.float32)
        return np.clip(raw + noise, env.action_low, env.action_high).astype(np.float32)

    def _update(
        update_step: int,
        b_obs: Any,
        b_act: torch.Tensor,
        b_rew: torch.Tensor,
        b_next: Any,
        b_term: torch.Tensor,
    ) -> None:
        actor = state["actor"]
        target_actor = state["target_actor"]
        critic = state["critic"]
        target_critic = state["target_critic"]
        actor_opt = state["actor_opt"]
        critic_opt = state["critic_opt"]
        act_low_t = state["act_low_t"]
        act_high_t = state["act_high_t"]
        act_scale_t = state["act_scale_t"]

        # [AI-Prompt Hook]: TD3 Clipped Double-Q + target policy smoothing.
        with torch.no_grad():
            raw_noise = torch.randn_like(b_act) * (cfg.policy_noise * act_scale_t)
            clipped_noise = torch.clamp(raw_noise, -cfg.noise_clip * act_scale_t, cfg.noise_clip * act_scale_t)
            next_a = torch.clamp(target_actor(b_next) + clipped_noise, act_low_t, act_high_t)
            tq1, tq2 = target_critic(b_next, next_a)
            min_tq = torch.min(tq1, tq2).squeeze(-1)
            td_target = b_rew + cfg.gamma * (1.0 - b_term) * min_tq

        q1, q2 = critic(b_obs, b_act)
        critic_loss = F.mse_loss(q1.squeeze(-1), td_target) + F.mse_loss(q2.squeeze(-1), td_target)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        q_hist.append(float(q1.mean().item()))

        if update_step % cfg.policy_delay == 0:
            actor_loss = -critic.q1_forward(b_obs, actor(b_obs)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            polyak_update(actor, target_actor, cfg.tau)
            polyak_update(critic, target_critic, cfg.tau)

    def _diag() -> dict[str, Any]:
        return {
            "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
            "max_q_value": float(max(q_hist, key=abs)) if q_hist else 0.0,
        }

    return run_offpolicy_continuous_loop(
        env_or_factory,
        algo_id=META.id,
        cfg=cfg,
        eval_env_or_factory=eval_env_or_factory,
        build_models_fn=_build,
        sample_action_fn=_sample,
        update_fn=_update,
        diagnostics_fn=_diag,
    )
