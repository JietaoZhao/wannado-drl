"""DDPG: Deep Deterministic Policy Gradient (Lillicrap et al., ICLR 2016).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : actor_critic_off / B (Foundational Continuous Off-Policy Actor-Critic)
- Parent        : `dqn` (Mnih et al., 2015)
- Key Diff      : Extends off-policy replay + target networks from discrete Q-learning
                  to continuous action vectors by learning a deterministic actor mu(s)
                  alongside a differentiable critic Q(s, a) with soft Polyak target updates.
- Extension Hook [AI-Prompt Hook]:
    1. Add Twin Critics, delayed actor update, and target action noise -> `td3`.
    2. Switch deterministic actor to maximum-entropy stochastic policy -> `sac`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import OnlineRunContext, run_offpolicy_continuous_loop
from wannado.drl.core.networks import DeterministicActor, SingleQCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import hard_update, polyak_update

META = AlgoMeta(
    id="ddpg",
    name="Deep Deterministic Policy Gradient (DDPG)",
    family="actor_critic_off",
    paradigm="off_policy",
    tier="B",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("continuous",),
    env_modalities=("continuous_control", "custom_sim"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="dqn",
    diff_from_parent="Replaces discrete argmax Q(s, a) with a continuous deterministic actor mu(s) trained via dQ/da.",
    summary_plain="First deep off-policy actor-critic for continuous control; direct ancestor of TD3 and SAC.",
    when_to_use="Continuous control baselines and comparing improvements brought by TD3/SAC.",
    when_not_to_use="Sensitive continuous tasks prone to Q-overestimation (prefer `td3` or `sac`) or discrete actions (`dqn`/`ppo`).",
    hyperparam_intuition={
        "tau": "Polyak soft target blending speed (0.005 keeps target critics smooth).",
        "exploration_noise": "Gaussian action noise scale added during exploration.",
        "learning_starts": "Random warmup transitions collected before gradient updates begin.",
    },
    extension_hooks=(
        "Add twin critics + target noise + policy delay -> `td3`",
        "Replace deterministic policy with entropy-regularized Gaussian -> `sac`",
    ),
    module_path="wannado.drl.algos.actor_critic_off.ddpg_lillicrap2016",
    citation=AlgoCitation(
        key="lillicrap2016continuous",
        title="Continuous control with deep reinforcement learning",
        authors="Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra",
        venue="International Conference on Learning Representations (ICLR)",
        year=2016,
        url="https://arxiv.org/abs/1509.02971",
        bibtex=(
            "@inproceedings{lillicrap2016continuous,\n"
            "  title={Continuous control with deep reinforcement learning},\n"
            "  author={Lillicrap, Timothy P. and Hunt, Jonathan J. and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2016},\n"
            "  url={https://arxiv.org/abs/1509.02971}\n"
            "}"
        ),
        ieee='T. P. Lillicrap et al., "Continuous control with deep reinforcement learning," in Proc. Int. Conf. Learn. Represent. (ICLR), 2016.',
        apa="Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control with deep reinforcement learning. ICLR.",
    ),
)


@dataclass
class DDPGConfig:
    total_timesteps: int = 8_000
    learning_starts: int = 500
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 50_000
    batch_size: int = 64
    train_freq: int = 1
    gradient_steps: int = 0
    exploration_noise: float = 0.15
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


def train_ddpg(
    env_or_factory: Any,
    config: DDPGConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train DDPG on continuous Box-action environments."""
    cfg = config or DDPGConfig()
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

        critic = SingleQCritic(
            env.obs_spec,
            env.action_dim,
            cfg.hidden_dims,
            action_low=env.action_low,
            action_high=env.action_high,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_critic = SingleQCritic(
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
        act_scale = (env.action_high - env.action_low) / 2.0

        state.update(
            {
                "actor": actor,
                "target_actor": target_actor,
                "critic": critic,
                "target_critic": target_critic,
                "actor_opt": actor_opt,
                "critic_opt": critic_opt,
                "act_scale": act_scale,
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
        noise = rng.normal(0.0, cfg.exploration_noise * state["act_scale"], size=raw.shape).astype(np.float32)
        return np.clip(raw + noise, env.action_low, env.action_high).astype(np.float32)

    def _update(
        _update_step: int,
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

        # [AI-Prompt Hook]: Single Q-target; replace with Twin Clipped Double-Q in `td3`.
        with torch.no_grad():
            next_a = target_actor(b_next)
            next_q = target_critic(b_next, next_a).squeeze(-1)
            td_target = b_rew + cfg.gamma * (1.0 - b_term) * next_q

        q_pred = critic(b_obs, b_act).squeeze(-1)
        critic_loss = F.mse_loss(q_pred, td_target)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        q_hist.append(float(q_pred.mean().item()))

        actor_loss = -critic(b_obs, actor(b_obs)).mean()
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
