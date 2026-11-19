"""SAC: Soft Actor-Critic with Automatic Entropy Tuning (Haarnoja et al., ICML 2018).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : actor_critic_off / S (Sample-Efficient Continuous Control Default)
- Parent        : `td3` (Fujimoto et al., 2018)
- Key Diff      : Replaces TD3's deterministic actor + manual Gaussian exploration noise
                  with a squashed stochastic Gaussian policy trained under maximum-entropy
                  RL, automatically tuning temperature alpha toward target entropy -dim(A).
- Extension Hook [AI-Prompt Hook]:
    1. Add LayerNorm Q-critics + 50/50 demo/intervention replay -> `hil_serl`.
    2. Penalize out-of-distribution actions in Q-loss for static datasets -> `cql`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import OnlineRunContext, run_offpolicy_continuous_loop
from wannado.drl.core.networks import SquashedGaussianActor, TwinQCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import hard_update, polyak_update

META = AlgoMeta(
    id="sac",
    name="Soft Actor-Critic (SAC)",
    family="actor_critic_off",
    paradigm="off_policy",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("continuous",),
    env_modalities=("continuous_control", "custom_sim"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="td3",
    diff_from_parent="Replaces deterministic actor + fixed exploration noise with maximum-entropy squashed Gaussian policy and auto-tuned alpha.",
    summary_plain="Go-to sample-efficient algorithm for continuous control; balances reward and automatic exploration curiosity.",
    when_to_use="Continuous Box actions when simulator steps are costly and you want strong sample efficiency + smooth exploration.",
    when_not_to_use="Discrete actions (use `ppo` or `dqn`) or zero-interaction static logs (use `cql` / `iql_offline`).",
    hyperparam_intuition={
        "init_alpha": "Starting entropy temperature; automatically tuned when autotune_alpha=True.",
        "tau": "Target Q-critic Polyak smoothing factor.",
        "learning_starts": "Initial random steps seeding replay buffer.",
    },
    extension_hooks=(
        "Add LayerNorm in critics + 50/50 demo/intervention replay -> `hil_serl`",
        "Add conservative log-sum-exp Q penalty on static data -> `cql`",
    ),
    module_path="wannado.drl.algos.actor_critic_off.sac_haarnoja2018",
    citation=AlgoCitation(
        key="haarnoja2018soft",
        title="Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
        authors="Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine",
        venue="Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80",
        year=2018,
        url="https://proceedings.mlr.press/v80/haarnoja18a.html",
        bibtex=(
            "@inproceedings{haarnoja2018soft,\n"
            "  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},\n"
            "  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={80},\n"
            "  year={2018},\n"
            "  url={https://proceedings.mlr.press/v80/haarnoja18a.html}\n"
            "}"
        ),
        ieee='T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018.',
        apa="Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. ICML (PMLR 80).",
    ),
)


@dataclass
class SACConfig:
    total_timesteps: int = 8_000
    learning_starts: int = 400
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    autotune_alpha: bool = True
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


def train_sac(
    env_or_factory: Any,
    config: SACConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Soft Actor-Critic (SAC) on bounded continuous Box action spaces."""
    cfg = config or SACConfig()
    state: dict[str, Any] = {}
    q_hist: list[float] = []
    ent_hist: list[float] = []

    def _build(
        ctx: OnlineRunContext, dev: torch.device
    ) -> tuple[Any, Any, str, dict[str, Any], dict[str, Any], dict[str, torch.optim.Optimizer]]:
        env = ctx.env
        assert env.action_low is not None and env.action_high is not None
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
        log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), device=dev, requires_grad=cfg.autotune_alpha)
        alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr) if cfg.autotune_alpha else None
        act_scale = np.maximum((env.action_high - env.action_low) / 2.0, 1e-6)
        target_entropy = -float(env.action_dim) + float(np.sum(np.log(act_scale)))

        state.update(
            {
                "actor": actor,
                "critic": critic,
                "target_critic": target_critic,
                "actor_opt": actor_opt,
                "critic_opt": critic_opt,
                "log_alpha": log_alpha,
                "alpha_opt": alpha_opt,
                "target_entropy": target_entropy,
            }
        )
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
        return actor, {"actor": actor, "critic": critic}, "squashed_actor", policy_spec, models_map, opts_map

    def _sample(actor: Any, obs_t: Any, _env: Any, _rng: np.random.Generator) -> np.ndarray:
        a_t, _ = actor.get_action(obs_t)
        return a_t.cpu().numpy().astype(np.float32)

    def _update(
        _update_step: int,
        b_obs: Any,
        b_act: torch.Tensor,
        b_rew: torch.Tensor,
        b_next: Any,
        b_term: torch.Tensor,
    ) -> None:
        actor = state["actor"]
        critic = state["critic"]
        target_critic = state["target_critic"]
        actor_opt = state["actor_opt"]
        critic_opt = state["critic_opt"]
        log_alpha = state["log_alpha"]
        alpha_opt = state["alpha_opt"]
        target_entropy = state["target_entropy"]

        alpha = log_alpha.exp().detach()
        # [AI-Prompt Hook]: Soft Bellman target subtracting alpha * log pi(a'|s').
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

        pi_a, log_pi = actor.get_action(b_obs)
        q1_pi, q2_pi = critic(b_obs, pi_a)
        actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
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

    def _diag() -> dict[str, Any]:
        log_alpha = state["log_alpha"]
        return {
            "mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0,
            "max_q_value": float(max(q_hist, key=abs)) if q_hist else 0.0,
            "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
            "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
            "final_alpha": float(log_alpha.exp().item()),
            "is_continuous": 1.0,
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
