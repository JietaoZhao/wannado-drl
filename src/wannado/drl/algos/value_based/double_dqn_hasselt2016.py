"""Double DQN: Eliminating Overestimation Bias (van Hasselt, Guez, & Silver, AAAI 2016).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : value_based / A (Fixes DQN maximization bias)
- Parent        : `dqn` (Mnih et al., 2015)
- Key Diff      : Selects the greedy next action with the online Q-network, then
                  evaluates that chosen action using the frozen target Q-network:
                  target = r + gamma * Q_target(s', argmax_{a'} Q_online(s', a')).
- Extension Hook [AI-Prompt Hook]:
    1. Replace QNetwork architecture with V(s) + centered A(s, a) -> `dueling_dqn`.
    2. Use Polyak soft target updates (tau) instead of hard periodic copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import OnlineRunContext, run_value_based_loop
from wannado.drl.core.networks import QNetwork
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult

META = AlgoMeta(
    id="double_dqn",
    name="Double Deep Q-Network (Double DQN)",
    family="value_based",
    paradigm="off_policy",
    tier="A",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("discrete",),
    env_modalities=("discrete_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="dqn",
    diff_from_parent="Decouples next-action argmax selection (online net) from Q-value evaluation (target net).",
    summary_plain="Fixes DQN's over-optimistic Q-value inflation with a one-line target decoupling change.",
    when_to_use="Drop-in upgrade over vanilla DQN whenever discrete Q-values overshoot or fluctuate.",
    when_not_to_use="Continuous action tasks (use `td3` or `sac`) or tasks needing state-value separation (`dueling_dqn`).",
    hyperparam_intuition={
        "lr": "Optimizer learning rate for Q-network updates.",
        "target_update_freq": "Target sync interval; larger values stabilize Bellman targets.",
        "gamma": "Discount horizon weighting future vs immediate rewards.",
    },
    extension_hooks=(
        "Swap QNetwork to DuelingQNetwork -> `dueling_dqn`",
        "Add n-step TD returns inside replay buffer",
    ),
    module_path="wannado.drl.algos.value_based.double_dqn_hasselt2016",
    citation=AlgoCitation(
        key="vanhasselt2016deep",
        title="Deep Reinforcement Learning with Double Q-learning",
        authors="Hado van Hasselt, Arthur Guez, and David Silver",
        venue="Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)",
        year=2016,
        url="https://arxiv.org/abs/1509.06461",
        bibtex=(
            "@inproceedings{vanhasselt2016deep,\n"
            "  title={Deep Reinforcement Learning with Double Q-learning},\n"
            "  author={van Hasselt, Hado and Guez, Arthur and Silver, David},\n"
            "  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},\n"
            "  year={2016},\n"
            "  url={https://arxiv.org/abs/1509.06461}\n"
            "}"
        ),
        ieee='H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in Proc. AAAI Conf. Artif. Intell., 2016.',
        apa="van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. Proceedings of the AAAI Conference on Artificial Intelligence.",
    ),
)


@dataclass
class DoubleDQNConfig:
    total_timesteps: int = 12_000
    learning_starts: int = 400
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 20_000
    batch_size: int = 64
    target_update_freq: int = 200
    train_freq: int = 1
    gradient_steps: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.5
    hidden_dims: tuple[int, ...] = (128, 128)
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
    save_replay_buffer: bool = False
    resume_from: str | None = None
    tb_log_dir: str | None = None
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_double_dqn(
    env_or_factory: Any,
    config: DoubleDQNConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Double DQN with VectorEnv, CNN/Dict/Attention encoders, and checkpoints."""
    cfg = config or DoubleDQNConfig()

    def _build_nets(ctx: OnlineRunContext, dev: torch.device) -> tuple[QNetwork, QNetwork, dict[str, Any]]:
        spec = {
            "net_class": "QNetwork",
            "init_kwargs": {
                "action_dim": int(ctx.env.action_dim),
                "hidden_dims": list(cfg.hidden_dims),
                "extractor_type": cfg.extractor_type,
                "extractor_kwargs": cfg.extractor_kwargs,
            },
        }
        q_net = QNetwork(
            ctx.env.obs_spec,
            ctx.env.action_dim,
            hidden_dims=cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_net = QNetwork(
            ctx.env.obs_spec,
            ctx.env.action_dim,
            hidden_dims=cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        return q_net, target_net, spec

    def _double_dqn_loss(
        q_net: QNetwork,
        target_net: QNetwork,
        b_obs: Any,
        b_act: torch.Tensor,
        b_rew: torch.Tensor,
        b_next: Any,
        b_term: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        # [AI-Prompt Hook]: Double Q-learning target decouples action choice (online) from valuation (target).
        with torch.no_grad():
            best_next_act = q_net(b_next).argmax(dim=1, keepdim=True)
            next_q = target_net(b_next).gather(1, best_next_act).squeeze(1)
            td_target = b_rew + cfg.gamma * (1.0 - b_term) * next_q
        q_pred = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
        return F.smooth_l1_loss(q_pred, td_target), float(q_pred.mean().item())

    return run_value_based_loop(
        env_or_factory,
        algo_id=META.id,
        cfg=cfg,
        eval_env_or_factory=eval_env_or_factory,
        build_q_nets=_build_nets,
        loss_fn=_double_dqn_loss,
        extra_diagnostics_fn=lambda _losses, qs: {"max_q_value": float(np.max(qs)) if qs else 0.0},
    )
