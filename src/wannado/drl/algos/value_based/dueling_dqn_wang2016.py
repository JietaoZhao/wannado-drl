"""Dueling Double DQN: State-Value & Advantage Decomposition (Wang et al., ICML 2016).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : value_based / A (SOTA discrete architecture before Rainbow)
- Parent        : `double_dqn` (van Hasselt et al., 2016)
- Key Diff      : Splits the network head into a scalar state-value V(s) and action
                  advantages A(s, a), combined as Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)).
                  Useful when many actions yield nearly identical outcomes in a state.
- Extension Hook [AI-Prompt Hook]:
    1. Replace uniform sampling with Prioritized Experience Replay (PER, Schaul 2016).
    2. Combine with distributional return atoms -> Rainbow-lite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import OnlineRunContext, run_value_based_loop
from wannado.drl.core.networks import DuelingQNetwork
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult

META = AlgoMeta(
    id="dueling_dqn",
    name="Dueling Double DQN",
    family="value_based",
    paradigm="off_policy",
    tier="A",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("discrete",),
    env_modalities=("discrete_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="double_dqn",
    diff_from_parent="Decomposes Q(s,a) into V(s) + (A(s,a) - mean A(s,.)) alongside Double-DQN targets.",
    summary_plain="Separates 'how good is this situation' from 'which action is better', speeding up multi-action discrete learning.",
    when_to_use="Discrete problems with many actions where individual action choices only matter in critical states.",
    when_not_to_use="Continuous action tasks (`sac`/`td3`) or tiny 2-action problems where standard Double DQN already suffices.",
    hyperparam_intuition={
        "hidden_dim": "Width of shared state trunk and V/A branches.",
        "target_update_freq": "Target sync interval stabilizing dueling advantage separation.",
        "lr": "Adam step size.",
    },
    extension_hooks=(
        "Add Prioritized Experience Replay weights w_i = (N * P(i))^{-beta}",
        "Replace MSE/Huber target with categorical distribution -> `c51`",
    ),
    module_path="wannado.drl.algos.value_based.dueling_dqn_wang2016",
    citation=AlgoCitation(
        key="wang2016dueling",
        title="Dueling Network Architectures for Deep Reinforcement Learning",
        authors="Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas",
        venue="Proceedings of the 33rd International Conference on Machine Learning (ICML), PMLR 48",
        year=2016,
        url="https://proceedings.mlr.press/v48/wangf16.html",
        bibtex=(
            "@inproceedings{wang2016dueling,\n"
            "  title={Dueling Network Architectures for Deep Reinforcement Learning},\n"
            "  author={Wang, Ziyu and Schaul, Tom and Hessel, Matteo and van Hasselt, Hado and Lanctot, Marc and de Freitas, Nando},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={48},\n"
            "  year={2016},\n"
            "  url={https://proceedings.mlr.press/v48/wangf16.html}\n"
            "}"
        ),
        ieee='Z. Wang et al., "Dueling network architectures for deep reinforcement learning," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 48, 2016.',
        apa="Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. ICML (PMLR 48).",
    ),
)


@dataclass
class DuelingDQNConfig:
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
    hidden_dim: int = 128
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


def train_dueling_dqn(
    env_or_factory: Any,
    config: DuelingDQNConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Dueling Double DQN with VectorEnv, CNN/Dict/Attention encoders, and checkpoints."""
    cfg = config or DuelingDQNConfig()

    def _build_nets(ctx: OnlineRunContext, dev: torch.device) -> tuple[DuelingQNetwork, DuelingQNetwork, dict[str, Any]]:
        spec = {
            "net_class": "DuelingQNetwork",
            "init_kwargs": {
                "action_dim": int(ctx.env.action_dim),
                "hidden_dim": int(cfg.hidden_dim),
                "extractor_type": cfg.extractor_type,
                "extractor_kwargs": cfg.extractor_kwargs,
            },
        }
        q_net = DuelingQNetwork(
            ctx.env.obs_spec,
            ctx.env.action_dim,
            hidden_dim=cfg.hidden_dim,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_net = DuelingQNetwork(
            ctx.env.obs_spec,
            ctx.env.action_dim,
            hidden_dim=cfg.hidden_dim,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        return q_net, target_net, spec

    def _dueling_loss(
        q_net: DuelingQNetwork,
        target_net: DuelingQNetwork,
        b_obs: Any,
        b_act: torch.Tensor,
        b_rew: torch.Tensor,
        b_next: Any,
        b_term: torch.Tensor,
        b_next_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        # [AI-Prompt Hook]: Combine Dueling V+A head with Double-DQN target decoupling.
        with torch.no_grad():
            online_next_q = q_net(b_next)
            if b_next_mask is not None:
                valid_row = b_next_mask.any(dim=1, keepdim=True)
                safe_mask = torch.where(valid_row, b_next_mask, torch.ones_like(b_next_mask))
                online_next_q = online_next_q.masked_fill(~safe_mask, -1e9)
            best_next_act = online_next_q.argmax(dim=1, keepdim=True)
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
        loss_fn=_dueling_loss,
        extra_diagnostics_fn=lambda _losses, qs: {"max_q_value": float(np.max(qs)) if qs else 0.0},
    )
