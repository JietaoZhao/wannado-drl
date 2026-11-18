"""DQN: Deep Q-Network (Mnih et al., Nature 2015).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : value_based / S (Landmark deep off-policy value baseline)
- Parent        : `q_learning` (Watkins & Dayan, 1992)
- Key Leap      : Replaces the discrete Q-table with a neural Q-function, stabilizes
                  correlated trajectories via Experience Replay, and freezes TD targets
                  periodically using a lagging Target Network.
- Extension Hook [AI-Prompt Hook]:
    1. Decouple argmax action choice (online net) from value estimate (target net) -> `double_dqn`.
    2. Split Q(s, a) into state-value V(s) and advantage A(s, a) -> `dueling_dqn`.
    3. Learn full return distribution atoms instead of scalar expectation -> `c51`.
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
    id="dqn",
    name="Deep Q-Network (DQN)",
    family="value_based",
    paradigm="off_policy",
    tier="S",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("discrete",),
    env_modalities=("discrete_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="q_learning",
    diff_from_parent="Replaces Q-table with neural network, circular Experience Replay, and target Q-network.",
    summary_plain="Learns Q-values via neural network + replay memory; standard starting point for discrete deep RL.",
    when_to_use="Discrete action spaces with continuous/high-dimensional state vectors where replay efficiency helps.",
    when_not_to_use="Continuous Box actions (use `sac` or `td3`) or large action branching (consider `ppo`).",
    hyperparam_intuition={
        "lr": "Adam step size for Q-network updates.",
        "buffer_size": "How many past transitions stay in replay memory.",
        "target_update_freq": "How often target network weights sync to prevent moving-target oscillations.",
    },
    extension_hooks=(
        "Decouple online argmax from target evaluation -> `double_dqn`",
        "Decompose Q into V(s) + centered A(s, a) -> `dueling_dqn`",
        "Add Prioritized Experience Replay weights from |TD error|",
    ),
    module_path="wannado.drl.algos.value_based.dqn_mnih2015",
    citation=AlgoCitation(
        key="mnih2015human",
        title="Human-level control through deep reinforcement learning",
        authors="Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al.",
        venue="Nature, vol. 518, no. 7540, pp. 529-533",
        year=2015,
        url="https://doi.org/10.1038/nature14236",
        bibtex=(
            "@article{mnih2015human,\n"
            "  title={Human-level control through deep reinforcement learning},\n"
            "  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and others},\n"
            "  journal={Nature},\n"
            "  volume={518},\n"
            "  number={7540},\n"
            "  pages={529--533},\n"
            "  year={2015},\n"
            "  doi={10.1038/nature14236}\n"
            "}"
        ),
        ieee='V. Mnih et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, pp. 529-533, 2015.',
        apa="Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.",
    ),
)


@dataclass
class DQNConfig:
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


def train_dqn(
    env_or_factory: Any,
    config: DQNConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Deep Q-Network (DQN) with VectorEnv, CNN/Dict/Attention encoders, and checkpoints."""
    cfg = config or DQNConfig()

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

    def _dqn_loss(
        q_net: QNetwork,
        target_net: QNetwork,
        b_obs: Any,
        b_act: torch.Tensor,
        b_rew: torch.Tensor,
        b_next: Any,
        b_term: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        # [AI-Prompt Hook]: Standard Bellman target; swap max(next_q) for Double DQN or C51 projection.
        with torch.no_grad():
            next_q_max = target_net(b_next).max(dim=1).values
            td_target = b_rew + cfg.gamma * (1.0 - b_term) * next_q_max
        q_pred = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
        return F.smooth_l1_loss(q_pred, td_target), float(q_pred.mean().item())

    return run_value_based_loop(
        env_or_factory,
        algo_id=META.id,
        cfg=cfg,
        eval_env_or_factory=eval_env_or_factory,
        build_q_nets=_build_nets,
        loss_fn=_dqn_loss,
        extra_diagnostics_fn=lambda _losses, qs: {"max_q_value": float(np.max(qs)) if qs else 0.0},
    )
