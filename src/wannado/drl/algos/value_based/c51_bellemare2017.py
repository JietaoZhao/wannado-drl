"""C51: Categorical Distributional Deep Q-Learning (Bellemare, Dabney, & Munos, ICML 2017).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : value_based / A (Distributional RL milestone)
- Parent        : `double_dqn` (van Hasselt et al., 2016)
- Key Diff      : Instead of predicting only the average return E[Z(s,a)], C51 predicts
                  the full probability histogram of returns across `n_atoms` (default 51)
                  support bins in [v_min, v_max] and projects Bellman targets onto those atoms.
- Extension Hook [AI-Prompt Hook]:
    1. Replace fixed atom grid with quantile regression -> QR-DQN / IQN.
    2. Use CVaR risk-sensitive policy selection over the learned return distribution!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from wannado.drl.core.algo_helpers import OnlineRunContext, run_value_based_loop
from wannado.drl.core.networks import CategoricalQNetwork
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult

META = AlgoMeta(
    id="c51",
    name="Categorical Distributional DQN (C51)",
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
    diff_from_parent="Learns the full categorical probability distribution over returns (51 atoms) via Cramer projection instead of a scalar Q mean.",
    summary_plain="Predicts the full return distribution (risk + upside histogram) rather than just an average Q-value.",
    when_to_use="Stochastic discrete environments where return variance matters or when exploring risk-sensitive control.",
    when_not_to_use="Continuous control (`sac`) or when true episode returns fall far outside [v_min, v_max].",
    hyperparam_intuition={
        "v_min": "Lower bound of return support; must cover the worst possible discounted return.",
        "v_max": "Upper bound of return support; must cover the best possible discounted return.",
        "n_atoms": "Histogram resolution across [v_min, v_max] (51 is the canonical paper default).",
    },
    extension_hooks=(
        "Compute CVaR over `probs * atoms` for risk-averse scheduling/finance",
        "Auto-scale [v_min, v_max] or switch to Quantile Huber loss (QR-DQN)",
    ),
    module_path="wannado.drl.algos.value_based.c51_bellemare2017",
    citation=AlgoCitation(
        key="bellemare2017distributional",
        title="A Distributional Perspective on Reinforcement Learning",
        authors="Marc G. Bellemare, Will Dabney, and Remi Munos",
        venue="Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR 70",
        year=2017,
        url="http://proceedings.mlr.press/v70/bellemare17a.html",
        bibtex=(
            "@inproceedings{bellemare2017distributional,\n"
            "  title={A Distributional Perspective on Reinforcement Learning},\n"
            "  author={Bellemare, Marc G. and Dabney, Will and Munos, R{\\'e}mi},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={70},\n"
            "  year={2017},\n"
            "  url={http://proceedings.mlr.press/v70/bellemare17a.html}\n"
            "}"
        ),
        ieee='M. G. Bellemare, W. Dabney, and R. Munos, "A distributional perspective on reinforcement learning," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 70, 2017.',
        apa="Bellemare, M. G., Dabney, W., & Munos, R. (2017). A distributional perspective on reinforcement learning. ICML (PMLR 70).",
    ),
)


@dataclass
class C51Config:
    total_timesteps: int = 12_000
    learning_starts: int = 400
    lr: float = 5e-4
    gamma: float = 0.99
    n_atoms: int = 51
    v_min: float = -20.0
    v_max: float = 200.0
    buffer_size: int = 20_000
    batch_size: int = 64
    target_update_freq: int = 200
    train_freq: int = 1
    gradient_steps: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.5
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


def train_c51(
    env_or_factory: Any,
    config: C51Config | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Categorical Distributional DQN (C51) on any Discrete-action environment."""
    cfg = config or C51Config()
    delta_z = (cfg.v_max - cfg.v_min) / max(1, cfg.n_atoms - 1)

    def _build(ctx: OnlineRunContext, dev: torch.device) -> tuple[Any, Any, dict[str, Any]]:
        kw = {
            "action_dim": int(ctx.env.action_dim),
            "n_atoms": int(cfg.n_atoms),
            "v_min": float(cfg.v_min),
            "v_max": float(cfg.v_max),
            "hidden_dims": list(cfg.hidden_dims),
            "extractor_type": cfg.extractor_type,
            "extractor_kwargs": cfg.extractor_kwargs,
        }
        q_net = CategoricalQNetwork(
            ctx.env.obs_spec,
            ctx.env.action_dim,
            n_atoms=cfg.n_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            hidden_dims=cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        target_net = CategoricalQNetwork(
            ctx.env.obs_spec,
            ctx.env.action_dim,
            n_atoms=cfg.n_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            hidden_dims=cfg.hidden_dims,
            extractor_type=cfg.extractor_type,
            extractor_kwargs=cfg.extractor_kwargs,
        ).to(dev)
        return q_net, target_net, {"net_class": "CategoricalQNetwork", "init_kwargs": kw}

    def _loss(
        q_net: Any,
        target_net: Any,
        b_obs: Any,
        b_act: torch.Tensor,
        b_rew: torch.Tensor,
        b_next: Any,
        b_term: torch.Tensor,
        b_next_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        bsz = int(b_act.shape[0])
        dev = b_act.device
        # [AI-Prompt Hook]: Categorical Bellman projection onto fixed [v_min, v_max] atom grid.
        with torch.no_grad():
            next_q_vals = target_net.get_q_values(b_next)
            if b_next_mask is not None:
                valid_row = b_next_mask.any(dim=1, keepdim=True)
                safe_mask = torch.where(valid_row, b_next_mask, torch.ones_like(b_next_mask))
                next_q_vals = next_q_vals.masked_fill(~safe_mask, -1e9)
            next_act = next_q_vals.argmax(dim=1)
            next_probs = target_net.get_probs(b_next)[torch.arange(bsz, device=dev), next_act]

            tz = b_rew.unsqueeze(1) + cfg.gamma * (1.0 - b_term.unsqueeze(1)) * target_net.atoms.unsqueeze(0)
            tz = tz.clamp(cfg.v_min, cfg.v_max)
            b_idx = (tz - cfg.v_min) / delta_z
            low = b_idx.floor().long().clamp(0, cfg.n_atoms - 1)
            up = b_idx.ceil().long().clamp(0, cfg.n_atoms - 1)

            d_m_l = (up.float() + (low == up).float() - b_idx) * next_probs
            d_m_u = (b_idx - low.float()) * next_probs
            proj = torch.zeros_like(next_probs)
            offset = (torch.arange(bsz, device=dev) * cfg.n_atoms).unsqueeze(1)
            proj.view(-1).index_add_(0, (low + offset).view(-1), d_m_l.view(-1))
            proj.view(-1).index_add_(0, (up + offset).view(-1), d_m_u.view(-1))

        logits = q_net.get_logits(b_obs)
        chosen_logits = logits[torch.arange(bsz, device=dev), b_act]
        log_probs = F.log_softmax(chosen_logits, dim=-1)
        loss = -(proj * log_probs).sum(dim=-1).mean()
        chosen_q = (F.softmax(chosen_logits.detach(), dim=-1) * q_net.atoms).sum(dim=-1).mean()
        return loss, float(chosen_q.item())

    return run_value_based_loop(
        env_or_factory,
        algo_id=META.id,
        cfg=cfg,
        eval_env_or_factory=eval_env_or_factory,
        build_q_nets=_build,
        loss_fn=_loss,
        use_get_q_values=True,
        policy_kind="c51_q",
        loss_metric_key="mean_cross_entropy_loss",
    )
