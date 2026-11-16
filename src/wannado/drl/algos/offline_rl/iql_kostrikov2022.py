"""IQL: Implicit Q-Learning for Offline RL (Kostrikov, Nair, & Levine, ICLR 2022).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : offline_rl / S (Clean, OOD-Free Offline RL SOTA)
- Parent        : `cql` (Kumar et al., 2020)
- Key Diff      : Eliminates querying out-of-distribution actions altogether! Uses
                  asymmetric expectile regression (`expectile` tau in (0.5, 1.0)) on in-dataset
                  (s, a) pairs to estimate an upper state-value V(s), then trains the actor
                  via Advantage-Weighted Regression (AWR).
- Extension Hook [AI-Prompt Hook]:
    1. Tune `expectile` (0.7-0.9) and `awr_beta` temperature to trade off safety vs stitching.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from wannado.drl.core.networks import QNetwork, TwinQCritic, build_mlp
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import OfflineTransitionDataset, RunningMeanStd
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="iql_offline",
    name="Implicit Q-Learning (IQL)",
    family="offline_rl",
    paradigm="offline",
    tier="S",
    on_policy=False,
    sample_efficiency="very_high",
    action_spaces=("discrete", "continuous"),
    env_modalities=("offline_rl", "discrete_control", "continuous_control"),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="cql",
    diff_from_parent="Replaces OOD action penalization with in-sample expectile V(s) regression + Advantage-Weighted Regression (AWR).",
    summary_plain="Learns purely from logged datasets without ever querying unseen actions; fast and very stable.",
    when_to_use="Preferred default for static offline datasets where CQL hyperparameter tuning feels brittle.",
    when_not_to_use="Online environments where live exploration is allowed (`ppo`/`sac`) or multi-agent systems (`mappo`).",
    hyperparam_intuition={
        "expectile": "Asymmetry parameter (0.7-0.8) steering V(s) toward the upper tail of dataset actions.",
        "awr_beta": "Inverse temperature controlling how sharply policy extraction favors high-advantage actions.",
        "num_grad_steps": "Offline gradient iterations over logged transitions.",
    },
    extension_hooks=(
        "Swap Gaussian AWR actor for diffusion/flow behavior policy",
        "Fine-tune online after offline pretraining using `sac`",
    ),
    module_path="wannado.drl.algos.offline_rl.iql_kostrikov2022",
    citation=AlgoCitation(
        key="kostrikov2022offline",
        title="Offline Reinforcement Learning with Implicit Q-Learning",
        authors="Ilya Kostrikov, Ashvin Nair, and Sergey Levine",
        venue="International Conference on Learning Representations (ICLR)",
        year=2022,
        url="https://arxiv.org/abs/2110.06169",
        bibtex=(
            "@inproceedings{kostrikov2022offline,\n"
            "  title={Offline Reinforcement Learning with Implicit Q-Learning},\n"
            "  author={Kostrikov, Ilya and Nair, Ashvin and Levine, Sergey},\n"
            "  booktitle={International Conference on Learning Representations (ICLR)},\n"
            "  year={2022},\n"
            "  url={https://arxiv.org/abs/2110.06169}\n"
            "}"
        ),
        ieee='I. Kostrikov, A. Nair, and S. Levine, "Offline reinforcement learning with implicit Q-learning," in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.',
        apa="Kostrikov, I., Nair, A., & Levine, S. (2022). Offline reinforcement learning with implicit Q-learning. ICLR.",
    ),
)


@dataclass
class IQLConfig:
    num_grad_steps: int = 2_500
    batch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.75
    awr_beta: float = 3.0
    max_awr_weight: float = 100.0
    hidden_dims: tuple[int, ...] = (128, 128)
    normalize_obs: bool = True
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0.0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def train_iql_offline(
    dataset: OfflineTransitionDataset,
    config: IQLConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Implicit Q-Learning (IQL) purely from an OfflineTransitionDataset."""
    cfg = config or IQLConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    n_data = len(dataset)
    if n_data < 2:
        raise ValueError("Offline dataset must contain at least 2 transitions.")

    obs_rms = RunningMeanStd((dataset.obs_dim,)) if cfg.normalize_obs else None
    if obs_rms is not None:
        obs_rms.update(dataset.obs)
        obs_rms.freeze(True)
        obs_all = obs_rms.normalize(dataset.obs)
        next_obs_all = obs_rms.normalize(dataset.next_obs)
    else:
        obs_all = dataset.obs
        next_obs_all = dataset.next_obs

    v_net = build_mlp(dataset.obs_dim, cfg.hidden_dims, 1, activation=torch.nn.ReLU).to(dev)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=cfg.lr)

    if dataset.is_discrete:
        q_net = QNetwork(dataset.obs_dim, dataset.action_dim, cfg.hidden_dims).to(dev)
        target_q = QNetwork(dataset.obs_dim, dataset.action_dim, cfg.hidden_dims).to(dev)
        hard_update(q_net, target_q)
        actor_logits = build_mlp(dataset.obs_dim, cfg.hidden_dims, dataset.action_dim, activation=torch.nn.ReLU).to(dev)
        q_opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
        actor_opt = torch.optim.Adam(actor_logits.parameters(), lr=cfg.lr)
    else:
        q_net_twin = TwinQCritic(dataset.obs_dim, dataset.action_dim, cfg.hidden_dims).to(dev)
        target_q_twin = TwinQCritic(dataset.obs_dim, dataset.action_dim, cfg.hidden_dims).to(dev)
        hard_update(q_net_twin, target_q_twin)
        actor_mean = build_mlp(dataset.obs_dim, cfg.hidden_dims, dataset.action_dim, activation=torch.nn.Tanh).to(dev)
        actor_logstd = torch.nn.Parameter(torch.zeros(1, dataset.action_dim, device=dev))
        q_opt = torch.optim.Adam(q_net_twin.parameters(), lr=cfg.lr)
        actor_opt = torch.optim.Adam(list(actor_mean.parameters()) + [actor_logstd], lr=cfg.lr)
        act_low = dataset.action_low if dataset.action_low is not None else -np.ones(dataset.action_dim, dtype=np.float32)
        act_high = dataset.action_high if dataset.action_high is not None else np.ones(dataset.action_dim, dtype=np.float32)
        scale_np = (act_high - act_low) / 2.0
        bias_np = (act_high + act_low) / 2.0

    t0 = time.perf_counter()
    learning_curve: list[dict[str, float]] = []
    v_losses, q_losses = [], []

    for step in range(1, cfg.num_grad_steps + 1):
        idx = rng.integers(0, n_data, size=cfg.batch_size)
        b_obs = torch.as_tensor(obs_all[idx], device=dev, dtype=torch.float32)
        b_rew = torch.as_tensor(dataset.rewards[idx], device=dev, dtype=torch.float32)
        b_next = torch.as_tensor(next_obs_all[idx], device=dev, dtype=torch.float32)
        b_term = torch.as_tensor(dataset.terminateds[idx], device=dev, dtype=torch.float32)

        if dataset.is_discrete:
            b_act = torch.as_tensor(dataset.actions[idx], device=dev, dtype=torch.long)
            with torch.no_grad():
                q_t = target_q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            v_pred = v_net(b_obs).squeeze(-1)
            v_loss = _expectile_loss(q_t - v_pred, cfg.expectile)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            with torch.no_grad():
                v_next = v_net(b_next).squeeze(-1)
                q_target = b_rew + cfg.gamma * (1.0 - b_term) * v_next
            q_pred = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(q_pred, q_target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()
            polyak_update(q_net, target_q, cfg.tau)

            with torch.no_grad():
                adv = target_q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1) - v_net(b_obs).squeeze(-1)
                weights = torch.exp(cfg.awr_beta * adv).clamp(max=cfg.max_awr_weight)
            log_probs = Categorical(logits=actor_logits(b_obs)).log_prob(b_act)
            actor_loss = -(weights * log_probs).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
        else:
            b_act = torch.as_tensor(dataset.actions[idx], device=dev, dtype=torch.float32)
            with torch.no_grad():
                tq1, tq2 = target_q_twin(b_obs, b_act)
                q_t = torch.min(tq1, tq2).squeeze(-1)
            v_pred = v_net(b_obs).squeeze(-1)
            v_loss = _expectile_loss(q_t - v_pred, cfg.expectile)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()

            with torch.no_grad():
                v_next = v_net(b_next).squeeze(-1)
                q_target = b_rew + cfg.gamma * (1.0 - b_term) * v_next
            q1, q2 = q_net_twin(b_obs, b_act)
            q_loss = F.mse_loss(q1.squeeze(-1), q_target) + F.mse_loss(q2.squeeze(-1), q_target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()
            polyak_update(q_net_twin, target_q_twin, cfg.tau)

            # [AI-Prompt Hook]: Advantage-Weighted Regression extracts actor strictly from dataset actions.
            with torch.no_grad():
                tq1_a, tq2_a = target_q_twin(b_obs, b_act)
                adv = torch.min(tq1_a, tq2_a).squeeze(-1) - v_net(b_obs).squeeze(-1)
                weights = torch.exp(cfg.awr_beta * adv).clamp(max=cfg.max_awr_weight)
            mean = torch.tanh(actor_mean(b_obs)) * torch.as_tensor(scale_np, device=dev) + torch.as_tensor(bias_np, device=dev)
            std = torch.exp(actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
            log_probs = Normal(mean, std).log_prob(b_act).sum(dim=-1)
            actor_loss = -(weights * log_probs).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

        v_losses.append(float(v_loss.item()))
        q_losses.append(float(q_loss.item()))
        if step % max(1, cfg.num_grad_steps // 10) == 0:
            learning_curve.append({"step": float(step), "v_loss": v_losses[-1], "q_loss": q_losses[-1]})

    if dataset.is_discrete:
        model_obj: Any = {"actor": actor_logits, "q": q_net, "v": v_net}

        def policy_fn(obs_in: np.ndarray, _info: dict[str, Any] | None = None) -> int:
            o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
            with eval_mode(actor_logits), torch.no_grad():
                return int(torch.argmax(actor_logits(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0))).item())
    else:
        model_obj = {"actor": actor_mean, "q": q_net_twin, "v": v_net}

        def policy_fn(obs_in: np.ndarray, _info: dict[str, Any] | None = None) -> np.ndarray:
            o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
            with eval_mode(actor_mean), torch.no_grad():
                m = torch.tanh(actor_mean(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0)))
                return m.squeeze(0).cpu().numpy() * scale_np + bias_np

    eval_stats = (
        evaluate_policy_episodes(eval_env_or_factory, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)
        if eval_env_or_factory is not None
        else None
    )

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.num_grad_steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=model_obj,
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_expectile_v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "mean_q_loss": float(np.mean(q_losses)) if q_losses else 0.0,
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True,
    )
