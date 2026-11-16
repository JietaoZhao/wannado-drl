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

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.buffers import ReplayBuffer
from wannado.drl.core.networks import CategoricalQNetwork
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

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
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.5
    hidden_dims: tuple[int, ...] = (128, 128)
    normalize_obs: bool = False
    eval_episodes: int = 10
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
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)
    if not env.is_discrete:
        raise ValueError("c51 supports Discrete action spaces only.")

    q_net = CategoricalQNetwork(
        env.obs_dim, env.action_dim, n_atoms=cfg.n_atoms, v_min=cfg.v_min, v_max=cfg.v_max, hidden_dims=cfg.hidden_dims
    ).to(dev)
    target_net = CategoricalQNetwork(
        env.obs_dim, env.action_dim, n_atoms=cfg.n_atoms, v_min=cfg.v_min, v_max=cfg.v_max, hidden_dims=cfg.hidden_dims
    ).to(dev)
    hard_update(q_net, target_net)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.buffer_size, env.obs_dim, env.action_dim, discrete_action=True, device=dev)
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None
    delta_z = (cfg.v_max - cfg.v_min) / (cfg.n_atoms - 1)

    t0 = time.perf_counter()
    obs, info = env.reset(seed=cfg.seed)
    ep_ret = 0.0
    ep_idx = 0
    learning_curve: list[dict[str, float]] = []
    kl_losses: list[float] = []

    for step in range(1, cfg.total_timesteps + 1):
        decay_steps = max(1.0, cfg.exploration_fraction * cfg.total_timesteps)
        eps = max(cfg.eps_end, cfg.eps_start - (step / decay_steps) * (cfg.eps_start - cfg.eps_end))
        if obs_rms is not None:
            obs_rms.update(obs)
            obs_proc = obs_rms.normalize(obs)
        else:
            obs_proc = obs

        mask = info.get("action_mask")
        if float(rng.random()) < eps:
            valid = np.flatnonzero(mask) if mask is not None else np.arange(env.action_dim)
            act = int(rng.choice(valid if valid.size > 0 else np.arange(env.action_dim)))
        else:
            with torch.no_grad():
                q_vals = q_net.get_q_values(torch.as_tensor(obs_proc, device=dev, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                if mask is not None:
                    m = torch.as_tensor(mask, device=dev, dtype=torch.bool).view_as(q_vals)
                    if bool(m.any()):
                        q_vals = q_vals.masked_fill(~m, -1e9)
                act = int(torch.argmax(q_vals).item())

        next_obs, reward, term, trunc, next_info = env.step(act)
        ep_ret += float(reward)
        next_proc = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs
        replay.add(obs_proc, act, float(reward), next_proc, bool(term))

        if term or trunc:
            ep_idx += 1
            if ep_idx % 5 == 0 or step == cfg.total_timesteps:
                learning_curve.append({"step": float(step), "episode_return": float(ep_ret), "epsilon": float(eps)})
            obs, info = env.reset(seed=cfg.seed + ep_idx)
            ep_ret = 0.0
        else:
            obs, info = next_obs, next_info

        if step >= cfg.learning_starts and replay.size >= cfg.batch_size:
            b_obs, b_act, b_rew, b_next, b_term = replay.sample(cfg.batch_size, rng=rng)
            # [AI-Prompt Hook]: Categorical Bellman projection onto fixed [v_min, v_max] atom grid.
            with torch.no_grad():
                next_q_vals = target_net.get_q_values(b_next)
                next_act = next_q_vals.argmax(dim=1)
                next_probs = target_net.get_probs(b_next)[torch.arange(cfg.batch_size, device=dev), next_act]

                tz = b_rew.unsqueeze(1) + cfg.gamma * (1.0 - b_term.unsqueeze(1)) * target_net.atoms.unsqueeze(0)
                tz = tz.clamp(cfg.v_min, cfg.v_max)
                b_idx = (tz - cfg.v_min) / delta_z
                low = b_idx.floor().long().clamp(0, cfg.n_atoms - 1)
                up = b_idx.ceil().long().clamp(0, cfg.n_atoms - 1)

                d_m_l = (up.float() + (low == up).float() - b_idx) * next_probs
                d_m_u = (b_idx - low.float()) * next_probs
                proj = torch.zeros_like(next_probs)
                offset = (torch.arange(cfg.batch_size, device=dev) * cfg.n_atoms).unsqueeze(1)
                proj.view(-1).index_add_(0, (low + offset).view(-1), d_m_l.view(-1))
                proj.view(-1).index_add_(0, (up + offset).view(-1), d_m_u.view(-1))

            logits = q_net.net(b_obs).view(cfg.batch_size, env.action_dim, cfg.n_atoms)
            chosen_logits = logits[torch.arange(cfg.batch_size, device=dev), b_act]
            log_probs = F.log_softmax(chosen_logits, dim=-1)
            loss = -(proj * log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()
            kl_losses.append(float(loss.item()))

        if step % cfg.target_update_freq == 0:
            hard_update(q_net, target_net)

    if obs_rms is not None:
        obs_rms.freeze(True)

    def policy_fn(obs_in: np.ndarray, info_in: dict[str, Any] | None = None) -> int:
        o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
        with eval_mode(q_net), torch.no_grad():
            q_vals = q_net.get_q_values(torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        mask_in = (info_in or {}).get("action_mask")
        if mask_in is not None:
            m = torch.as_tensor(mask_in, device=dev, dtype=torch.bool).view_as(q_vals)
            if bool(m.any()):
                q_vals = q_vals.masked_fill(~m, -1e9)
        return int(torch.argmax(q_vals).item())

    eval_target = eval_env_or_factory if eval_env_or_factory is not None else env
    eval_stats = evaluate_policy_episodes(eval_target, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.total_timesteps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=q_net,
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={"mean_cross_entropy_loss": float(np.mean(kl_losses)) if kl_losses else 0.0},
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
    )
