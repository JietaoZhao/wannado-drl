"""QMIX: Monotonic Value Function Factorisation for Cooperative MARL (Rashid et al., ICML 2018).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : multi_agent / S (Classic Cooperative Off-Policy MARL SOTA)
- Parent        : `vdn` (Sunehag et al., 2018)
- Key Diff      : Replaces VDN's rigid additive sum with a state-conditioned hypernetwork
                  mixer whose weights are constrained >= 0 via abs(), guaranteeing IGM
                  consistency (argmax_a Q_tot == (argmax_{a_1} Q_1, ..., argmax_{a_N} Q_N))
                  while capturing rich non-linear team coordination.
- Extension Hook [AI-Prompt Hook]:
    1. Switch to on-policy CTDE actor-critic -> `mappo`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.networks import QMixMixer, QNetwork
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import MultiAgentEnvAdapter
from wannado.drl.mdp.baselines import evaluate_marl_episodes

META = AlgoMeta(
    id="qmix",
    name="QMIX Monotonic Value Factorisation",
    family="multi_agent",
    paradigm="multi_agent_ctde",
    tier="S",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("multi_agent_discrete",),
    env_modalities=("multi_agent",),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="vdn",
    diff_from_parent="Replaces additive Q_tot sum with state-conditioned non-negative hypernetwork mixing.",
    summary_plain="Gold-standard off-policy cooperative MARL using a monotonic mixing hypernetwork driven by global state.",
    when_to_use="Cooperative discrete multi-agent tasks with global state available during training.",
    when_not_to_use="Continuous multi-agent actions (`maddpg`), mixed competitive tasks (`mappo`), or non-monotonic coordination.",
    hyperparam_intuition={
        "mixing_embed_dim": "Hidden width of hypernetwork mixer conditioning on global state.",
        "target_update_freq": "Target sync interval for both agent Q-net and mixer.",
        "lr": "Adam step size.",
    },
    extension_hooks=(
        "Add RNN/GRU into per-agent Q_i(tau_i, a_i) for POMDP SMAC-style coordination",
        "Switch to on-policy PPO with centralized critic -> `mappo`",
    ),
    module_path="wannado.drl.algos.multi_agent.qmix_rashid2018",
    citation=AlgoCitation(
        key="rashid2018qmix",
        title="QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning",
        authors="Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson",
        venue="Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80",
        year=2018,
        url="https://proceedings.mlr.press/v80/rashid18a.html",
        bibtex=(
            "@inproceedings{rashid2018qmix,\n"
            "  title={QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning},\n"
            "  author={Rashid, Tabish and Samvelyan, Mikayel and de Witt, Christian Schroeder and Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={80},\n"
            "  year={2018},\n"
            "  url={https://proceedings.mlr.press/v80/rashid18a.html}\n"
            "}"
        ),
        ieee='T. Rashid et al., "QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 80, 2018.',
        apa="Rashid, T., Samvelyan, M., de Witt, C. S., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. ICML (PMLR 80).",
    ),
)


@dataclass
class QMIXConfig:
    total_timesteps: int = 6_000
    learning_starts: int = 200
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 10_000
    batch_size: int = 64
    target_update_freq: int = 150
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.5
    hidden_dims: tuple[int, ...] = (128, 128)
    mixing_embed_dim: int = 32
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def _add_agent_id(obs_dict: dict[str, np.ndarray], agents: list[str]) -> np.ndarray:
    eye = np.eye(len(agents), dtype=np.float32)
    return np.stack([np.concatenate([obs_dict[a], eye[i]]) for i, a in enumerate(agents)], axis=0)


def train_qmix(
    env_or_adapter: Any,
    config: QMIXConfig | None = None,
) -> TrainResult:
    """Train QMIX on cooperative multi-agent discrete environments."""
    cfg = config or QMIXConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = env_or_adapter if isinstance(env_or_adapter, MultiAgentEnvAdapter) else MultiAgentEnvAdapter(env_or_adapter)
    if not env.is_discrete:
        raise ValueError("qmix supports discrete multi-agent action spaces only.")

    in_dim = env.obs_dim + env.num_agents
    q_net = QNetwork(in_dim, env.action_dim, cfg.hidden_dims).to(dev)
    target_net = QNetwork(in_dim, env.action_dim, cfg.hidden_dims).to(dev)
    hard_update(q_net, target_net)

    mixer = QMixMixer(env.num_agents, env.state_dim, mixing_embed_dim=cfg.mixing_embed_dim).to(dev)
    target_mixer = QMixMixer(env.num_agents, env.state_dim, mixing_embed_dim=cfg.mixing_embed_dim).to(dev)
    hard_update(mixer, target_mixer)

    params = list(q_net.parameters()) + list(mixer.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)

    cap = cfg.buffer_size
    buf_obs = np.zeros((cap, env.num_agents, in_dim), dtype=np.float32)
    buf_next = np.zeros((cap, env.num_agents, in_dim), dtype=np.float32)
    buf_state = np.zeros((cap, env.state_dim), dtype=np.float32)
    buf_next_state = np.zeros((cap, env.state_dim), dtype=np.float32)
    buf_act = np.zeros((cap, env.num_agents), dtype=np.int64)
    buf_rew = np.zeros((cap,), dtype=np.float32)
    buf_term = np.zeros((cap,), dtype=np.float32)
    ptr, size = 0, 0

    t0 = time.perf_counter()
    obs_dict, g_state, _ = env.reset(seed=cfg.seed)
    obs_arr = _add_agent_id(obs_dict, env.agents)
    ep_ret, ep_idx = 0.0, 0
    learning_curve: list[dict[str, float]] = []
    losses: list[float] = []

    for step in range(1, cfg.total_timesteps + 1):
        decay = max(1.0, cfg.exploration_fraction * cfg.total_timesteps)
        eps = max(cfg.eps_end, cfg.eps_start - (step / decay) * (cfg.eps_start - cfg.eps_end))
        if float(rng.random()) < eps:
            acts_arr = rng.integers(0, env.action_dim, size=env.num_agents)
        else:
            with torch.no_grad():
                acts_arr = q_net(torch.as_tensor(obs_arr, device=dev, dtype=torch.float32)).argmax(dim=1).cpu().numpy()

        act_dict = {a: int(acts_arr[i]) for i, a in enumerate(env.agents)}
        next_dict, rew_dict, term_dict, trunc_dict, next_state, _ = env.step(act_dict)
        team_rew = float(np.mean(list(rew_dict.values())))
        any_term = any(term_dict.values())
        any_trunc = any(trunc_dict.values())
        next_arr = _add_agent_id(next_dict, env.agents)

        buf_obs[ptr] = obs_arr
        buf_next[ptr] = next_arr
        buf_state[ptr] = g_state
        buf_next_state[ptr] = next_state
        buf_act[ptr] = acts_arr
        buf_rew[ptr] = team_rew
        buf_term[ptr] = float(any_term)
        ptr = (ptr + 1) % cap
        size = min(size + 1, cap)
        ep_ret += team_rew

        if any_term or any_trunc:
            ep_idx += 1
            if ep_idx % 5 == 0 or step == cfg.total_timesteps:
                learning_curve.append({"step": float(step), "episode_return": float(ep_ret)})
            obs_dict, g_state, _ = env.reset(seed=cfg.seed + ep_idx)
            obs_arr = _add_agent_id(obs_dict, env.agents)
            ep_ret = 0.0
        else:
            obs_arr = next_arr
            g_state = next_state

        if step >= cfg.learning_starts and size >= cfg.batch_size:
            idx = rng.integers(0, size, size=cfg.batch_size)
            b_o = torch.as_tensor(buf_obs[idx], device=dev)
            b_no = torch.as_tensor(buf_next[idx], device=dev)
            b_s = torch.as_tensor(buf_state[idx], device=dev)
            b_ns = torch.as_tensor(buf_next_state[idx], device=dev)
            b_a = torch.as_tensor(buf_act[idx], device=dev)
            b_r = torch.as_tensor(buf_rew[idx], device=dev)
            b_t = torch.as_tensor(buf_term[idx], device=dev)

            # [AI-Prompt Hook]: Monotonic hypernetwork mixes per-agent Q_i conditioned on global state.
            q_all = q_net(b_o.view(-1, in_dim)).view(cfg.batch_size, env.num_agents, env.action_dim)
            q_chosen = q_all.gather(2, b_a.unsqueeze(-1)).squeeze(-1)
            q_tot = mixer(q_chosen, b_s)

            with torch.no_grad():
                nq_max = target_net(b_no.view(-1, in_dim)).view(cfg.batch_size, env.num_agents, env.action_dim).max(dim=2).values
                nq_tot = target_mixer(nq_max, b_ns)
                td_target = b_r + cfg.gamma * (1.0 - b_t) * nq_tot

            loss = F.mse_loss(q_tot, td_target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            losses.append(float(loss.item()))

        if step % cfg.target_update_freq == 0:
            hard_update(q_net, target_net)
            hard_update(mixer, target_mixer)

    def policy_fn(o_dict: dict[str, np.ndarray], _info: dict[str, Any] | None = None) -> dict[str, int]:
        arr = _add_agent_id(o_dict, env.agents)
        with eval_mode(q_net), torch.no_grad():
            acts = q_net(torch.as_tensor(arr, device=dev, dtype=torch.float32)).argmax(dim=1).cpu().numpy()
        return {a: int(acts[i]) for i, a in enumerate(env.agents)}

    eval_stats = evaluate_marl_episodes(env, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.total_timesteps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model={"q_net": q_net, "mixer": mixer},
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={"mean_td_loss": float(np.mean(losses)) if losses else 0.0},
        eval_stats=eval_stats,
    )
