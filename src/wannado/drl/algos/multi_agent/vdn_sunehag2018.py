"""VDN: Value-Decomposition Networks for Cooperative MARL (Sunehag et al., AAMAS 2018).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : multi_agent / B (Foundational Cooperative Value Factorization)
- Parent        : `dqn` (Mnih et al., 2015)
- Key Diff      : Solves cooperative multi-agent credit assignment under Centralized
                  Training with Decentralized Execution (CTDE) by summing per-agent
                  utilities Q_tot = sum_i Q_i(o_i, a_i) during Bellman TD training, while
                  each agent picks greedy actions purely from its own local observation.
- Extension Hook [AI-Prompt Hook]:
    1. Replace additive sum with state-conditioned monotonic hypernetwork -> `qmix`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.networks import QNetwork
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import MultiAgentEnvAdapter
from wannado.drl.mdp.baselines import evaluate_marl_episodes

META = AlgoMeta(
    id="vdn",
    name="Value-Decomposition Networks (VDN)",
    family="multi_agent",
    paradigm="multi_agent_ctde",
    tier="B",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("multi_agent_discrete",),
    env_modalities=("multi_agent",),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="dqn",
    diff_from_parent="Decomposes team Q_tot as sum_i Q_i(o_i, a_i) across decentralized discrete agents.",
    summary_plain="Teaches cooperative multi-agent teams by splitting team rewards across agents additively.",
    when_to_use="Cooperative discrete multi-agent games as a clean baseline before trying `qmix` or `mappo`.",
    when_not_to_use="Non-additive coordination requiring global context (`qmix`), continuous actions (`maddpg`), or competitive games (`mappo`).",
    hyperparam_intuition={
        "lr": "Adam step size for shared agent Q-network.",
        "gamma": "Team reward discount factor.",
        "target_update_freq": "Target network sync interval.",
    },
    extension_hooks=(
        "Replace sum(Q_i) with state-conditioned monotonic hypernetwork -> `qmix`",
    ),
    module_path="wannado.drl.algos.multi_agent.vdn_sunehag2018",
    citation=AlgoCitation(
        key="sunehag2018value",
        title="Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward",
        authors="Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, and Thore Graepel",
        venue="Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS)",
        year=2018,
        url="https://arxiv.org/abs/1706.05296",
        bibtex=(
            "@inproceedings{sunehag2018value,\n"
            "  title={Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward},\n"
            "  author={Sunehag, Peter and Lever, Guy and Gruslys, Audrunas and Czarnecki, Wojciech Marian and Zambaldi, Vinicius and Jaderberg, Max and Lanctot, Marc and Sonnerat, Nicolas and Leibo, Joel Z. and Tuyls, Karl and Graepel, Thore},\n"
            "  booktitle={Proceedings of the International Conference on Autonomous Agents and MultiAgent Systems (AAMAS)},\n"
            "  year={2018},\n"
            "  url={https://arxiv.org/abs/1706.05296}\n"
            "}"
        ),
        ieee='P. Sunehag et al., "Value-decomposition networks for cooperative multi-agent learning based on team reward," in Proc. AAMAS, 2018.',
        apa="Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., Lanctot, M., Sonnerat, N., Leibo, J. Z., Tuyls, K., & Graepel, T. (2018). Value-decomposition networks for cooperative multi-agent learning based on team reward. AAMAS.",
    ),
)


@dataclass
class VDNConfig:
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
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def _add_agent_id(obs_dict: dict[str, np.ndarray], agents: list[str]) -> np.ndarray:
    n = len(agents)
    eye = np.eye(n, dtype=np.float32)
    return np.stack([np.concatenate([obs_dict[a], eye[i]]) for i, a in enumerate(agents)], axis=0)


def train_vdn(
    env_or_adapter: Any,
    config: VDNConfig | None = None,
) -> TrainResult:
    """Train Value-Decomposition Networks (VDN) on cooperative multi-agent discrete envs."""
    cfg = config or VDNConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = env_or_adapter if isinstance(env_or_adapter, MultiAgentEnvAdapter) else MultiAgentEnvAdapter(env_or_adapter)
    if not env.is_discrete:
        raise ValueError("vdn supports discrete multi-agent action spaces only.")

    in_dim = env.obs_dim + env.num_agents
    q_net = QNetwork(in_dim, env.action_dim, cfg.hidden_dims).to(dev)
    target_net = QNetwork(in_dim, env.action_dim, cfg.hidden_dims).to(dev)
    hard_update(q_net, target_net)
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    cap = cfg.buffer_size
    buf_obs = np.zeros((cap, env.num_agents, in_dim), dtype=np.float32)
    buf_next = np.zeros((cap, env.num_agents, in_dim), dtype=np.float32)
    buf_act = np.zeros((cap, env.num_agents), dtype=np.int64)
    buf_rew = np.zeros((cap,), dtype=np.float32)
    buf_term = np.zeros((cap,), dtype=np.float32)
    ptr, size = 0, 0

    t0 = time.perf_counter()
    obs_dict, _, _ = env.reset(seed=cfg.seed)
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
                q_vals = q_net(torch.as_tensor(obs_arr, device=dev, dtype=torch.float32))
                acts_arr = q_vals.argmax(dim=1).cpu().numpy()

        act_dict = {a: int(acts_arr[i]) for i, a in enumerate(env.agents)}
        next_dict, rew_dict, term_dict, trunc_dict, _, _ = env.step(act_dict)
        team_rew = float(np.mean(list(rew_dict.values())))
        any_term = any(term_dict.values())
        any_trunc = any(trunc_dict.values())
        next_arr = _add_agent_id(next_dict, env.agents)

        buf_obs[ptr] = obs_arr
        buf_next[ptr] = next_arr
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
            obs_dict, _, _ = env.reset(seed=cfg.seed + ep_idx)
            obs_arr = _add_agent_id(obs_dict, env.agents)
            ep_ret = 0.0
        else:
            obs_arr = next_arr

        if step >= cfg.learning_starts and size >= cfg.batch_size:
            idx = rng.integers(0, size, size=cfg.batch_size)
            b_o = torch.as_tensor(buf_obs[idx], device=dev)
            b_no = torch.as_tensor(buf_next[idx], device=dev)
            b_a = torch.as_tensor(buf_act[idx], device=dev)
            b_r = torch.as_tensor(buf_rew[idx], device=dev)
            b_t = torch.as_tensor(buf_term[idx], device=dev)

            # [AI-Prompt Hook]: VDN sums per-agent Q_i into Q_tot; replace sum with QMixMixer for state-dependent mixing.
            q_all = q_net(b_o.view(-1, in_dim)).view(cfg.batch_size, env.num_agents, env.action_dim)
            q_chosen = q_all.gather(2, b_a.unsqueeze(-1)).squeeze(-1)
            q_tot = q_chosen.sum(dim=1)

            with torch.no_grad():
                nq_all = target_net(b_no.view(-1, in_dim)).view(cfg.batch_size, env.num_agents, env.action_dim)
                nq_tot = nq_all.max(dim=2).values.sum(dim=1)
                td_target = b_r + cfg.gamma * (1.0 - b_t) * nq_tot

            loss = F.mse_loss(q_tot, td_target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.item()))

        if step % cfg.target_update_freq == 0:
            hard_update(q_net, target_net)

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
        model=q_net,
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={"mean_td_loss": float(np.mean(losses)) if losses else 0.0},
        eval_stats=eval_stats,
    )
