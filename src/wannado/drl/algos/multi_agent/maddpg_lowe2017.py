"""MADDPG: Multi-Agent Deep Deterministic Policy Gradient (Lowe et al., NeurIPS 2017).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : multi_agent / A (Continuous Mixed Cooperative-Competitive MARL)
- Parent        : `ddpg` (Lillicrap et al., 2016)
- Key Diff      : Gives each agent a decentralized continuous actor mu_i(o_i) while
                  training a centralized critic Q_i(s_global, a_1, ..., a_N) that sees
                  everyone's actions during training, eliminating environment non-stationarity.
- Extension Hook [AI-Prompt Hook]:
    1. Add Twin Clipped Double-Q per agent (MATD3) to reduce critic overestimation.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wannado.drl.core.networks import DeterministicActor, build_mlp
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import MultiAgentEnvAdapter
from wannado.drl.mdp.baselines import evaluate_marl_episodes

META = AlgoMeta(
    id="maddpg",
    name="Multi-Agent DDPG (MADDPG)",
    family="multi_agent",
    paradigm="multi_agent_ctde",
    tier="A",
    on_policy=False,
    sample_efficiency="high",
    action_spaces=("multi_agent_continuous",),
    env_modalities=("multi_agent",),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="ddpg",
    diff_from_parent="Trains per-agent decentralized continuous actors against centralized critics conditioned on joint actions.",
    summary_plain="Off-policy multi-agent actor-critic for continuous cooperative or competitive games.",
    when_to_use="Continuous multi-agent control/games where agents may have individual rewards or mixed motives.",
    when_not_to_use="Discrete cooperative MARL (`qmix` / `mappo`) or single-agent tasks (`td3` / `sac`).",
    hyperparam_intuition={
        "tau": "Polyak soft target update speed for actors and critics.",
        "exploration_noise": "Continuous Gaussian noise added to each agent's action during training.",
        "lr": "Adam step size.",
    },
    extension_hooks=(
        "Add twin critics + delayed policy update per agent -> MATD3",
    ),
    module_path="wannado.drl.algos.multi_agent.maddpg_lowe2017",
    citation=AlgoCitation(
        key="lowe2017multi",
        title="Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
        authors="Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch",
        venue="Advances in Neural Information Processing Systems (NeurIPS)",
        year=2017,
        url="https://arxiv.org/abs/1706.02275",
        bibtex=(
            "@inproceedings{lowe2017multi,\n"
            "  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},\n"
            "  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},\n"
            "  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},\n"
            "  year={2017},\n"
            "  url={https://arxiv.org/abs/1706.02275}\n"
            "}"
        ),
        ieee='R. Lowe et al., "Multi-agent actor-critic for mixed cooperative-competitive environments," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017.',
        apa="Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. NeurIPS.",
    ),
)


@dataclass
class MADDPGConfig:
    total_timesteps: int = 5_000
    learning_starts: int = 300
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.01
    buffer_size: int = 10_000
    batch_size: int = 64
    exploration_noise: float = 0.15
    hidden_dims: tuple[int, ...] = (64, 64)
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_maddpg(
    env_or_adapter: Any,
    config: MADDPGConfig | None = None,
) -> TrainResult:
    """Train Multi-Agent DDPG (MADDPG) on continuous multi-agent environments."""
    cfg = config or MADDPGConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = env_or_adapter if isinstance(env_or_adapter, MultiAgentEnvAdapter) else MultiAgentEnvAdapter(env_or_adapter)
    if env.is_discrete or env.action_low is None or env.action_high is None:
        raise ValueError("maddpg requires bounded continuous multi-agent action spaces.")

    n_ag = env.num_agents
    joint_act_dim = n_ag * env.action_dim
    actors = nn.ModuleList(
        [DeterministicActor(env.obs_dim, env.action_dim, env.action_low, env.action_high, cfg.hidden_dims).to(dev) for _ in range(n_ag)]
    )
    target_actors = nn.ModuleList(
        [DeterministicActor(env.obs_dim, env.action_dim, env.action_low, env.action_high, cfg.hidden_dims).to(dev) for _ in range(n_ag)]
    )
    critics = nn.ModuleList(
        [build_mlp(env.state_dim + joint_act_dim, cfg.hidden_dims, 1, activation=nn.ReLU).to(dev) for _ in range(n_ag)]
    )
    target_critics = nn.ModuleList(
        [build_mlp(env.state_dim + joint_act_dim, cfg.hidden_dims, 1, activation=nn.ReLU).to(dev) for _ in range(n_ag)]
    )
    for i in range(n_ag):
        hard_update(actors[i], target_actors[i])
        hard_update(critics[i], target_critics[i])

    actor_opts = [torch.optim.Adam(actors[i].parameters(), lr=cfg.lr) for i in range(n_ag)]
    critic_opts = [torch.optim.Adam(critics[i].parameters(), lr=cfg.lr) for i in range(n_ag)]

    cap = cfg.buffer_size
    buf_obs = np.zeros((cap, n_ag, env.obs_dim), dtype=np.float32)
    buf_next_obs = np.zeros((cap, n_ag, env.obs_dim), dtype=np.float32)
    buf_state = np.zeros((cap, env.state_dim), dtype=np.float32)
    buf_next_state = np.zeros((cap, env.state_dim), dtype=np.float32)
    buf_act = np.zeros((cap, n_ag, env.action_dim), dtype=np.float32)
    buf_rew = np.zeros((cap, n_ag), dtype=np.float32)
    buf_term = np.zeros((cap,), dtype=np.float32)
    ptr, size = 0, 0
    act_scale = (env.action_high - env.action_low) / 2.0

    t0 = time.perf_counter()
    obs_dict, g_state, _ = env.reset(seed=cfg.seed)
    obs_arr = np.stack([obs_dict[a] for a in env.agents], axis=0)
    ep_ret, ep_idx = 0.0, 0
    learning_curve: list[dict[str, float]] = []
    q_hist: list[float] = []

    for step in range(1, cfg.total_timesteps + 1):
        if step < cfg.learning_starts:
            acts_arr = rng.uniform(env.action_low, env.action_high, size=(n_ag, env.action_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                o_t = torch.as_tensor(obs_arr, device=dev, dtype=torch.float32)
                raw = torch.stack([actors[i](o_t[i : i + 1]).squeeze(0) for i in range(n_ag)], dim=0).cpu().numpy()
            noise = rng.normal(0.0, cfg.exploration_noise * act_scale, size=(n_ag, env.action_dim)).astype(np.float32)
            acts_arr = np.clip(raw + noise, env.action_low, env.action_high).astype(np.float32)

        act_dict = {a: acts_arr[i] for i, a in enumerate(env.agents)}
        next_dict, rew_dict, term_dict, trunc_dict, next_state, _ = env.step(act_dict)
        rew_arr = np.asarray([rew_dict[a] for a in env.agents], dtype=np.float32)
        next_arr = np.stack([next_dict[a] for a in env.agents], axis=0)
        any_term = any(term_dict.values())
        any_trunc = any(trunc_dict.values())

        buf_obs[ptr] = obs_arr
        buf_next_obs[ptr] = next_arr
        buf_state[ptr] = g_state
        buf_next_state[ptr] = next_state
        buf_act[ptr] = acts_arr
        buf_rew[ptr] = rew_arr
        buf_term[ptr] = float(any_term)
        ptr = (ptr + 1) % cap
        size = min(size + 1, cap)
        ep_ret += float(np.mean(rew_arr))

        if any_term or any_trunc:
            ep_idx += 1
            if ep_idx % 5 == 0 or step == cfg.total_timesteps:
                learning_curve.append({"step": float(step), "episode_return": float(ep_ret)})
            obs_dict, g_state, _ = env.reset(seed=cfg.seed + ep_idx)
            obs_arr = np.stack([obs_dict[a] for a in env.agents], axis=0)
            ep_ret = 0.0
        else:
            obs_arr = next_arr
            g_state = next_state

        if step >= cfg.learning_starts and size >= cfg.batch_size:
            idx = rng.integers(0, size, size=cfg.batch_size)
            b_o = torch.as_tensor(buf_obs[idx], device=dev)
            b_no = torch.as_tensor(buf_next_obs[idx], device=dev)
            b_s = torch.as_tensor(buf_state[idx], device=dev)
            b_ns = torch.as_tensor(buf_next_state[idx], device=dev)
            b_a = torch.as_tensor(buf_act[idx], device=dev)
            b_r = torch.as_tensor(buf_rew[idx], device=dev)
            b_t = torch.as_tensor(buf_term[idx], device=dev)

            with torch.no_grad():
                next_joint_a = torch.cat([target_actors[j](b_no[:, j, :]) for j in range(n_ag)], dim=-1)
            cur_joint_a = b_a.view(cfg.batch_size, joint_act_dim)

            # [AI-Prompt Hook]: Centralized critic update per agent using joint actions across all N agents.
            for i in range(n_ag):
                with torch.no_grad():
                    nq_i = target_critics[i](torch.cat([b_ns, next_joint_a], dim=-1)).squeeze(-1)
                    td_target = b_r[:, i] + cfg.gamma * (1.0 - b_t) * nq_i
                q_i = critics[i](torch.cat([b_s, cur_joint_a], dim=-1)).squeeze(-1)
                c_loss = F.mse_loss(q_i, td_target)
                critic_opts[i].zero_grad()
                c_loss.backward()
                critic_opts[i].step()
                if i == 0:
                    q_hist.append(float(q_i.mean().item()))

                pred_actions = [actors[j](b_o[:, j, :]) if j == i else b_a[:, j, :] for j in range(n_ag)]
                a_loss = -critics[i](torch.cat([b_s, torch.cat(pred_actions, dim=-1)], dim=-1)).mean()
                actor_opts[i].zero_grad()
                a_loss.backward()
                actor_opts[i].step()

                polyak_update(actors[i], target_actors[i], cfg.tau)
                polyak_update(critics[i], target_critics[i], cfg.tau)

    def policy_fn(o_dict: dict[str, np.ndarray], _info: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        arr = np.stack([o_dict[a] for a in env.agents], axis=0)
        with eval_mode(actors), torch.no_grad():
            o_t = torch.as_tensor(arr, device=dev, dtype=torch.float32)
            acts = torch.stack([actors[i](o_t[i : i + 1]).squeeze(0) for i in range(n_ag)], dim=0).cpu().numpy()
        return {a: acts[i] for i, a in enumerate(env.agents)}

    eval_stats = evaluate_marl_episodes(env, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=cfg.total_timesteps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model={"actors": actors, "critics": critics},
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={"mean_q_value": float(np.mean(q_hist)) if q_hist else 0.0},
        eval_stats=eval_stats,
    )
