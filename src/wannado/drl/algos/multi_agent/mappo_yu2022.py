"""MAPPO: Multi-Agent Proximal Policy Optimization (Yu et al., NeurIPS 2022).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : multi_agent / S (Default On-Policy MARL SOTA Benchmark)
- Parent        : `ppo` (Schulman et al., 2017)
- Key Diff      : Applies PPO under Centralized Training with Decentralized Execution
                  (CTDE): decentralized actors observe ONLY local views o_i (+ agent ID),
                  while a centralized Critic V(s_global) sees global state during training
                  to stabilize non-stationary multi-agent credit assignment.
- Extension Hook [AI-Prompt Hook]:
    1. Replace parameter-shared actor with per-agent unshared policies for heterogeneous roles.
    2. Inject PopArt / RunningMeanStd value normalization on V(s_global) targets.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

from wannado.drl.core.networks import build_mlp
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import compute_gae, eval_mode, resolve_device, set_global_seed
from wannado.drl.core.wrappers import MultiAgentEnvAdapter
from wannado.drl.mdp.baselines import evaluate_marl_episodes

META = AlgoMeta(
    id="mappo",
    name="Multi-Agent PPO (MAPPO)",
    family="multi_agent",
    paradigm="multi_agent_ctde",
    tier="S",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("multi_agent_discrete", "multi_agent_continuous"),
    env_modalities=("multi_agent",),
    supports_action_mask=False,
    supports_recurrent=False,
    parent_id="ppo",
    diff_from_parent="Pairs decentralized local-observation actors with a centralized global-state critic under CTDE.",
    summary_plain="Gold-standard on-policy MARL algorithm; easy to tune and handles both discrete and continuous multi-agent teams.",
    when_to_use="Cooperative or mixed multi-agent tasks when simulator steps are fast and stability matters.",
    when_not_to_use="Ultra-slow simulators requiring replay reuse (`qmix`/`maddpg`) or single-agent tasks (`ppo`).",
    hyperparam_intuition={
        "clip_coef": "PPO policy trust-region clip width.",
        "n_steps": "Joint steps collected per rollout update.",
        "share_parameters": "Share actor weights across homogeneous agents (with one-hot ID) for faster convergence.",
    },
    extension_hooks=(
        "Disable `share_parameters` for heterogeneous agent action/observation roles",
        "Add agent-specific local + global state concatenation in centralized critic",
    ),
    module_path="wannado.drl.algos.multi_agent.mappo_yu2022",
    citation=AlgoCitation(
        key="yu2022surprising",
        title="The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games",
        authors="Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu",
        venue="Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track",
        year=2022,
        url="https://arxiv.org/abs/2103.01955",
        bibtex=(
            "@inproceedings{yu2022surprising,\n"
            "  title={The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games},\n"
            "  author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},\n"
            "  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},\n"
            "  year={2022},\n"
            "  url={https://arxiv.org/abs/2103.01955}\n"
            "}"
        ),
        ieee='C. Yu et al., "The surprising effectiveness of PPO in cooperative multi-agent games," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2022.',
        apa="Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. NeurIPS.",
    ),
)


@dataclass
class MAPPOConfig:
    total_timesteps: int = 8_000
    n_steps: int = 128
    update_epochs: int = 5
    minibatch_size: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dims: tuple[int, ...] = (64, 64)
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def _stack_local_obs(obs_dict: dict[str, np.ndarray], agents: list[str]) -> np.ndarray:
    eye = np.eye(len(agents), dtype=np.float32)
    return np.stack([np.concatenate([obs_dict[a], eye[i]]) for i, a in enumerate(agents)], axis=0)


def train_mappo(
    env_or_adapter: Any,
    config: MAPPOConfig | None = None,
) -> TrainResult:
    """Train Multi-Agent PPO (MAPPO) under CTDE on discrete or continuous multi-agent envs."""
    cfg = config or MAPPOConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = env_or_adapter if isinstance(env_or_adapter, MultiAgentEnvAdapter) else MultiAgentEnvAdapter(env_or_adapter)

    in_obs_dim = env.obs_dim + env.num_agents
    actor = build_mlp(in_obs_dim, cfg.hidden_dims, env.action_dim, activation=nn.Tanh, orthogonal_init=True, last_std=0.01).to(dev)
    critic = build_mlp(env.state_dim, cfg.hidden_dims, 1, activation=nn.Tanh, orthogonal_init=True, last_std=1.0).to(dev)
    actor_logstd = nn.Parameter(torch.zeros(1, env.action_dim, device=dev)) if not env.is_discrete else None
    scale_np = np.ones((env.action_dim,), dtype=np.float32)
    bias_np = np.zeros((env.action_dim,), dtype=np.float32)
    if not env.is_discrete and env.action_low is not None and env.action_high is not None:
        lo = np.asarray(env.action_low, dtype=np.float32).reshape(-1)
        hi = np.asarray(env.action_high, dtype=np.float32).reshape(-1)
        if lo.shape == (env.action_dim,) and hi.shape == (env.action_dim,):
            finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
            scale_np = np.where(finite, 0.5 * (hi - lo), 1.0).astype(np.float32)
            bias_np = np.where(finite, 0.5 * (hi + lo), 0.0).astype(np.float32)
    act_scale = torch.as_tensor(scale_np, device=dev, dtype=torch.float32)
    act_bias = torch.as_tensor(bias_np, device=dev, dtype=torch.float32)
    params = list(actor.parameters()) + list(critic.parameters()) + ([actor_logstd] if actor_logstd is not None else [])
    opt = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)

    t0 = time.perf_counter()
    obs_dict, g_state, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_idx, steps = 0.0, 0, 0
    learning_curve: list[dict[str, float]] = []
    ent_hist: list[float] = []
    ev_hist: list[float] = []

    while steps < cfg.total_timesteps:
        b_local, b_states, b_acts, b_lps, b_rews, b_terms, b_dones, b_vals = [], [], [], [], [], [], [], []
        last_done = False
        for _ in range(cfg.n_steps):
            if steps >= cfg.total_timesteps:
                break
            local_arr = _stack_local_obs(obs_dict, env.agents)
            loc_t = torch.as_tensor(local_arr, device=dev, dtype=torch.float32)
            gs_t = torch.as_tensor(g_state, device=dev, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                val = float(critic(gs_t).item())
                logits_or_mean = actor(loc_t)
                if env.is_discrete:
                    dist: Any = Categorical(logits=logits_or_mean)
                    act_t = dist.sample()
                    lp_t = dist.log_prob(act_t)
                    act_np: Any = act_t.cpu().numpy()
                    act_dict = {a: int(act_np[i]) for i, a in enumerate(env.agents)}
                else:
                    mean_c = act_bias + act_scale * logits_or_mean
                    std = act_scale * torch.exp(actor_logstd.clamp(-5.0, 2.0)).expand_as(mean_c)  # type: ignore[union-attr]
                    dist = Normal(mean_c, std)
                    act_t = dist.sample()
                    lp_t = dist.log_prob(act_t).sum(dim=-1)
                    act_np = act_t.cpu().numpy()
                    if env.action_low is not None and env.action_high is not None:
                        clipped = np.clip(act_np, env.action_low, env.action_high)
                    else:
                        clipped = act_np
                    act_dict = {a: clipped[i] for i, a in enumerate(env.agents)}

            next_dict, rew_dict, term_dict, trunc_dict, next_state, _ = env.step(act_dict)
            team_rew = float(np.mean(list(rew_dict.values())))
            any_term = any(term_dict.values())
            any_trunc = any(trunc_dict.values())
            done_flag = bool(any_term or any_trunc)
            step_rew = team_rew
            if any_trunc and not any_term:
                with torch.no_grad():
                    trunc_v = float(
                        critic(torch.as_tensor(next_state, device=dev, dtype=torch.float32).unsqueeze(0)).item()
                    )
                step_rew += cfg.gamma * trunc_v
            steps += 1
            ep_ret += team_rew

            b_local.append(local_arr)
            b_states.append(g_state)
            b_acts.append(act_np)
            b_lps.append(lp_t.cpu().numpy())
            b_rews.append(step_rew)
            b_terms.append(float(any_term))
            b_dones.append(float(done_flag))
            b_vals.append(val)
            last_done = done_flag

            if done_flag:
                ep_idx += 1
                if ep_idx % 5 == 0:
                    learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
                obs_dict, g_state, _ = env.reset(seed=cfg.seed + ep_idx)
                ep_ret = 0.0
            else:
                obs_dict, g_state = next_dict, next_state

        n = len(b_rews)
        if n < 2:
            break
        with torch.no_grad():
            next_v = float(critic(torch.as_tensor(g_state, device=dev, dtype=torch.float32).unsqueeze(0)).item())

        vals_arr = np.asarray(b_vals, dtype=np.float32)
        adv_arr, ret_arr = compute_gae(
            np.asarray(b_rews, dtype=np.float32),
            vals_arr,
            np.asarray(b_terms, dtype=np.float32),
            next_v,
            last_done,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            dones=np.asarray(b_dones, dtype=np.float32),
            next_done=last_done,
        )
        ev_hist.append(float(1.0 - np.var(ret_arr - vals_arr) / max(np.var(ret_arr), 1e-8)))

        t_loc = torch.as_tensor(np.asarray(b_local, dtype=np.float32), device=dev)
        t_gs = torch.as_tensor(np.asarray(b_states, dtype=np.float32), device=dev)
        t_act = torch.as_tensor(np.asarray(b_acts, dtype=np.int64 if env.is_discrete else np.float32), device=dev)
        t_old_lp = torch.as_tensor(np.asarray(b_lps, dtype=np.float32), device=dev)
        t_adv = torch.as_tensor(adv_arr, device=dev)
        t_ret = torch.as_tensor(ret_arr, device=dev)

        mb_sz = min(cfg.minibatch_size, n)
        indices = np.arange(n)
        for _ in range(cfg.update_epochs):
            rng.shuffle(indices)
            for start in range(0, n, mb_sz):
                mb = indices[start : start + mb_sz]
                if len(mb) < 2:
                    continue
                loc_flat = t_loc[mb].view(-1, in_obs_dim)
                out = actor(loc_flat)
                if env.is_discrete:
                    dist_n: Any = Categorical(logits=out)
                    new_lp = dist_n.log_prob(t_act[mb].view(-1)).view(len(mb), env.num_agents)
                    entropy = dist_n.entropy().mean()
                else:
                    mean_c = act_bias + act_scale * out
                    std = act_scale * torch.exp(actor_logstd.clamp(-5.0, 2.0)).expand_as(mean_c)  # type: ignore[union-attr]
                    dist_n = Normal(mean_c, std)
                    new_lp = dist_n.log_prob(t_act[mb].view(-1, env.action_dim)).sum(-1).view(len(mb), env.num_agents)
                    entropy = dist_n.entropy().sum(-1).mean()

                # [AI-Prompt Hook]: Centralized critic V(s_global) evaluates team advantage for all agents.
                ratio = (new_lp - t_old_lp[mb]).exp()
                mb_adv = ((t_adv[mb] - t_adv[mb].mean()) / (t_adv[mb].std() + 1e-8)).unsqueeze(1)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)).mean()
                v_pred = critic(t_gs[mb]).squeeze(-1)
                v_loss = 0.5 * F.mse_loss(v_pred, t_ret[mb])
                loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                ent_hist.append(float(entropy.item()))

    def policy_fn(o_dict: dict[str, np.ndarray], _info: dict[str, Any] | None = None) -> dict[str, Any]:
        loc = _stack_local_obs(o_dict, env.agents)
        with eval_mode(actor), torch.no_grad():
            out = actor(torch.as_tensor(loc, device=dev, dtype=torch.float32))
            if env.is_discrete:
                acts = out.argmax(dim=1).cpu().numpy()
                return {a: int(acts[i]) for i, a in enumerate(env.agents)}
            acts = (act_bias + act_scale * out).cpu().numpy()
            if env.action_low is not None and env.action_high is not None:
                acts = np.clip(acts, env.action_low, env.action_high)
            return {a: acts[i] for i, a in enumerate(env.agents)}

    eval_stats = evaluate_marl_episodes(env, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model={"actor": actor, "critic": critic},
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
            "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
            "mean_explained_variance": float(np.mean(ev_hist)) if ev_hist else 0.0,
            "is_continuous": 0.0 if env.is_discrete else 1.0,
            **(
                {"final_logstd_mean": float(actor_logstd.detach().mean().item())}
                if actor_logstd is not None
                else {}
            ),
        },
        eval_stats=eval_stats,
    )
