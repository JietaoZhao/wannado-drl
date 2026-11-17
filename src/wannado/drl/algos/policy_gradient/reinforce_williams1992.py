"""REINFORCE: Monte Carlo Policy Gradient with Optional Baseline (Williams, 1992).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / C (Foundational Policy Gradient Baseline)
- Parent        : None (Origin of neural policy-gradient reinforcement learning)
- Plain Intuition:
    Instead of estimating Q-values for every move, REINFORCE adjusts the policy directly:
    after finishing an episode, moves that led to higher-than-average total returns get
    their probabilities boosted, while worse moves get suppressed.
- Extension Hook [AI-Prompt Hook]:
    1. Replace episodic Monte Carlo return with TD Advantage + Critic -> `a2c`.
    2. Constrain KL jump per step -> `trpo` or clipped surrogate -> `ppo`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.networks import ContinuousActorCritic, DiscreteActorCritic, MultiDiscreteActorCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="reinforce",
    name="REINFORCE (Monte Carlo Policy Gradient)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="C",
    on_policy=True,
    sample_efficiency="low",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id=None,
    diff_from_parent="Origin node of neural policy-gradient RL: optimizes log pi(a|s) weighted by episodic returns.",
    summary_plain="Classic policy-gradient baseline that boosts action probabilities along entire finished episodes.",
    when_to_use="Teaching policy gradients, simple episodic tasks, or verifying gradient flow before scaling to PPO.",
    when_not_to_use="Long horizons, noisy rewards, or sample-constrained tasks (high variance; upgrade to `a2c` or `ppo`).",
    hyperparam_intuition={
        "lr": "Policy optimizer step size; too high causes premature deterministic collapse.",
        "gamma": "Discount horizon weighting future rewards.",
        "use_baseline": "Subtracts a learned state-value baseline V(s) to slash gradient variance without adding bias.",
    },
    extension_hooks=(
        "Replace full return G_t with bootstrapped GAE -> `a2c` / `ppo`",
        "Add group-relative reward normalization across K rollouts -> `grpo`",
    ),
    module_path="wannado.drl.algos.policy_gradient.reinforce_williams1992",
    citation=AlgoCitation(
        key="williams1992simple",
        title="Simple statistical gradient-following algorithms for connectionist reinforcement learning",
        authors="Ronald J. Williams",
        venue="Machine Learning, vol. 8, no. 3-4, pp. 229-256",
        year=1992,
        url="https://doi.org/10.1007/BF00992696",
        bibtex=(
            "@article{williams1992simple,\n"
            "  title={Simple statistical gradient-following algorithms for connectionist reinforcement learning},\n"
            "  author={Williams, Ronald J.},\n"
            "  journal={Machine Learning},\n"
            "  volume={8},\n"
            "  number={3-4},\n"
            "  pages={229--256},\n"
            "  year={1992},\n"
            "  doi={10.1007/BF00992696}\n"
            "}"
        ),
        ieee='R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," Machine Learning, vol. 8, no. 3-4, pp. 229-256, 1992.',
        apa="Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4), 229-256.",
    ),
)


@dataclass
class ReinforceConfig:
    total_timesteps: int = 12_000
    lr: float = 1e-3
    gamma: float = 0.99
    ent_coef: float = 0.01
    use_baseline: bool = True
    hidden_dims: tuple[int, ...] = (64, 64)
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_reinforce(
    env_or_factory: Any,
    config: ReinforceConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train REINFORCE (with optional learned state-value baseline) on Discrete or Continuous envs."""
    cfg = config or ReinforceConfig()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)

    net: DiscreteActorCritic | MultiDiscreteActorCritic | ContinuousActorCritic
    if env.is_discrete:
        net = DiscreteActorCritic(env.obs_dim, env.action_dim, hidden_dims=cfg.hidden_dims).to(dev)
    elif env.action_nvec is not None:
        net = MultiDiscreteActorCritic(env.obs_dim, env.action_nvec, hidden_dims=cfg.hidden_dims).to(dev)
    else:
        net = ContinuousActorCritic(
            env.obs_dim,
            env.action_dim,
            hidden_dims=cfg.hidden_dims,
            action_low=env.action_low,
            action_high=env.action_high,
        ).to(dev)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    t0 = time.perf_counter()
    steps = 0
    ep_idx = 0
    learning_curve: list[dict[str, float]] = []
    entropies: list[float] = []

    while steps < cfg.total_timesteps:
        obs, info = env.reset(seed=cfg.seed + ep_idx)
        ep_idx += 1
        ep_obs, ep_acts, ep_rews, ep_masks = [], [], [], []
        done = False
        last_trunc = False
        last_term = False

        while not done and steps < cfg.total_timesteps:
            if obs_rms is not None:
                obs_rms.update(obs)
                obs_proc = obs_rms.normalize(obs)
            else:
                obs_proc = obs

            obs_t = torch.as_tensor(obs_proc, device=dev, dtype=torch.float32).unsqueeze(0)
            mask = info.get("action_mask")
            mask_t = torch.as_tensor(mask, device=dev, dtype=torch.bool).unsqueeze(0) if mask is not None else None
            with torch.no_grad():
                if env.is_discrete:
                    act_t, _, _, _ = net.get_action_and_value(obs_t, action_mask=mask_t)  # type: ignore[call-arg]
                    act_env: Any = int(act_t.item())
                else:
                    act_t, _, _, _ = net.get_action_and_value(obs_t)
                    act_env = act_t.squeeze(0).cpu().numpy()

            next_obs, reward, term, trunc, info = env.step(act_env)
            ep_obs.append(obs_proc)
            ep_acts.append(act_env)
            ep_rews.append(float(reward))
            if env.is_discrete:
                ep_masks.append(mask if mask is not None else np.ones(env.action_dim, dtype=np.bool_))
            steps += 1
            obs = next_obs
            last_term = bool(term)
            last_trunc = bool(trunc)
            done = bool(term or trunc)

        # Compute backward discounted returns G_t (bootstrapping across TimeLimit truncation when baseline exists)
        returns = np.zeros(len(ep_rews), dtype=np.float32)
        g = 0.0
        if cfg.use_baseline and (last_trunc or not last_term) and not last_term:
            tail_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
            with torch.no_grad():
                g = float(net.get_value(torch.as_tensor(tail_proc, device=dev, dtype=torch.float32).unsqueeze(0)).item())
        for i in range(len(ep_rews) - 1, -1, -1):
            g = ep_rews[i] + cfg.gamma * g
            returns[i] = g

        b_obs = torch.as_tensor(np.asarray(ep_obs, dtype=np.float32), device=dev)
        b_ret = torch.as_tensor(returns, device=dev)
        b_act = torch.as_tensor(
            np.asarray(ep_acts, dtype=np.int64 if (env.is_discrete or env.action_nvec is not None) else np.float32), device=dev
        )
        b_mask = torch.as_tensor(np.asarray(ep_masks, dtype=np.bool_), device=dev) if env.is_discrete else None

        if env.is_discrete:
            _, logprob, entropy, values = net.get_action_and_value(b_obs, action=b_act, action_mask=b_mask)  # type: ignore[call-arg]
        else:
            _, logprob, entropy, values = net.get_action_and_value(b_obs, action=b_act)

        # [AI-Prompt Hook]: Replace Monte Carlo G_t - V(s) below with GAE or group-normalized rewards.
        if cfg.use_baseline:
            adv = (b_ret - values.detach())
            value_loss = 0.5 * F.mse_loss(values, b_ret)
        else:
            adv = b_ret
            value_loss = torch.tensor(0.0, device=dev)

        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = -(logprob * adv).mean()
        loss = policy_loss + value_loss - cfg.ent_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        entropies.append(float(entropy.mean().item()))

        if ep_idx % 5 == 0 or steps >= cfg.total_timesteps:
            learning_curve.append({"step": float(steps), "episode_return": float(sum(ep_rews)), "entropy": entropies[-1]})

    if obs_rms is not None:
        obs_rms.freeze(True)

    def policy_fn(obs_in: np.ndarray, info_in: dict[str, Any] | None = None) -> Any:
        o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
        obs_t = torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0)
        with eval_mode(net), torch.no_grad():
            if env.is_discrete:
                m = (info_in or {}).get("action_mask")
                m_t = torch.as_tensor(m, device=dev, dtype=torch.bool).unsqueeze(0) if m is not None else None
                act_t, _, _, _ = net.get_action_and_value(obs_t, action_mask=m_t, deterministic=True)  # type: ignore[call-arg]
                return int(act_t.item())
            act_t, _, _, _ = net.get_action_and_value(obs_t, deterministic=True)
            return act_t.squeeze(0).cpu().numpy()

    eval_target = eval_env_or_factory if eval_env_or_factory is not None else env
    eval_stats = evaluate_policy_episodes(eval_target, policy_fn, num_episodes=cfg.eval_episodes, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=net,
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "final_entropy": float(entropies[-1]) if entropies else 0.0,
            "is_continuous": 0.0 if (env.is_discrete or env.action_nvec is not None) else 1.0,
            **(
                {"final_logstd_mean": float(net.actor_logstd.detach().mean().item())}
                if isinstance(net, ContinuousActorCritic)
                else {}
            ),
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
    )
