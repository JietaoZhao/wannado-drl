"""A2C: Synchronous Advantage Actor-Critic (Mnih et al., ICML 2016).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / B (Bridge from REINFORCE to PPO)
- Parent        : `reinforce` (Williams, 1992)
- Key Diff      : Replaces full-episode Monte Carlo returns with fixed-horizon n-step
                  bootstrapped TD/GAE advantages via a learned Critic V(s) plus entropy bonus.
- Extension Hook [AI-Prompt Hook]:
    1. Replace single-step policy gradient with clipped importance ratio -> `ppo`.
    2. Replace hand-crafted TD/policy objective with meta-learned rule -> `discorl`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.buffers import RolloutBuffer
from wannado.drl.core.networks import ContinuousActorCritic, DiscreteActorCritic, MultiDiscreteActorCritic
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, explained_variance, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="a2c",
    name="Advantage Actor-Critic (A2C)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="B",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="reinforce",
    diff_from_parent="Uses n-step bootstrapped GAE advantages from a Critic V(s) so updates happen mid-episode with much lower variance.",
    summary_plain="Fast, lightweight synchronous Actor-Critic updating every n steps without waiting for episodes to finish.",
    when_to_use="Fast environments needing an immediate upgrade over REINFORCE or lightweight on-policy baseline.",
    when_not_to_use="Complex tasks where policy steps overshoot and collapse (prefer `ppo` with clipped ratios).",
    hyperparam_intuition={
        "n_steps": "Rollout horizon before each Actor-Critic update (longer = less bias, more variance).",
        "ent_coef": "Bonus discouraging premature greedy determinism.",
        "vf_coef": "Weight balancing Critic value loss against Actor policy loss.",
    },
    extension_hooks=(
        "Add PPO clipped surrogate objective over multi-epoch minibatches -> `ppo`",
        "Replace hand-written advantage target with learned meta-update network -> `discorl`",
    ),
    module_path="wannado.drl.algos.policy_gradient.a2c_mnih2016",
    citation=AlgoCitation(
        key="mnih2016asynchronous",
        title="Asynchronous Methods for Deep Reinforcement Learning",
        authors="Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu",
        venue="Proceedings of the 33rd International Conference on Machine Learning (ICML), PMLR 48",
        year=2016,
        url="http://proceedings.mlr.press/v48/mnih16.html",
        bibtex=(
            "@inproceedings{mnih2016asynchronous,\n"
            "  title={Asynchronous Methods for Deep Reinforcement Learning},\n"
            "  author={Mnih, Volodymyr and Badia, Adri{\\`a} Puigdom{\\`e}nech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy P. and Harley, Tim and Silver, David and Kavukcuoglu, Koray},\n"
            "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
            "  volume={48},\n"
            "  year={2016},\n"
            "  url={http://proceedings.mlr.press/v48/mnih16.html}\n"
            "}"
        ),
        ieee='V. Mnih et al., "Asynchronous methods for deep reinforcement learning," in Proc. Int. Conf. Mach. Learn. (ICML), PMLR 48, 2016.',
        apa="Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. ICML (PMLR 48).",
    ),
)


@dataclass
class A2CConfig:
    total_timesteps: int = 12_000
    n_steps: int = 32
    lr: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dims: tuple[int, ...] = (64, 64)
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_a2c(
    env_or_factory: Any,
    config: A2CConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train synchronous Advantage Actor-Critic (A2C) on Discrete or Continuous envs."""
    cfg = config or A2CConfig()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)

    net: DiscreteActorCritic | MultiDiscreteActorCritic | ContinuousActorCritic
    act_dtype: np.dtype[Any]
    if env.is_discrete:
        net = DiscreteActorCritic(env.obs_dim, env.action_dim, hidden_dims=cfg.hidden_dims).to(dev)
        act_shape: tuple[int, ...] = ()
        act_dtype = np.dtype(np.int64)
    elif env.action_nvec is not None:
        net = MultiDiscreteActorCritic(env.obs_dim, env.action_nvec, hidden_dims=cfg.hidden_dims).to(dev)
        act_shape = (len(env.action_nvec),)
        act_dtype = np.dtype(np.int64)
    else:
        net = ContinuousActorCritic(
            env.obs_dim,
            env.action_dim,
            hidden_dims=cfg.hidden_dims,
            action_low=env.action_low,
            action_high=env.action_high,
        ).to(dev)
        act_shape = (env.action_dim,)
        act_dtype = np.dtype(np.float32)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    buffer = RolloutBuffer(
        cfg.n_steps,
        env.obs_dim,
        act_shape,
        action_dtype=act_dtype,
        mask_dim=env.action_dim if env.is_discrete else None,
    )
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    t0 = time.perf_counter()
    obs, info = env.reset(seed=cfg.seed)
    ep_ret = 0.0
    ep_idx = 0
    steps = 0
    learning_curve: list[dict[str, float]] = []
    entropies, ev_list = [], []

    while steps < cfg.total_timesteps:
        buffer.reset()
        last_done = False
        for _ in range(cfg.n_steps):
            if steps >= cfg.total_timesteps:
                break
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
                    act_t, lp_t, _, val_t = net.get_action_and_value(obs_t, action_mask=mask_t)  # type: ignore[call-arg]
                    act_env: Any = int(act_t.item())
                else:
                    act_t, lp_t, _, val_t = net.get_action_and_value(obs_t)
                    act_env = act_t.squeeze(0).cpu().numpy()

            next_obs, reward, term, trunc, next_info = env.step(act_env)
            ep_ret += float(reward)
            steps += 1
            step_reward = float(reward)
            done_flag = bool(term or trunc)
            if trunc and not term:
                trunc_proc = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs
                with torch.no_grad():
                    trunc_v = float(
                        net.get_value(torch.as_tensor(trunc_proc, device=dev, dtype=torch.float32).unsqueeze(0)).item()
                    )
                step_reward += cfg.gamma * trunc_v

            buffer.add(
                obs_proc,
                act_env,
                float(lp_t.item()),
                step_reward,
                bool(term),
                float(val_t.item()),
                action_mask=mask if mask is not None else (np.ones(env.action_dim, dtype=np.bool_) if env.is_discrete else None),
                done=done_flag,
            )
            last_done = done_flag
            if done_flag:
                ep_idx += 1
                if ep_idx % 5 == 0:
                    learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
                obs, info = env.reset(seed=cfg.seed + ep_idx)
                ep_ret = 0.0
            else:
                obs, info = next_obs, next_info

        if buffer.ptr == 0:
            break
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            next_v = float(net.get_value(torch.as_tensor(next_proc, device=dev, dtype=torch.float32).unsqueeze(0)).item())
        buffer.compute_returns_and_advantage(next_v, last_done, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        n = buffer.ptr
        b_obs = torch.as_tensor(buffer.obs[:n], device=dev)
        b_act = torch.as_tensor(buffer.actions[:n], device=dev)
        b_adv = torch.as_tensor(buffer.advantages[:n], device=dev)
        b_ret = torch.as_tensor(buffer.returns[:n], device=dev)
        b_mask = torch.as_tensor(buffer.action_masks[:n], device=dev) if buffer.action_masks is not None else None

        if env.is_discrete:
            _, new_lp, entropy, new_val = net.get_action_and_value(b_obs, action=b_act, action_mask=b_mask)  # type: ignore[call-arg]
        else:
            _, new_lp, entropy, new_val = net.get_action_and_value(b_obs, action=b_act)

        # [AI-Prompt Hook]: Single-step policy gradient loss; replace with clipped surrogate for PPO.
        if b_adv.numel() > 1:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        pg_loss = -(new_lp * b_adv).mean()
        v_loss = 0.5 * F.mse_loss(new_val, b_ret)
        loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
        optimizer.step()

        entropies.append(float(entropy.mean().item()))
        ev_list.append(explained_variance(buffer.values[:n], buffer.returns[:n]))

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
            "mean_explained_variance": float(np.mean(ev_list)) if ev_list else 0.0,
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
