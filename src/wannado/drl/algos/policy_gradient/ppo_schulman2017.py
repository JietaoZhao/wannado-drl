"""PPO: Proximal Policy Optimization (Schulman et al., arXiv:1707.06347).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / S (Default First-Try Benchmark Across Discrete & Continuous DRL)
- Parent        : `trpo` (Schulman et al., ICML 2015)
- Key Diff      : Replaces TRPO's expensive second-order Conjugate Gradient KL solver with
                  a simple first-order clipped surrogate objective min(r * A, clip(r, 1-e, 1+e) * A),
                  multi-epoch minibatch SGD, GAE advantages, and optional invalid-action masking.
- Extension Hook [AI-Prompt Hook]:
    1. Inject GRU hidden memory across time steps for POMDPs -> `ppo_gru`.
    2. Share global state in Critic across N decentralized actors -> `mappo`.
    3. Drop Critic completely and normalize across K rollouts per prompt -> `grpo`.
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
from wannado.drl.core.provenance import resolve_run_optimizations
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, explained_variance, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="ppo",
    name="Proximal Policy Optimization (PPO-Clip)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="S",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=False,
    parent_id="trpo",
    diff_from_parent="Replaces second-order Fisher KL trust region with first-order clipped surrogate objective + minibatch epochs.",
    summary_plain="Reliable default workhorse for both discrete (with action masking) and continuous control.",
    when_to_use="First-try online DRL benchmark when simulator speed is reasonable and stability matters most.",
    when_not_to_use="Expensive slow simulators needing maximal sample reuse (`sac`/`td3`), partial observability (`ppo_gru`), or offline logs (`iql_offline`).",
    hyperparam_intuition={
        "clip_coef": "Trust-region clip width (0.2 keeps policy changes conservative per batch).",
        "n_steps": "Steps collected before each policy update batch.",
        "ent_coef": "Exploration curiosity weight preventing premature greedy collapse.",
        "gamma": "Future reward discount horizon.",
    },
    extension_hooks=(
        "Add GRU state recurrence across episode steps -> `ppo_gru`",
        "Add asymmetric Clip-Higher (eps_low vs eps_high) -> `dapo`-style exploration preservation",
        "Use centralized global state in critic for N agents -> `mappo`",
    ),
    module_path="wannado.drl.algos.policy_gradient.ppo_schulman2017",
    citation=AlgoCitation(
        key="schulman2017proximal",
        title="Proximal Policy Optimization Algorithms",
        authors="John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov",
        venue="arXiv preprint arXiv:1707.06347",
        year=2017,
        url="https://arxiv.org/abs/1707.06347",
        bibtex=(
            "@article{schulman2017proximal,\n"
            "  title={Proximal Policy Optimization Algorithms},\n"
            "  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},\n"
            "  journal={arXiv preprint arXiv:1707.06347},\n"
            "  year={2017},\n"
            "  url={https://arxiv.org/abs/1707.06347}\n"
            "}"
        ),
        ieee='J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.',
        apa="Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.",
    ),
)


@dataclass
class PPOConfig:
    total_timesteps: int = 16_000
    n_steps: int = 256
    update_epochs: int = 6
    minibatch_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dims: tuple[int, ...] = (64, 64)
    normalize_obs: bool = False
    anneal_lr: bool = True
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_ppo(
    env_or_factory: Any,
    config: PPOConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train PPO-Clip with GAE, action masking, and eval-frozen observation normalization."""
    cfg = config or PPOConfig()
    rng = set_global_seed(cfg.seed)
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
    ep_ret, ep_idx, steps = 0.0, 0, 0
    saw_action_mask = False
    learning_curve: list[dict[str, float]] = []
    ent_hist, kl_hist, ev_hist = [], [], []

    while steps < cfg.total_timesteps:
        if cfg.anneal_lr:
            frac = 1.0 - (steps / max(1, cfg.total_timesteps))
            optimizer.param_groups[0]["lr"] = max(1e-6, cfg.lr * frac)

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
            if mask is not None:
                saw_action_mask = True
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

        n = buffer.ptr
        if n < 2:
            break
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            next_v = float(net.get_value(torch.as_tensor(next_proc, device=dev, dtype=torch.float32).unsqueeze(0)).item())
        buffer.compute_returns_and_advantage(next_v, last_done, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

        b_obs = torch.as_tensor(buffer.obs[:n], device=dev)
        b_act = torch.as_tensor(buffer.actions[:n], device=dev)
        b_old_lp = torch.as_tensor(buffer.logprobs[:n], device=dev)
        b_adv = torch.as_tensor(buffer.advantages[:n], device=dev)
        b_ret = torch.as_tensor(buffer.returns[:n], device=dev)
        b_mask = torch.as_tensor(buffer.action_masks[:n], device=dev) if buffer.action_masks is not None else None

        mb_size = min(cfg.minibatch_size, n)
        indices = np.arange(n)
        for _ in range(cfg.update_epochs):
            rng.shuffle(indices)
            for start in range(0, n, mb_size):
                mb_idx = indices[start : start + mb_size]
                if len(mb_idx) < 2:
                    continue
                if env.is_discrete:
                    _, new_lp, entropy, new_val = net.get_action_and_value(
                        b_obs[mb_idx], action=b_act[mb_idx], action_mask=b_mask[mb_idx] if b_mask is not None else None  # type: ignore[call-arg]
                    )
                else:
                    _, new_lp, entropy, new_val = net.get_action_and_value(b_obs[mb_idx], action=b_act[mb_idx])

                log_ratio = new_lp - b_old_lp[mb_idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                    kl_hist.append(approx_kl)

                mb_adv = b_adv[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # [AI-Prompt Hook]: PPO-Clip surrogate; customize asymmetric Clip-Higher or KL penalty here.
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * F.mse_loss(new_val, b_ret[mb_idx])
                loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()
                ent_hist.append(float(entropy.mean().item()))

        ev_hist.append(explained_variance(buffer.values[:n], buffer.returns[:n]))

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
    cfg_dict = asdict(cfg)
    opts = resolve_run_optimizations(META.id, config=cfg_dict, saw_action_mask=saw_action_mask)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=steps,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=policy_fn,
        model=net,
        config=cfg_dict,
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
            "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
            "mean_approx_kl": float(np.mean(kl_hist)) if kl_hist else 0.0,
            "mean_explained_variance": float(np.mean(ev_hist)) if ev_hist else 0.0,
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
        optimizations_used=[o.id for o in opts],
    )
