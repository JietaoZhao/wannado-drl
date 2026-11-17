"""PPO-GRU: Recurrent Proximal Policy Optimization for POMDPs (Schulman et al., 2017).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : policy_gradient / A (POMDP & Partial Observability Extension)
- Parent        : `ppo` (Schulman et al., 2017)
- Key Diff      : Inserts a GRU memory cell between encoder and Actor/Critic heads,
                  resetting hidden states cleanly at episode boundaries so the agent can
                  infer hidden velocity, queue state, or latent channel dynamics over time.
- Extension Hook [AI-Prompt Hook]:
    1. Replace GRU with Causal Transformer / Attention block for longer context windows.
    2. Add auxiliary state-reconstruction loss on hidden state h_t.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from wannado.drl.core.networks import MultiCategorical, RecurrentActorCriticGRU
from wannado.drl.core.provenance import resolve_run_optimizations
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import compute_gae, eval_mode, explained_variance, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="ppo_gru",
    name="Recurrent PPO (PPO-GRU for POMDP)",
    family="policy_gradient",
    paradigm="on_policy",
    tier="A",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("pomdp", "discrete_control", "continuous_control", "wireless", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=True,
    parent_id="ppo",
    diff_from_parent="Adds a GRU memory cell unrolled sequentially with episode-boundary hidden state resets for POMDPs.",
    summary_plain="PPO with short-term working memory (GRU) so agents infer hidden dynamics like velocity or noisy channels.",
    when_to_use="Stage 0 identifies partial observability (missing velocity, occluded sensors, delayed feedback).",
    when_not_to_use="Fully Markovian environments where single-frame observations already contain complete state (use `ppo`).",
    hyperparam_intuition={
        "n_steps": "Sequence rollout chunk length for truncated backpropagation through time.",
        "hidden_dim": "GRU memory capacity (64-128 for compact control/wireless tasks).",
        "clip_coef": "PPO trust-region clip width.",
    },
    extension_hooks=(
        "Swap GRU cell for causal Transformer block",
        "Add belief-state prediction loss to supervise latent GRU state h_t",
    ),
    module_path="wannado.drl.algos.policy_gradient.ppo_gru_schulman2017",
    citation=AlgoCitation(
        key="schulman2017proximal_gru",
        title="Proximal Policy Optimization Algorithms",
        authors="John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov",
        venue="arXiv preprint arXiv:1707.06347 (Recurrent GRU POMDP variant)",
        year=2017,
        url="https://arxiv.org/abs/1707.06347",
        bibtex=(
            "@article{schulman2017proximal_gru,\n"
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
class PPOGRUConfig:
    total_timesteps: int = 16_000
    n_steps: int = 128
    update_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_ppo_gru(
    env_or_factory: Any,
    config: PPOGRUConfig | None = None,
    *,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train Recurrent PPO-GRU preserving temporal sequence ordering and episode resets."""
    cfg = config or PPOGRUConfig()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)

    net = RecurrentActorCriticGRU(
        env.obs_dim,
        env.action_dim,
        hidden_dim=cfg.hidden_dim,
        is_discrete=env.is_discrete,
        action_low=env.action_low,
        action_high=env.action_high,
        action_nvec=env.action_nvec if not env.is_discrete else None,
    ).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    t0 = time.perf_counter()
    obs, info = env.reset(seed=cfg.seed)
    h_cur = torch.zeros(1, 1, cfg.hidden_dim, device=dev)
    ep_ret, ep_idx, steps = 0.0, 0, 0
    saw_action_mask = False
    learning_curve: list[dict[str, float]] = []
    ent_hist, ev_hist = [], []

    while steps < cfg.total_timesteps:
        h_start = h_cur.detach().clone()
        seq_obs, seq_act, seq_lp, seq_rew, seq_term, seq_done, seq_val, seq_mask = (
            [], [], [], [], [], [], [], []
        )
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
                act_t, lp_t, _, val_t, h_cur = net.step_action_value(obs_t, h_cur, action_mask=mask_t)
            act_env: Any = int(act_t.item()) if env.is_discrete else act_t.squeeze(0).cpu().numpy()

            next_obs, reward, term, trunc, next_info = env.step(act_env)
            ep_ret += float(reward)
            steps += 1
            done_flag = bool(term or trunc)
            step_reward = float(reward)
            if trunc and not term:
                trunc_proc = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs
                with torch.no_grad():
                    _, _, _, trunc_nv_t, _ = net.step_action_value(
                        torch.as_tensor(trunc_proc, device=dev, dtype=torch.float32).unsqueeze(0), h_cur
                    )
                    step_reward += cfg.gamma * float(trunc_nv_t.item())

            seq_obs.append(obs_proc)
            seq_act.append(act_env)
            seq_lp.append(float(lp_t.item()))
            seq_rew.append(step_reward)
            seq_term.append(float(term))
            seq_done.append(float(done_flag))
            seq_val.append(float(val_t.item()))
            if env.is_discrete:
                seq_mask.append(mask if mask is not None else np.ones(env.action_dim, dtype=np.bool_))

            if done_flag:
                ep_idx += 1
                if ep_idx % 5 == 0:
                    learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
                obs, info = env.reset(seed=cfg.seed + ep_idx)
                h_cur = torch.zeros(1, 1, cfg.hidden_dim, device=dev)
                ep_ret = 0.0
            else:
                obs, info = next_obs, next_info

        n = len(seq_obs)
        if n < 2:
            break
        next_proc = obs_rms.normalize(obs) if obs_rms is not None else obs
        with torch.no_grad():
            _, _, _, nv_t, _ = net.step_action_value(
                torch.as_tensor(next_proc, device=dev, dtype=torch.float32).unsqueeze(0), h_cur
            )
            next_v = float(nv_t.item())

        rews_arr = np.asarray(seq_rew, dtype=np.float32)
        vals_arr = np.asarray(seq_val, dtype=np.float32)
        terms_arr = np.asarray(seq_term, dtype=np.float32)
        dones_arr = np.asarray(seq_done, dtype=np.float32)
        adv_arr, ret_arr = compute_gae(
            rews_arr,
            vals_arr,
            terms_arr,
            next_v,
            bool(seq_term[-1]),
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            dones=dones_arr,
            next_done=bool(seq_done[-1]),
        )

        b_obs = torch.as_tensor(np.asarray(seq_obs, dtype=np.float32), device=dev).unsqueeze(1)
        b_done = torch.as_tensor(np.asarray(seq_done, dtype=np.float32), device=dev).unsqueeze(1)
        b_act = torch.as_tensor(np.asarray(seq_act, dtype=np.int64 if (env.is_discrete or env.action_nvec is not None) else np.float32), device=dev)
        b_old_lp = torch.as_tensor(np.asarray(seq_lp, dtype=np.float32), device=dev)
        b_adv = torch.as_tensor(adv_arr, device=dev)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        b_ret = torch.as_tensor(ret_arr, device=dev)
        b_mask = torch.as_tensor(np.asarray(seq_mask, dtype=np.bool_), device=dev) if env.is_discrete else None

        # [AI-Prompt Hook]: Unroll full sequence through GRU with done_seq episode-boundary resets.
        for _ in range(cfg.update_epochs):
            feats, _ = net.forward_seq(b_obs, h_start, done_seq=b_done)
            feats = feats.squeeze(1)
            new_val = net.critic(feats).squeeze(-1)
            if env.is_discrete:
                logits = net.actor(feats)
                if b_mask is not None:
                    logits = logits.masked_fill(~b_mask, -1e8)
                dist: Any = Categorical(logits=logits)
                new_lp = dist.log_prob(b_act)
                entropy = dist.entropy()
            elif net.action_nvec is not None:
                logits = net.actor(feats)
                dist = MultiCategorical(torch.split(logits, list(net.action_nvec), dim=-1))
                new_lp = dist.log_prob(b_act)
                entropy = dist.entropy()
            else:
                mean = net.action_bias + net.action_scale * net.actor(feats)
                std = net.action_scale * torch.exp(net.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
                dist = Normal(mean, std)
                new_lp = dist.log_prob(b_act).sum(-1)
                entropy = dist.entropy().sum(-1)

            ratio = (new_lp - b_old_lp).exp()
            pg_loss = torch.max(-b_adv * ratio, -b_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)).mean()
            v_loss = 0.5 * F.mse_loss(new_val, b_ret)
            loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            ent_hist.append(float(entropy.mean().item()))

        ev_hist.append(explained_variance(vals_arr, ret_arr))

    if obs_rms is not None:
        obs_rms.freeze(True)

    class _RecurrentPolicy:
        def __init__(self) -> None:
            self.h = torch.zeros(1, 1, cfg.hidden_dim, device=dev)

        def reset(self) -> None:
            self.h = torch.zeros(1, 1, cfg.hidden_dim, device=dev)

        def __call__(self, obs_in: np.ndarray, info_in: dict[str, Any] | None = None) -> Any:
            o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
            obs_t = torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0)
            m = (info_in or {}).get("action_mask")
            m_t = torch.as_tensor(m, device=dev, dtype=torch.bool).unsqueeze(0) if m is not None else None
            with eval_mode(net), torch.no_grad():
                act_t, _, _, _, self.h = net.step_action_value(obs_t, self.h, action_mask=m_t, deterministic=True)
            return int(act_t.item()) if env.is_discrete else act_t.squeeze(0).cpu().numpy()

    policy_fn = _RecurrentPolicy()
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
            "mean_explained_variance": float(np.mean(ev_hist)) if ev_hist else 0.0,
            "is_continuous": 0.0 if (env.is_discrete or env.action_nvec is not None) else 1.0,
            **(
                {"final_logstd_mean": float(net.actor_logstd.detach().mean().item())}
                if hasattr(net, "actor_logstd")
                else {}
            ),
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
        optimizations_used=[o.id for o in opts],
    )
