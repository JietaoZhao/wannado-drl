"""DiscoRL: Meta-Learned Reinforcement Learning Update Rule (Oh et al., Nature 2025).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : llm_and_discovered / S (Meta-Discovered Update Rule Paradigm)
- Parent        : `a2c` (Mnih et al., 2016)
- Key Diff      : Instead of hard-coding hand-written Bellman/GAE equations, DiscoRL learns
                  a neural Meta-Update Rule (`DiscoMetaUpdateNet`) via bi-level meta-gradients.
                  The inner agent outputs policy logits plus unconstrained latent prediction
                  vectors y(s) and z(s,a); the meta-network unrolls backward over experience
                  to emit policy gradient targets and latent prediction targets!
- Extension Hook [AI-Prompt Hook]:
    1. Run `meta_train=True` to meta-discover update rules across environments.
    2. Supply a frozen pretrained `meta_rule` (`meta_train=False`) to train inner agents fast.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.func import functional_call

from wannado.drl.core.networks import build_mlp
from wannado.drl.core.provenance import resolve_run_optimizations
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, resolve_device, set_global_seed
from wannado.drl.core.wrappers import RunningMeanStd, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes

META = AlgoMeta(
    id="discorl",
    name="Discovered RL Update Rule (DiscoRL)",
    family="llm_and_discovered",
    paradigm="meta_discovered",
    tier="S",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("discrete", "continuous"),
    env_modalities=("discrete_control", "continuous_control", "meta_rl", "custom_sim"),
    supports_action_mask=True,
    supports_recurrent=True,
    parent_id="a2c",
    diff_from_parent="Replaces hand-crafted TD/GAE target equations with a meta-learned GRU update rule and unconstrained latent predictions y(s).",
    summary_plain="Meta-learned RL algorithm (Nature 2025) where a neural update rule invents its own latent targets instead of rigid TD math.",
    when_to_use="Exploring meta-learned RL rules, latent auxiliary predictions, or research on automated RL algorithm discovery.",
    when_not_to_use="Production baselines needing zero meta-overhead or predictable classical tuning (`ppo`/`sac`).",
    hyperparam_intuition={
        "latent_dim": "Size of unconstrained latent vector y(s) replacing fixed scalar V(s).",
        "meta_train": "If True, jointly meta-optimizes the update rule itself via differentiable inner updates.",
        "inner_lr": "Step size used by the inner agent following the meta-discovered update rule.",
    },
    extension_hooks=(
        "Meta-train across a distribution of diverse tasks/environments",
        "Inspect discovered latent prediction channels y(s) for emergent value/entropy semantics",
    ),
    module_path="wannado.drl.algos.llm_and_discovered.discorl_oh2025",
    citation=AlgoCitation(
        key="oh2025discovering",
        title="Discovering state-of-the-art reinforcement learning algorithms",
        authors="Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, Matteo Hessel, Luisa Zintgraf, Satinder Singh, Hado van Hasselt, and David Silver",
        venue="Nature, vol. 648, pp. 312-319",
        year=2025,
        url="https://doi.org/10.1038/s41586-025-09761-x",
        bibtex=(
            "@article{oh2025discovering,\n"
            "  title={Discovering state-of-the-art reinforcement learning algorithms},\n"
            "  author={Oh, Junhyuk and Farquhar, Gregory and Kemaev, Iurii and Calian, Dan A. and Hessel, Matteo and Zintgraf, Luisa and Singh, Satinder and van Hasselt, Hado and Silver, David},\n"
            "  journal={Nature},\n"
            "  volume={648},\n"
            "  pages={312--319},\n"
            "  year={2025},\n"
            "  doi={10.1038/s41586-025-09761-x}\n"
            "}"
        ),
        ieee='J. Oh et al., "Discovering state-of-the-art reinforcement learning algorithms," Nature, vol. 648, pp. 312-319, 2025.',
        apa="Oh, J., Farquhar, G., Kemaev, I., Calian, D. A., Hessel, M., Zintgraf, L., Singh, S., van Hasselt, H., & Silver, D. (2025). Discovering state-of-the-art reinforcement learning algorithms. Nature, 648, 312-319.",
    ),
)


class DiscoAgentNet(nn.Module):
    """Inner DiscoRL agent outputting policy parameters + unconstrained latent predictions y(s)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 8,
        hidden_dims: tuple[int, ...] = (64, 64),
        is_discrete: bool = True,
    ) -> None:
        super().__init__()
        self.is_discrete = is_discrete
        self.trunk = build_mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1], activation=nn.Tanh)
        last_h = hidden_dims[-1]
        self.policy_head = nn.Linear(last_h, action_dim)
        self.latent_y_head = nn.Linear(last_h, latent_dim)
        if not is_discrete:
            self.logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.tanh(self.trunk(obs))
        return self.policy_head(feat), self.latent_y_head(feat)


class DiscoMetaUpdateNet(nn.Module):
    """Backward-unrolled GRU meta-network emitting policy advantage signals & latent prediction targets."""

    def __init__(self, latent_dim: int = 8, meta_hidden: int = 32) -> None:
        super().__init__()
        # Inputs per step: reward (1), nonterminal (1), logp (1), y_t (K), y_next (K)
        in_dim = 3 + 2 * latent_dim
        self.gru = nn.GRU(in_dim, meta_hidden, batch_first=False)
        self.adv_head = nn.Linear(meta_hidden, 1)
        self.target_y_head = nn.Linear(meta_hidden, latent_dim)
        # Initialize policy gate near a stable positive TD-like prior so early updates are constructive.
        nn.init.zeros_(self.adv_head.weight)
        nn.init.zeros_(self.adv_head.bias)

    def forward(
        self,
        rewards: torch.Tensor,
        nonterminals: torch.Tensor,
        logps: torch.Tensor,
        y_curr: torch.Tensor,
        y_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unroll backward from T-1 -> 0 to produce (pi_adv_hat, y_target_hat)."""
        feats = torch.cat(
            [rewards.unsqueeze(-1), nonterminals.unsqueeze(-1), logps.unsqueeze(-1), y_curr, y_next],
            dim=-1,
        ).unsqueeze(1)  # (T, 1, D)
        rev_feats = torch.flip(feats, dims=[0])
        rev_out, _ = self.gru(rev_feats)
        out = torch.flip(rev_out, dims=[0]).squeeze(1)

        # Blend discovered meta-output with a differentiable bootstrap residual anchor for stable discovery.
        td_anchor = rewards + 0.99 * nonterminals * y_next[:, 0].detach() - y_curr[:, 0].detach()
        pi_adv = td_anchor + self.adv_head(out).squeeze(-1)
        y_target = self.target_y_head(out)
        y_target = torch.cat([(rewards + 0.99 * nonterminals * y_next[:, 0]).unsqueeze(-1), y_target[:, 1:]], dim=-1)
        return pi_adv, y_target


@dataclass
class DiscoRLConfig:
    total_timesteps: int = 10_000
    n_steps: int = 32
    inner_lr: float = 1e-3
    meta_lr: float = 5e-4
    latent_dim: int = 8
    meta_hidden: int = 32
    ent_coef: float = 0.01
    pred_coef: float = 0.5
    meta_train: bool = True
    hidden_dims: tuple[int, ...] = (64, 64)
    normalize_obs: bool = False
    eval_episodes: int = 10
    device: str = "auto"
    seed: int = 42


def train_discorl(
    env_or_factory: Any,
    config: DiscoRLConfig | None = None,
    *,
    meta_rule: DiscoMetaUpdateNet | None = None,
    eval_env_or_factory: Any | None = None,
) -> TrainResult:
    """Train an inner RL agent with DiscoRL's meta-learned latent prediction update rule."""
    cfg = config or DiscoRLConfig()
    set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    env = ensure_gym_env(env_or_factory)

    agent = DiscoAgentNet(
        env.obs_dim, env.action_dim, latent_dim=cfg.latent_dim, hidden_dims=cfg.hidden_dims, is_discrete=env.is_discrete
    ).to(dev)
    rule = meta_rule.to(dev) if meta_rule is not None else DiscoMetaUpdateNet(
        latent_dim=cfg.latent_dim, meta_hidden=cfg.meta_hidden
    ).to(dev)

    agent_opt = torch.optim.Adam(agent.parameters(), lr=cfg.inner_lr)
    meta_opt = torch.optim.Adam(rule.parameters(), lr=cfg.meta_lr) if cfg.meta_train else None
    obs_rms = RunningMeanStd((env.obs_dim,)) if cfg.normalize_obs else None

    t0 = time.perf_counter()
    obs, info = env.reset(seed=cfg.seed)
    ep_ret, ep_idx, steps = 0.0, 0, 0
    saw_action_mask = False
    learning_curve: list[dict[str, float]] = []
    ent_hist, meta_loss_hist = [], []

    def _compute_inner_loss(
        params_dict: dict[str, torch.Tensor] | None,
        b_obs: torch.Tensor,
        b_next: torch.Tensor,
        b_act: torch.Tensor,
        b_rew: torch.Tensor,
        b_nt: torch.Tensor,
        b_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if params_dict is None:
            pol_out, y_curr = agent(b_obs)
            with torch.no_grad():
                _, y_next = agent(b_next)
        else:
            pol_out, y_curr = functional_call(agent, params_dict, (b_obs,))
            with torch.no_grad():
                _, y_next = functional_call(agent, params_dict, (b_next,))

        if env.is_discrete:
            logits = pol_out.masked_fill(~b_mask, -1e8) if b_mask is not None else pol_out
            dist: Any = Categorical(logits=logits)
            lp = dist.log_prob(b_act)
            ent = dist.entropy()
        else:
            logstd = params_dict["logstd"] if params_dict is not None else agent.logstd
            std = torch.exp(logstd.clamp(-5.0, 2.0)).expand_as(pol_out)
            dist = Normal(pol_out, std)
            lp = dist.log_prob(b_act).sum(-1)
            ent = dist.entropy().sum(-1)

        # [AI-Prompt Hook]: DiscoMetaUpdateNet generates policy advantage & latent prediction targets!
        pi_adv, y_target = rule(b_rew, b_nt, lp.detach(), y_curr.detach(), y_next.detach())
        if pi_adv.numel() > 1:
            pi_adv_norm = (pi_adv - pi_adv.mean()) / (pi_adv.std() + 1e-8)
        else:
            pi_adv_norm = pi_adv
        pg_loss = -(lp * pi_adv_norm).mean()
        pred_loss = F.mse_loss(y_curr, y_target.detach() if params_dict is None else y_target)
        return pg_loss + cfg.pred_coef * pred_loss - cfg.ent_coef * ent.mean(), ent.mean()

    prev_batch: tuple[torch.Tensor, ...] | None = None

    while steps < cfg.total_timesteps:
        buf_obs, buf_next, buf_act, buf_rew, buf_nt, buf_mask = [], [], [], [], [], []
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
                pol_out, _ = agent(obs_t)
                if env.is_discrete:
                    logits = pol_out.masked_fill(~mask_t, -1e8) if mask_t is not None else pol_out
                    act_env: Any = int(Categorical(logits=logits).sample().item())
                else:
                    std = torch.exp(agent.logstd.clamp(-5.0, 2.0)).expand_as(pol_out)
                    act_env = Normal(pol_out, std).sample().squeeze(0).cpu().numpy()

            next_obs, reward, term, trunc, next_info = env.step(act_env)
            ep_ret += float(reward)
            steps += 1
            next_proc = obs_rms.normalize(next_obs) if obs_rms is not None else next_obs

            buf_obs.append(obs_proc)
            buf_next.append(next_proc)
            buf_act.append(act_env)
            buf_rew.append(float(reward))
            buf_nt.append(0.0 if term else 1.0)
            if env.is_discrete:
                buf_mask.append(mask if mask is not None else np.ones(env.action_dim, dtype=np.bool_))

            if term or trunc:
                ep_idx += 1
                if ep_idx % 5 == 0:
                    learning_curve.append({"step": float(steps), "episode_return": float(ep_ret)})
                obs, info = env.reset(seed=cfg.seed + ep_idx)
                ep_ret = 0.0
            else:
                obs, info = next_obs, next_info

        if len(buf_obs) < 2:
            break

        b_obs = torch.as_tensor(np.asarray(buf_obs, dtype=np.float32), device=dev)
        b_next = torch.as_tensor(np.asarray(buf_next, dtype=np.float32), device=dev)
        b_act = torch.as_tensor(np.asarray(buf_act, dtype=np.int64 if env.is_discrete else np.float32), device=dev)
        b_rew = torch.as_tensor(np.asarray(buf_rew, dtype=np.float32), device=dev)
        b_nt = torch.as_tensor(np.asarray(buf_nt, dtype=np.float32), device=dev)
        b_mask = torch.as_tensor(np.asarray(buf_mask, dtype=np.bool_), device=dev) if env.is_discrete else None

        # Bi-level outer meta-gradient step evaluating how inner update affects held-out rollout loss
        if cfg.meta_train and meta_opt is not None and prev_batch is not None:
            p_obs, p_next, p_act, p_rew, p_nt, p_mask = prev_batch
            named_params = dict(agent.named_parameters())
            inner_loss, _ = _compute_inner_loss(named_params, p_obs, p_next, p_act, p_rew, p_nt, p_mask)
            inner_grads = torch.autograd.grad(inner_loss, list(named_params.values()), create_graph=True)
            fast_params = {k: v - cfg.inner_lr * g for (k, v), g in zip(named_params.items(), inner_grads)}

            # Evaluate updated fast_params on the fresh validation trajectory
            pol_val, _ = functional_call(agent, fast_params, (b_obs,))
            if env.is_discrete:
                logits_v = pol_val.masked_fill(~b_mask, -1e8) if b_mask is not None else pol_val
                lp_v = Categorical(logits=logits_v).log_prob(b_act)
            else:
                std_v = torch.exp(fast_params["logstd"].clamp(-5.0, 2.0)).expand_as(pol_val)
                lp_v = Normal(pol_val, std_v).log_prob(b_act).sum(-1)

            # Discounted empirical return signal on validation rollout for outer meta-gradient
            ret_v = torch.zeros_like(b_rew)
            g_ret = torch.tensor(0.0, device=dev)
            for t in range(len(b_rew) - 1, -1, -1):
                g_ret = b_rew[t] + 0.99 * b_nt[t] * g_ret
                ret_v[t] = g_ret
            ret_v = (ret_v - ret_v.mean()) / (ret_v.std() + 1e-8)
            outer_loss = -(lp_v * ret_v).mean()

            meta_opt.zero_grad()
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(rule.parameters(), 1.0)
            meta_opt.step()
            meta_loss_hist.append(float(outer_loss.item()))

        # Inner agent update using current DiscoMetaUpdateNet
        agent_loss, mean_ent = _compute_inner_loss(None, b_obs, b_next, b_act, b_rew, b_nt, b_mask)
        agent_opt.zero_grad()
        agent_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        agent_opt.step()
        ent_hist.append(float(mean_ent.item()))
        prev_batch = (b_obs, b_next, b_act, b_rew, b_nt, b_mask)

    if obs_rms is not None:
        obs_rms.freeze(True)

    def policy_fn(obs_in: np.ndarray, info_in: dict[str, Any] | None = None) -> Any:
        o = obs_rms.normalize(obs_in) if obs_rms is not None else np.asarray(obs_in, dtype=np.float32)
        obs_t = torch.as_tensor(o, device=dev, dtype=torch.float32).unsqueeze(0)
        with eval_mode(agent), torch.no_grad():
            pol_out, _ = agent(obs_t)
            if env.is_discrete:
                m = (info_in or {}).get("action_mask")
                if m is not None:
                    m_t = torch.as_tensor(m, device=dev, dtype=torch.bool).unsqueeze(0)
                    pol_out = pol_out.masked_fill(~m_t, -1e8)
                return int(pol_out.argmax(dim=-1).item())
            return pol_out.squeeze(0).cpu().numpy()

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
        model={"agent": agent, "meta_rule": rule},
        config=cfg_dict,
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
            "mean_meta_loss": float(np.mean(meta_loss_hist)) if meta_loss_hist else 0.0,
        },
        eval_stats=eval_stats,
        obs_rms_state=obs_rms.state_dict() if obs_rms is not None else None,
        obs_rms_frozen_on_eval=True if obs_rms is None else obs_rms.frozen,
        optimizations_used=[o.id for o in opts],
    )
