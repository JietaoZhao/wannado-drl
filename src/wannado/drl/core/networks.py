"""Shared lightweight PyTorch neural building blocks for CleanRL-style DRL algorithms."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal


def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization used across CleanRL PPO/A2C implementations."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    *,
    activation: type[nn.Module] = nn.Tanh,
    orthogonal_init: bool = False,
    last_std: float = 0.01,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    cur = in_dim
    for h in hidden_dims:
        lin = nn.Linear(cur, h)
        if orthogonal_init:
            layer_init(lin)
        layers.append(lin)
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        cur = h
    out = nn.Linear(cur, out_dim)
    if orthogonal_init:
        layer_init(out, std=last_std)
    layers.append(out)
    return nn.Sequential(*layers)


class DiscreteActorCritic(nn.Module):
    """Actor-Critic supporting optional invalid-action masking for Discrete action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        self.critic = build_mlp(obs_dim, hidden_dims, 1, activation=nn.Tanh, orthogonal_init=True, last_std=1.0)
        self.actor = build_mlp(obs_dim, hidden_dims, action_dim, activation=nn.Tanh, orthogonal_init=True, last_std=0.01)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_dist(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None) -> Categorical:
        logits = self.actor(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        return Categorical(logits=logits)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs, action_mask=action_mask)
        if action is None:
            action = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.get_value(obs)


def _compute_affine_action_bounds(
    action_dim: int,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(scale, bias)`` tensors centering/scaling finite Box action bounds."""
    scale = np.ones((action_dim,), dtype=np.float32)
    bias = np.zeros((action_dim,), dtype=np.float32)
    if action_low is not None and action_high is not None:
        lo = np.asarray(action_low, dtype=np.float32).reshape(-1)
        hi = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if lo.shape == (action_dim,) and hi.shape == (action_dim,):
            finite = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
            scale = np.where(finite, 0.5 * (hi - lo), 1.0).astype(np.float32)
            bias = np.where(finite, 0.5 * (hi + lo), 0.0).astype(np.float32)
    return torch.as_tensor(scale, dtype=torch.float32), torch.as_tensor(bias, dtype=torch.float32)


class ContinuousActorCritic(nn.Module):
    """Gaussian Actor-Critic for continuous Box action spaces (with optional bound-aware scaling)."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.critic = build_mlp(obs_dim, hidden_dims, 1, activation=nn.Tanh, orthogonal_init=True, last_std=1.0)
        self.actor_mean = build_mlp(obs_dim, hidden_dims, action_dim, activation=nn.Tanh, orthogonal_init=True, last_std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        scale_t, bias_t = _compute_affine_action_bounds(action_dim, action_low, action_high)
        self.register_buffer("action_scale", scale_t)
        self.register_buffer("action_bias", bias_t)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean = self.action_bias + self.action_scale * self.actor_mean(obs)
        std = self.action_scale * torch.exp(self.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        if action is None:
            action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, self.get_value(obs)


class RecurrentActorCriticGRU(nn.Module):
    """GRU-backed Actor-Critic for POMDP / partially observable tasks."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        is_discrete: bool = True,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.is_discrete = is_discrete
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        self.actor = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        if not is_discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        scale_t, bias_t = _compute_affine_action_bounds(action_dim, action_low, action_high)
        self.register_buffer("action_scale", scale_t)
        self.register_buffer("action_bias", bias_t)

    def forward_seq(
        self,
        obs_seq: torch.Tensor,
        h0: torch.Tensor,
        done_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """obs_seq: (T, B, obs_dim), h0: (1, B, H), done_seq: (T, B) resets state on episode end."""
        t_steps, batch_size, _ = obs_seq.shape
        x = self.encoder(obs_seq.reshape(-1, obs_seq.shape[-1])).view(t_steps, batch_size, self.hidden_dim)
        if done_seq is None:
            out, h_next = self.gru(x, h0)
            return out, h_next
        outputs = []
        h = h0
        for t in range(t_steps):
            if t > 0:
                mask = (1.0 - done_seq[t - 1]).view(1, batch_size, 1)
                h = h * mask
            step_out, h = self.gru(x[t : t + 1], h)
            outputs.append(step_out)
        return torch.cat(outputs, dim=0), h

    def get_dist_from_features(
        self,
        feats: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> Categorical | Normal:
        if self.is_discrete:
            logits = self.actor(feats)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.bool(), -1e8)
            return Categorical(logits=logits)
        mean = self.action_bias + self.action_scale * self.actor(feats)
        std = self.action_scale * torch.exp(self.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
        return Normal(mean, std)

    def step_action_value(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, h_next = self.forward_seq(obs.unsqueeze(0), h)
        feat = feat.squeeze(0)
        val = self.critic(feat).squeeze(-1)
        if self.is_discrete:
            logits = self.actor(feat)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.bool(), -1e8)
            disc_dist = Categorical(logits=logits)
            if action is None:
                action = logits.argmax(dim=-1) if deterministic else disc_dist.sample()
            return action, disc_dist.log_prob(action), disc_dist.entropy(), val, h_next
        mean = self.action_bias + self.action_scale * self.actor(feat)
        std = self.action_scale * torch.exp(self.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
        cont_dist = Normal(mean, std)
        if action is None:
            action = mean if deterministic else cont_dist.sample()
        return action, cont_dist.log_prob(action).sum(-1), cont_dist.entropy().sum(-1), val, h_next


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, action_dim, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network: Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.adv_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.feature(obs)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + (a - a.mean(dim=-1, keepdim=True))


class CategoricalQNetwork(nn.Module):
    """C51 Distributional Q-Network outputting logits over n_atoms per action."""

    atoms: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        self.net = build_mlp(obs_dim, hidden_dims, action_dim * n_atoms, activation=nn.ReLU)

    def get_probs(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs).view(-1, self.action_dim, self.n_atoms)
        return F.softmax(logits, dim=-1)

    def get_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        probs = self.get_probs(obs)
        return torch.sum(probs * self.atoms, dim=-1)


class DeterministicActor(nn.Module):
    """Bounded continuous deterministic actor mu(s) for DDPG / TD3 / MADDPG."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, action_dim, activation=nn.ReLU)
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.action_scale + self.action_bias


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor with exact Jacobian log-prob correction for SAC / HIL-SERL / CQL."""

    LOG_STD_MAX = 2.0
    LOG_STD_MIN = -5.0
    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.trunk = build_mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1], activation=nn.ReLU)
        last_h = hidden_dims[-1]
        self.mean_head = nn.Linear(last_h, action_dim)
        self.log_std_head = nn.Linear(last_h, action_dim)
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.trunk(obs))
        mean = self.mean_head(h)
        log_std = torch.tanh(self.log_std_head(h))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = mean if deterministic else dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t) - torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class TwinQCritic(nn.Module):
    """Clipped Double Q-Critic with optional LayerNorm (essential for RLPD / HIL-SERL stability)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.q1 = build_mlp(obs_dim + action_dim, hidden_dims, 1, activation=nn.ReLU, layer_norm=use_layer_norm)
        self.q2 = build_mlp(obs_dim + action_dim, hidden_dims, 1, activation=nn.ReLU, layer_norm=use_layer_norm)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))


class QMixMixer(nn.Module):
    """QMIX monotonic hypernetwork mixer ensuring d(Q_tot)/d(Q_i) >= 0 (Rashid et al., ICML 2018)."""

    def __init__(self, num_agents: int, state_dim: int, mixing_embed_dim: int = 32) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = mixing_embed_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, num_agents * mixing_embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """agent_qs: (B, N_agents), states: (B, state_dim) -> Q_tot: (B,)."""
        b = agent_qs.size(0)
        q_in = agent_qs.view(b, 1, self.num_agents)
        w1 = torch.abs(self.hyper_w1(states)).view(b, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(b, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(q_in, w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(b, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(b, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(b)


class AutoregressiveSeqPolicy(nn.Module):
    """Lightweight Causal Transformer autoregressive token policy for GRPO / DAPO / GSPO / ARPO."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 48,
        hidden_dim: int = 64,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        n_heads = 4 if embed_dim % 4 == 0 else (2 if embed_dim % 2 == 0 else 1)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        out = self.encoder(x, mask=causal_mask, is_causal=True)
        return self.lm_head(self.ln(out))

    @torch.no_grad()
    def generate(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively sample response tokens and return (responses, old_token_logprobs)."""
        cur = prompts
        gen_tokens: list[torch.Tensor] = []
        gen_logprobs: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            logits = self.forward_logits(cur)[:, -1, :] / max(temperature, 1e-4)
            dist = Categorical(logits=logits)
            tok = logits.argmax(dim=-1) if deterministic else dist.sample()
            lp = dist.log_prob(tok)
            gen_tokens.append(tok.unsqueeze(1))
            gen_logprobs.append(lp.unsqueeze(1))
            cur = torch.cat([cur, tok.unsqueeze(1)], dim=1)
        return torch.cat(gen_tokens, dim=1), torch.cat(gen_logprobs, dim=1)

    def sequence_logprobs_and_entropy(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log-probs and entropies for generated response positions."""
        full_seq = torch.cat([prompts, responses], dim=1)
        logits = self.forward_logits(full_seq[:, :-1])
        resp_logits = logits[:, prompts.shape[1] - 1 :, :]
        dist = Categorical(logits=resp_logits)
        logprobs = dist.log_prob(responses)
        entropy = dist.entropy()
        return logprobs, entropy
