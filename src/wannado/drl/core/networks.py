"""Shared PyTorch neural building blocks and multimodal feature extractors for wannado-drl."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Distribution, Normal
from torch.distributions.kl import kl_divergence, register_kl

from wannado.drl.core.obs_spec import LeafObsSpec, ObsSpec, vector_obs_spec


def _resolve_obs_spec(obs_dim_or_spec: int | ObsSpec) -> ObsSpec:
    if isinstance(obs_dim_or_spec, ObsSpec):
        return obs_dim_or_spec
    return vector_obs_spec(int(obs_dim_or_spec))


class IdentityFlattenExtractor(nn.Module):
    """Zero-parameter passthrough flattening 1D/numeric inputs (preserves v1.0.5 MLP layouts)."""

    is_identity: bool = True

    def __init__(self, obs_spec: ObsSpec) -> None:
        super().__init__()
        self.obs_spec = obs_spec
        self.output_dim = int(obs_spec.flat_dim)

    def forward(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, dict):
            parts = [obs[k].float().flatten(start_dim=1) for k in self.obs_spec.leaves if k in obs]
            return torch.cat(parts, dim=-1)
        t = obs.float()
        if t.ndim == 1:
            return t.unsqueeze(0)
        return t.flatten(start_dim=1)


class NatureCNNExtractor(nn.Module):
    """Mnih et al. (2015) CNN backbone with adaptive small-grid fallback (e.g. MinAtar 10x10)."""

    is_identity: bool = False

    def __init__(self, obs_shape_chw: tuple[int, int, int], features_dim: int = 256) -> None:
        super().__init__()
        c, h, w = (int(obs_shape_chw[0]), int(obs_shape_chw[1]), int(obs_shape_chw[2]))
        if min(h, w) >= 36:
            convs: list[nn.Module] = [
                layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU(),
            ]
        elif min(h, w) >= 12:
            convs = [
                layer_init(nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU(),
            ]
        else:
            convs = [
                layer_init(nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1)),
                nn.ReLU(),
            ]
        self.cnn = nn.Sequential(*convs, nn.Flatten())
        with torch.no_grad():
            n_flat = int(self.cnn(torch.zeros(1, c, h, w)).shape[1])
        self.proj = nn.Sequential(layer_init(nn.Linear(n_flat, int(features_dim))), nn.ReLU())
        self.output_dim = int(features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        return self.proj(self.cnn(x))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1), std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(F.relu(x)))
        return x + self.conv2(out)


class ImpalaResNetExtractor(nn.Module):
    """Espeholt et al. (ICML 2018) IMPALA residual CNN stack with adaptive spatial pooling."""

    is_identity: bool = False

    def __init__(
        self,
        obs_shape_chw: tuple[int, int, int],
        channels: tuple[int, ...] = (16, 32, 32),
        features_dim: int = 256,
    ) -> None:
        super().__init__()
        c, h, w = (int(obs_shape_chw[0]), int(obs_shape_chw[1]), int(obs_shape_chw[2]))
        use_pool = min(h, w) >= 24
        stages_ch = channels if min(h, w) >= 24 else channels[:2]
        blocks: list[nn.Module] = []
        in_c = c
        for out_c in stages_ch:
            blocks.append(layer_init(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)))
            if use_pool:
                blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            blocks.append(_ResidualBlock(out_c))
            blocks.append(_ResidualBlock(out_c))
            in_c = out_c
        self.stem = nn.Sequential(*blocks, nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) if use_pool else nn.Identity()
        with torch.no_grad():
            n_flat = int(torch.flatten(self.pool(self.stem(torch.zeros(1, c, h, w))), 1).shape[1])
        self.head = nn.Sequential(nn.Flatten(), layer_init(nn.Linear(n_flat, int(features_dim))), nn.ReLU())
        self.output_dim = int(features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.stem(obs.float())))


class SetAttentionExtractor(nn.Module):
    """Permutation-invariant Multi-Head Self-Attention + Masked Pooling for ``(B, N, F)`` entity sets."""

    is_identity: bool = False

    def __init__(
        self,
        in_features: int,
        *,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        features_dim: int = 64,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        resolved_heads = num_heads if embed_dim % num_heads == 0 else (2 if embed_dim % 2 == 0 else 1)
        self.input_proj = nn.Sequential(
            layer_init(nn.Linear(int(in_features), embed_dim)),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=resolved_heads,
            dim_feedforward=2 * embed_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=max(1, int(num_layers)), enable_nested_tensor=False
        )
        self.pooling = pooling
        self.attn_score = layer_init(nn.Linear(embed_dim, 1)) if pooling == "attention" else None
        self.out_proj = nn.Sequential(
            layer_init(nn.Linear(embed_dim, int(features_dim))),
            nn.LayerNorm(int(features_dim)),
            nn.ReLU(),
        )
        self.output_dim = int(features_dim)

    def forward(self, obs: torch.Tensor, entity_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = obs.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        valid_mask: torch.Tensor | None = None
        if entity_mask is not None:
            valid_mask = entity_mask.bool()
            if valid_mask.ndim == 1:
                valid_mask = valid_mask.unsqueeze(0)
            # Guard against all-False rows causing NaNs inside MultiheadAttention softmax
            any_valid = valid_mask.any(dim=-1, keepdim=True)
            safe_valid = valid_mask.clone()
            safe_valid[:, :1] = torch.where(any_valid, safe_valid[:, :1], torch.ones_like(safe_valid[:, :1]))
            h = self.encoder(h, src_key_padding_mask=~safe_valid)
            h = h * valid_mask.unsqueeze(-1).to(h.dtype)
        else:
            h = self.encoder(h)

        if self.attn_score is not None:
            scores = self.attn_score(h).squeeze(-1)
            if valid_mask is not None:
                any_valid = valid_mask.any(dim=-1, keepdim=True)
                scores = scores.masked_fill(~valid_mask, -1e9)
                weights = torch.softmax(scores, dim=-1) * any_valid.to(scores.dtype)
            else:
                weights = torch.softmax(scores, dim=-1)
            pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        elif valid_mask is not None:
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
            pooled = h.sum(dim=1) / denom
        else:
            pooled = h.mean(dim=1)
        return self.out_proj(pooled)


class DictCombinedExtractor(nn.Module):
    """Per-key multimodal extractor combining CNN, SetAttention, and Vector encoders for Dict obs."""

    is_identity: bool = False

    def __init__(
        self,
        obs_spec: ObsSpec,
        *,
        cnn_extractor: str = "nature_cnn",
        features_dim: int | None = None,
        cnn_features_dim: int = 128,
        entity_features_dim: int = 64,
        vector_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_spec = obs_spec
        self.mask_keys = {lf.mask_key for lf in obs_spec.leaves.values() if lf.mask_key is not None}
        self.extractors = nn.ModuleDict()
        self.mask_links: dict[str, str] = {}
        total_dim = 0

        for key, leaf in obs_spec.leaves.items():
            if key in self.mask_keys:
                continue
            sub_mod, sub_dim = _build_leaf_extractor(
                leaf,
                cnn_extractor=cnn_extractor,
                cnn_features_dim=cnn_features_dim,
                entity_features_dim=entity_features_dim,
                vector_hidden_dim=vector_hidden_dim,
            )
            self.extractors[key] = sub_mod
            if leaf.mask_key is not None:
                self.mask_links[key] = leaf.mask_key
            total_dim += sub_dim

        if total_dim == 0:
            raise ValueError("DictCombinedExtractor found zero non-mask observation leaves.")
        if features_dim is not None:
            self.fusion: nn.Module = nn.Sequential(
                layer_init(nn.Linear(total_dim, int(features_dim))),
                nn.LayerNorm(int(features_dim)),
                nn.ReLU(),
            )
            self.output_dim = int(features_dim)
        else:
            self.fusion = nn.Identity()
            self.output_dim = total_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        feats: list[torch.Tensor] = []
        for key, mod in self.extractors.items():
            val = obs[key]
            mkey = self.mask_links.get(key)
            if mkey is not None and mkey in obs and isinstance(mod, SetAttentionExtractor):
                feats.append(mod(val, entity_mask=obs[mkey]))
            else:
                feats.append(mod(val))
        return self.fusion(torch.cat(feats, dim=-1))


class _VectorBranchExtractor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.raw_passthrough = in_dim <= 32
        if self.raw_passthrough:
            self.net: nn.Module = nn.Identity()
            self.output_dim = in_dim
        else:
            self.net = nn.Sequential(layer_init(nn.Linear(in_dim, hidden_dim)), nn.LayerNorm(hidden_dim), nn.ReLU())
            self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float().flatten(start_dim=1))


def _build_leaf_extractor(
    leaf: LeafObsSpec,
    *,
    cnn_extractor: str = "nature_cnn",
    cnn_features_dim: int = 256,
    entity_features_dim: int = 64,
    vector_hidden_dim: int = 64,
) -> tuple[nn.Module, int]:
    if leaf.kind == "image":
        chw = (int(leaf.shape[0]), int(leaf.shape[1]), int(leaf.shape[2]))
        if cnn_extractor == "impala_cnn":
            mod: nn.Module = ImpalaResNetExtractor(chw, features_dim=cnn_features_dim)
        else:
            mod = NatureCNNExtractor(chw, features_dim=cnn_features_dim)
        return mod, cnn_features_dim
    if leaf.kind == "entities":
        mod = SetAttentionExtractor(int(leaf.shape[-1]), features_dim=entity_features_dim)
        return mod, entity_features_dim
    vmod = _VectorBranchExtractor(leaf.flat_dim, hidden_dim=vector_hidden_dim)
    return vmod, vmod.output_dim


def build_feature_extractor(
    obs_spec: ObsSpec,
    *,
    extractor_type: str = "auto",
    extractor_kwargs: dict[str, Any] | None = None,
) -> tuple[nn.Module, int]:
    """Build a compatible feature extractor and return ``(module, output_dim)``."""
    kw = dict(extractor_kwargs or {})
    etype = str(extractor_type).lower()
    if etype == "flatten" or (etype == "auto" and obs_spec.kind == "vector"):
        ident = IdentityFlattenExtractor(obs_spec)
        return ident, ident.output_dim
    if obs_spec.kind == "image" or etype in ("nature_cnn", "impala_cnn", "small_cnn", "cnn"):
        leaf = next(iter(obs_spec.leaves.values()))
        shape_3d = leaf.shape if len(leaf.shape) == 3 else (1, int(leaf.shape[0]), int(leaf.shape[1]))
        chw = (int(shape_3d[0]), int(shape_3d[1]), int(shape_3d[2]))
        fdim = int(kw.pop("features_dim", 256))
        if etype == "impala_cnn":
            mod_cnn: nn.Module = ImpalaResNetExtractor(chw, features_dim=fdim, **kw)
        else:
            mod_cnn = NatureCNNExtractor(chw, features_dim=fdim)
        return mod_cnn, fdim
    if obs_spec.kind == "entities" or etype == "attention":
        leaf = next(iter(obs_spec.leaves.values()))
        in_f = int(leaf.shape[-1])
        fdim = int(kw.pop("features_dim", 64))
        mod_attn = SetAttentionExtractor(in_f, features_dim=fdim, **kw)
        return mod_attn, fdim
    if obs_spec.kind == "dict" or etype == "dict":
        mod_dict = DictCombinedExtractor(obs_spec, **kw)
        return mod_dict, mod_dict.output_dim
    if etype == "mlp":
        fdim = int(kw.pop("features_dim", 64))
        mlp_mod = _VectorBranchExtractor(obs_spec.flat_dim, hidden_dim=fdim)
        return mlp_mod, mlp_mod.output_dim
    ident = IdentityFlattenExtractor(obs_spec)
    return ident, ident.output_dim


def layer_init(layer: Any, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> Any:
    """Orthogonal initialization used across CleanRL PPO/A2C/CNN implementations."""
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
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
    """Actor-Critic supporting optional invalid-action masking and multimodal encoders."""

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.critic = build_mlp(feat_dim, hidden_dims, 1, activation=nn.Tanh, orthogonal_init=True, last_std=1.0)
        self.actor = build_mlp(feat_dim, hidden_dims, action_dim, activation=nn.Tanh, orthogonal_init=True, last_std=0.01)

    def _encode(self, obs: Any) -> torch.Tensor:
        return self.features_extractor(obs)

    def actor_parameters(self) -> list[nn.Parameter]:
        if getattr(self.features_extractor, "is_identity", True):
            return list(self.actor.parameters())
        return list(self.features_extractor.parameters()) + list(self.actor.parameters())

    def get_value(self, obs: Any) -> torch.Tensor:
        return self.critic(self._encode(obs)).squeeze(-1)

    def get_dist(self, obs: Any, action_mask: torch.Tensor | None = None) -> Categorical:
        logits = self.actor(self._encode(obs))
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        return Categorical(logits=logits)

    def get_action_and_value(
        self,
        obs: Any,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._encode(obs)
        logits = self.actor(feat)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feat).squeeze(-1)


_EMPTY_SAMPLE_SHAPE = torch.Size()


class MultiCategorical(Distribution):
    """Factorized product of Categorical distributions over MultiDiscrete / MultiBinary branches."""

    has_enumerate_support = False

    def __init__(self, logits_list: Sequence[torch.Tensor]) -> None:
        self.dists = [Categorical(logits=logits) for logits in logits_list]
        batch_shape = self.dists[0].batch_shape if self.dists else torch.Size()
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size((len(self.dists),)), validate_args=False)

    @property
    def logits_list(self) -> list[torch.Tensor]:
        return [d.logits for d in self.dists]

    def detached_copy(self) -> MultiCategorical:
        return MultiCategorical([d.logits.detach() for d in self.dists])

    def sample(
        self,
        sample_shape: torch.Size | list[int] | tuple[int, ...] = _EMPTY_SAMPLE_SHAPE,
    ) -> torch.Tensor:
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)

    def argmax_action(self) -> torch.Tensor:
        return torch.stack([d.logits.argmax(dim=-1) for d in self.dists], dim=-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.stack([d.log_prob(value[..., i]) for i, d in enumerate(self.dists)], dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)


@register_kl(MultiCategorical, MultiCategorical)
def _kl_multicategorical(p: MultiCategorical, q: MultiCategorical) -> torch.Tensor:
    return torch.stack([kl_divergence(dp, dq) for dp, dq in zip(p.dists, q.dists, strict=True)], dim=-1).sum(dim=-1)


class MultiDiscreteActorCritic(nn.Module):
    """Factorized Actor-Critic supporting MultiDiscrete/MultiBinary spaces and multimodal encoders."""

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_nvec: Sequence[int],
        hidden_dims: Sequence[int] = (64, 64),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.action_nvec = tuple(int(v) for v in action_nvec)
        total_logits = int(sum(self.action_nvec))
        self.critic = build_mlp(feat_dim, hidden_dims, 1, activation=nn.Tanh, orthogonal_init=True, last_std=1.0)
        self.actor = build_mlp(feat_dim, hidden_dims, total_logits, activation=nn.Tanh, orthogonal_init=True, last_std=0.01)

    def _encode(self, obs: Any) -> torch.Tensor:
        return self.features_extractor(obs)

    def actor_parameters(self) -> list[nn.Parameter]:
        if getattr(self.features_extractor, "is_identity", True):
            return list(self.actor.parameters())
        return list(self.features_extractor.parameters()) + list(self.actor.parameters())

    def get_value(self, obs: Any) -> torch.Tensor:
        return self.critic(self._encode(obs)).squeeze(-1)

    def get_dist(self, obs: Any, action_mask: torch.Tensor | None = None) -> MultiCategorical:
        logits = self.actor(self._encode(obs))
        if action_mask is not None and action_mask.shape[-1] == logits.shape[-1]:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        splits = torch.split(logits, list(self.action_nvec), dim=-1)
        return MultiCategorical(splits)

    def get_action_and_value(
        self,
        obs: Any,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._encode(obs)
        logits = self.actor(feat)
        if action_mask is not None and action_mask.shape[-1] == logits.shape[-1]:
            logits = logits.masked_fill(~action_mask.bool(), -1e8)
        dist = MultiCategorical(torch.split(logits, list(self.action_nvec), dim=-1))
        if action is None:
            action = dist.argmax_action() if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(feat).squeeze(-1)


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
    """Gaussian Actor-Critic for continuous Box spaces with multimodal encoders."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.critic = build_mlp(feat_dim, hidden_dims, 1, activation=nn.Tanh, orthogonal_init=True, last_std=1.0)
        self.actor_mean = build_mlp(feat_dim, hidden_dims, action_dim, activation=nn.Tanh, orthogonal_init=True, last_std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        scale_t, bias_t = _compute_affine_action_bounds(action_dim, action_low, action_high)
        self.register_buffer("action_scale", scale_t)
        self.register_buffer("action_bias", bias_t)

    def _encode(self, obs: Any) -> torch.Tensor:
        return self.features_extractor(obs)

    def actor_parameters(self) -> list[nn.Parameter]:
        params = list(self.actor_mean.parameters()) + [self.actor_logstd]
        if getattr(self.features_extractor, "is_identity", True):
            return params
        return list(self.features_extractor.parameters()) + params

    def get_value(self, obs: Any) -> torch.Tensor:
        return self.critic(self._encode(obs)).squeeze(-1)

    def get_dist(self, obs: Any) -> Normal:
        mean = self.action_bias + self.action_scale * self.actor_mean(self._encode(obs))
        std = self.action_scale * torch.exp(self.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(
        self,
        obs: Any,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self._encode(obs)
        mean = self.action_bias + self.action_scale * self.actor_mean(feat)
        std = self.action_scale * torch.exp(self.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, self.critic(feat).squeeze(-1)


class RecurrentActorCriticGRU(nn.Module):
    """GRU-backed Actor-Critic for POMDP / partially observable tasks with multimodal encoders."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        hidden_dim: int = 64,
        is_discrete: bool = True,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        action_nvec: Sequence[int] | None = None,
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.is_discrete = is_discrete
        self.action_nvec = tuple(int(v) for v in action_nvec) if action_nvec is not None else None
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(layer_init(nn.Linear(feat_dim, hidden_dim)), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        out_dim = int(sum(self.action_nvec)) if (self.action_nvec is not None and not is_discrete) else action_dim
        self.actor = layer_init(nn.Linear(hidden_dim, out_dim), std=0.01)
        if not is_discrete and self.action_nvec is None:
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        scale_t, bias_t = _compute_affine_action_bounds(action_dim, action_low, action_high)
        self.register_buffer("action_scale", scale_t)
        self.register_buffer("action_bias", bias_t)

    def forward_seq(
        self,
        obs_seq: Any,
        h0: torch.Tensor,
        done_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """obs_seq: (T, B, *obs_shape) or dict[(T, B, ...)], h0: (1, B, H), done_seq: (T, B)."""
        if isinstance(obs_seq, dict):
            first = next(iter(obs_seq.values()))
            t_steps, batch_size = int(first.shape[0]), int(first.shape[1])
            flat_obs: Any = {k: v.reshape(t_steps * batch_size, *v.shape[2:]) for k, v in obs_seq.items()}
        else:
            t_steps, batch_size = int(obs_seq.shape[0]), int(obs_seq.shape[1])
            flat_obs = obs_seq.reshape(t_steps * batch_size, *obs_seq.shape[2:])
        encoded = self.features_extractor(flat_obs)
        x = self.encoder(encoded).view(t_steps, batch_size, self.hidden_dim)
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
    ) -> Categorical | MultiCategorical | Normal:
        if self.is_discrete:
            logits = self.actor(feats)
            if action_mask is not None:
                logits = logits.masked_fill(~action_mask.bool(), -1e8)
            return Categorical(logits=logits)
        if self.action_nvec is not None:
            logits = self.actor(feats)
            if action_mask is not None and action_mask.shape[-1] == logits.shape[-1]:
                logits = logits.masked_fill(~action_mask.bool(), -1e8)
            return MultiCategorical(torch.split(logits, list(self.action_nvec), dim=-1))
        mean = self.action_bias + self.action_scale * self.actor(feats)
        std = self.action_scale * torch.exp(self.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
        return Normal(mean, std)

    def step_action_value(
        self,
        obs: Any,
        h: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_seq: Any = {k: v.unsqueeze(0) for k, v in obs.items()} if isinstance(obs, dict) else obs.unsqueeze(0)
        feat, h_next = self.forward_seq(obs_seq, h)
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
        if self.action_nvec is not None:
            logits = self.actor(feat)
            if action_mask is not None and action_mask.shape[-1] == logits.shape[-1]:
                logits = logits.masked_fill(~action_mask.bool(), -1e8)
            md_dist = MultiCategorical(torch.split(logits, list(self.action_nvec), dim=-1))
            if action is None:
                action = md_dist.argmax_action() if deterministic else md_dist.sample()
            return action, md_dist.log_prob(action), md_dist.entropy(), val, h_next
        mean = self.action_bias + self.action_scale * self.actor(feat)
        std = self.action_scale * torch.exp(self.actor_logstd.clamp(-5.0, 2.0)).expand_as(mean)
        cont_dist = Normal(mean, std)
        if action is None:
            action = mean if deterministic else cont_dist.sample()
        return action, cont_dist.log_prob(action).sum(-1), cont_dist.entropy().sum(-1), val, h_next


class QNetwork(nn.Module):
    """Discrete Q-Network supporting 1D vectors as well as CNN/Dict/Attention feature extractors."""

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.net = build_mlp(feat_dim, hidden_dims, action_dim, activation=nn.ReLU)

    def forward(self, obs: Any) -> torch.Tensor:
        return self.net(self.features_extractor(obs))


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network: Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))."""

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.feature = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.adv_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, obs: Any) -> torch.Tensor:
        h = self.feature(self.features_extractor(obs))
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + (a - a.mean(dim=-1, keepdim=True))


class CategoricalQNetwork(nn.Module):
    """C51 Distributional Q-Network outputting logits over n_atoms per action."""

    atoms: torch.Tensor

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        self.net = build_mlp(feat_dim, hidden_dims, action_dim * n_atoms, activation=nn.ReLU)

    def get_logits(self, obs: Any) -> torch.Tensor:
        return self.net(self.features_extractor(obs)).view(-1, self.action_dim, self.n_atoms)

    def get_probs(self, obs: Any) -> torch.Tensor:
        return F.softmax(self.get_logits(obs), dim=-1)

    def get_q_values(self, obs: Any) -> torch.Tensor:
        probs = self.get_probs(obs)
        return torch.sum(probs * self.atoms, dim=-1)


class DeterministicActor(nn.Module):
    """Bounded continuous deterministic actor mu(s) for DDPG / TD3 / MADDPG."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.net = build_mlp(feat_dim, hidden_dims, action_dim, activation=nn.ReLU)
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, obs: Any) -> torch.Tensor:
        return torch.tanh(self.net(self.features_extractor(obs))) * self.action_scale + self.action_bias


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor with exact Jacobian log-prob correction for SAC / HIL-SERL / CQL."""

    LOG_STD_MAX = 2.0
    LOG_STD_MIN = -5.0
    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.trunk = build_mlp(feat_dim, hidden_dims[:-1], hidden_dims[-1], activation=nn.ReLU)
        last_h = hidden_dims[-1]
        self.mean_head = nn.Linear(last_h, action_dim)
        self.log_std_head = nn.Linear(last_h, action_dim)
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.trunk(self.features_extractor(obs)))
        mean = self.mean_head(h)
        log_std = torch.tanh(self.log_std_head(h))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(
        self,
        obs: Any,
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


class SingleQCritic(nn.Module):
    """Single continuous Q-Critic Q(s, a) with optional multimodal encoder (for DDPG)."""

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.q = build_mlp(feat_dim + action_dim, hidden_dims, 1, activation=nn.ReLU)

    def forward(self, obs: Any, action: torch.Tensor) -> torch.Tensor:
        feat = self.features_extractor(obs)
        return self.q(torch.cat([feat, action], dim=-1))


class TwinQCritic(nn.Module):
    """Clipped Double Q-Critic with optional LayerNorm and multimodal encoder."""

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        use_layer_norm: bool = False,
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.q1 = build_mlp(feat_dim + action_dim, hidden_dims, 1, activation=nn.ReLU, layer_norm=use_layer_norm)
        self.q2 = build_mlp(feat_dim + action_dim, hidden_dims, 1, activation=nn.ReLU, layer_norm=use_layer_norm)

    def forward(self, obs: Any, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features_extractor(obs)
        sa = torch.cat([feat, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, obs: Any, action: torch.Tensor) -> torch.Tensor:
        feat = self.features_extractor(obs)
        return self.q1(torch.cat([feat, action], dim=-1))


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


class ValueNetwork(nn.Module):
    """Standalone state value network V(s) with optional multimodal feature extractor (for IQL)."""

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.v = build_mlp(feat_dim, hidden_dims, 1, activation=nn.ReLU)

    def forward(self, obs: Any) -> torch.Tensor:
        return self.v(self.features_extractor(obs))


class DiscretePolicyNet(nn.Module):
    """Standalone discrete action-logit policy network with optional multimodal feature extractor (for IQL)."""

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.net = build_mlp(feat_dim, hidden_dims, action_dim, activation=nn.ReLU)

    def forward(self, obs: Any) -> torch.Tensor:
        return self.net(self.features_extractor(obs))


class ContinuousGaussianActor(nn.Module):
    """Bounded continuous Gaussian actor with learnable log-std (for IQL continuous AWR)."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim_or_spec: int | ObsSpec,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: Sequence[int] = (128, 128),
        *,
        extractor_type: str = "auto",
        extractor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_obs_spec(obs_dim_or_spec)
        self.features_extractor, feat_dim = build_feature_extractor(
            spec, extractor_type=extractor_type, extractor_kwargs=extractor_kwargs
        )
        self.mean_net = build_mlp(feat_dim, hidden_dims, action_dim, activation=nn.Tanh)
        self.logstd = nn.Parameter(torch.zeros(1, action_dim))
        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def get_dist(self, obs: Any) -> Normal:
        mean = torch.tanh(self.mean_net(self.features_extractor(obs))) * self.action_scale + self.action_bias
        std = self.action_scale * torch.exp(self.logstd.clamp(-5.0, 2.0)).expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs: Any) -> torch.Tensor:
        return torch.tanh(self.mean_net(self.features_extractor(obs))) * self.action_scale + self.action_bias
