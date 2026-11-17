"""Shared utilities: seeding, device resolution, statistical aggregation (IQM), and GAE."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import nn


def set_global_seed(seed: int, deterministic_torch: bool = False) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch RNGs consistently for reproducible RL experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(False)
    return np.random.default_rng(seed)


def resolve_device(device: str | torch.device = "auto", *, prefer_mps: bool = False) -> torch.device:
    """Resolve 'auto' to cuda -> cpu (avoiding MPS tiny-batch kernel overhead on vector DRL).

    Explicit ``device='mps'`` or ``prefer_mps=True`` still selects Apple Silicon MPS for large vision/seq nets.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@contextmanager
def eval_mode(model: Any) -> Iterator[None]:
    """Temporarily set an nn.Module (or list/dict of modules) to eval() and restore afterward."""
    modules: list[nn.Module] = []
    if isinstance(model, nn.Module):
        modules = [model]
    elif isinstance(model, dict):
        modules = [v for v in model.values() if isinstance(v, nn.Module)]
    elif isinstance(model, (list, tuple)):
        modules = [v for v in model if isinstance(v, nn.Module)]

    prev_states = [m.training for m in modules]
    for m in modules:
        m.eval()
    try:
        yield
    finally:
        for m, was_train in zip(modules, prev_states):
            m.train(was_train)


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Soft-update target network parameters: target <- (1 - tau) * target + tau * source."""
    with torch.no_grad():
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.mul_(1.0 - tau).add_(src_p.data, alpha=tau)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    """Copy parameters verbatim from source module to target module."""
    target.load_state_dict(source.state_dict())


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute 1 - Var(y_true - y_pred) / Var(y_true) to diagnose critic value fit."""
    var_y = float(np.var(y_true))
    if var_y < 1e-12:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def compute_iqm(values: Sequence[float]) -> float:
    """Compute Interquartile Mean (IQM) per Agarwal et al. (NeurIPS 2021).

    Drops the bottom 25% and top 25% of seed scores (with fractional interpolation
    on small sample sizes) to resist both lucky outliers and catastrophic seed outliers.
    """
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = len(arr)
    if n == 0:
        return 0.0
    if n <= 2:
        return float(np.mean(arr))
    low = 0.25 * n
    high = 0.75 * n
    total = 0.0
    weight_sum = 0.0
    for idx, val in enumerate(arr):
        left = max(float(idx), low)
        right = min(float(idx + 1), high)
        if right > left:
            w = right - left
            total += w * float(val)
            weight_sum += w
    return float(total / max(weight_sum, 1e-12))


def aggregate_seed_stats(values: Sequence[float]) -> dict[str, float]:
    """Return mean, std, median, min, max, and IQM over multi-seed scores."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "iqm": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "iqm": compute_iqm(arr.tolist()),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminateds: np.ndarray,
    next_value: float | np.ndarray,
    next_terminated: bool | np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    dones: np.ndarray | None = None,
    next_done: bool | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (Schulman et al., ICLR 2016).

    CRITICAL MDP BOUNDARY RULE:
    When ``dones`` (``term or trunc``) is passed with TimeLimit-bootstrapped rewards
    (``r + gamma * V(next_obs)`` on ``trunc and not term``), GAE cuts recursion cleanly at
    every episode boundary without leaking next-episode returns backward across resets.
    """
    num_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros_like(rewards[0], dtype=np.float32)
    boundaries = terminateds if dones is None else dones
    end_boundary = next_terminated if next_done is None else next_done

    for t in reversed(range(num_steps)):
        b_t = np.asarray(boundaries[t], dtype=np.float32)
        if t == num_steps - 1:
            b_t = np.maximum(b_t, np.asarray(end_boundary, dtype=np.float32))
            next_v_t = np.asarray(next_value, dtype=np.float32)
        else:
            next_v_t = values[t + 1]
        nonterminal_t = 1.0 - b_t
        delta = rewards[t] + gamma * next_v_t * nonterminal_t - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal_t * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def discretize_observation(
    obs: np.ndarray,
    *,
    is_discrete_obs: bool = False,
    bins: int = 8,
    clip_range: float = 4.0,
) -> tuple[int, ...]:
    """Convert either a discrete or continuous flat observation vector into a hashable tabular state key."""
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if is_discrete_obs:
        return tuple(round(float(x)) for x in arr)
    clipped = np.clip(arr, -clip_range, clip_range)
    edges = np.linspace(-clip_range, clip_range, bins + 1)[1:-1]
    return tuple(int(np.digitize(float(v), edges)) for v in clipped)

