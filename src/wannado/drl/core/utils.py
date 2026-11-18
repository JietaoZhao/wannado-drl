"""Shared utilities: seeding, device resolution, statistical aggregation (IQM), and GAE."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
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


def _iqm_weights(n: int) -> np.ndarray:
    """Return exact sorted-index IQM weights for sample count n."""
    if n <= 2:
        return np.full(max(1, n), 1.0 / max(1, n), dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    left = np.maximum(idx, 0.25 * n)
    right = np.minimum(idx + 1.0, 0.75 * n)
    w = np.clip(right - left, 0.0, 1.0)
    return w / max(float(np.sum(w)), 1e-12)


def bootstrap_iqm_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute IQM with stratified percentile bootstrap confidence interval (Agarwal et al., NeurIPS 2021)."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = compute_iqm(arr.tolist())
    if n == 1:
        return point, point, point

    rng = np.random.default_rng(seed)
    b = max(100, int(n_bootstrap))
    idx_mat = rng.integers(0, n, size=(b, n))
    sorted_boot = np.sort(arr[idx_mat], axis=1)
    boot_iqms = np.sum(sorted_boot * _iqm_weights(n)[None, :], axis=1)
    alpha = max(1e-4, 1.0 - float(ci_level))
    lo = float(np.quantile(boot_iqms, 0.5 * alpha))
    hi = float(np.quantile(boot_iqms, 1.0 - 0.5 * alpha))
    return float(point), lo, hi


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute sample mean with deterministic percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = float(np.mean(arr))
    if n == 1:
        return point, point, point

    rng = np.random.default_rng(seed)
    b = max(100, int(n_bootstrap))
    idx_mat = rng.integers(0, n, size=(b, n))
    boot_means = np.mean(arr[idx_mat], axis=1)
    alpha = max(1e-4, 1.0 - float(ci_level))
    lo = float(np.quantile(boot_means, 0.5 * alpha))
    hi = float(np.quantile(boot_means, 1.0 - 0.5 * alpha))
    return point, lo, hi


def probability_of_improvement(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Mann-Whitney P(A > B) and bootstrap 95% CI per rliable (Agarwal et al., NeurIPS 2021)."""
    a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return {"prob": 0.5, "ci_low": 0.0, "ci_high": 1.0, "significant": 0.0}

    diff = a[:, None] - b[None, :]
    p_obs = float(np.mean((diff > 1e-12).astype(np.float64) + 0.5 * (np.abs(diff) <= 1e-12).astype(np.float64)))

    if a.size == 1 and b.size == 1:
        return {"prob": p_obs, "ci_low": p_obs, "ci_high": p_obs, "significant": float(abs(p_obs - 0.5) > 1e-6)}

    rng = np.random.default_rng(seed)
    n_boot = max(100, int(n_bootstrap))
    idx_a = rng.integers(0, a.size, size=(n_boot, a.size))
    idx_b = rng.integers(0, b.size, size=(n_boot, b.size))

    if a.size * b.size <= 512:
        diff_boot = a[idx_a][:, :, None] - b[idx_b][:, None, :]
        probs = np.mean(
            (diff_boot > 1e-12).astype(np.float64) + 0.5 * (np.abs(diff_boot) <= 1e-12).astype(np.float64),
            axis=(1, 2),
        )
    else:
        probs = np.empty(n_boot, dtype=np.float64)
        for bi in range(n_boot):
            d_i = a[idx_a[bi]][:, None] - b[idx_b[bi]][None, :]
            probs[bi] = float(np.mean((d_i > 1e-12) + 0.5 * (np.abs(d_i) <= 1e-12)))

    alpha = max(1e-4, 1.0 - float(ci_level))
    lo = float(np.quantile(probs, 0.5 * alpha))
    hi = float(np.quantile(probs, 1.0 - 0.5 * alpha))
    sig = float(lo > 0.5 or hi < 0.5)
    return {"prob": p_obs, "ci_low": lo, "ci_high": hi, "significant": sig}


def compute_performance_profile(
    scores_by_algo: Mapping[str, Sequence[float]],
    *,
    thresholds: Sequence[float] | None = None,
    num_points: int = 25,
    normalize: bool = False,
) -> dict[str, Any]:
    """Compute rliable-style score distribution profiles tau -> P(score >= tau)."""
    arrays: dict[str, np.ndarray] = {
        str(k): np.asarray(v, dtype=np.float64).reshape(-1)
        for k, v in scores_by_algo.items()
        if len(v) > 0
    }
    if not arrays:
        return {"thresholds": [], "normalized": normalize, "curves": {}}

    all_vals = np.concatenate(list(arrays.values()))
    v_min = float(np.min(all_vals))
    v_max = float(np.max(all_vals))
    span = v_max - v_min

    if normalize:
        if span > 1e-12:
            arrays = {k: (v - v_min) / span for k, v in arrays.items()}
        else:
            arrays = {k: np.ones_like(v) for k, v in arrays.items()}
        tau_grid = np.linspace(0.0, 1.0, max(5, int(num_points)))
    elif thresholds is not None:
        tau_grid = np.asarray(list(thresholds), dtype=np.float64)
    else:
        lo = v_min
        hi = v_max if span > 1e-12 else v_min + 1.0
        tau_grid = np.linspace(lo, hi, max(5, int(num_points)))

    curves: dict[str, list[float]] = {}
    for k, arr in arrays.items():
        curves[k] = [round(float(np.mean(arr >= float(tau) - 1e-12)), 4) for tau in tau_grid]

    return {
        "thresholds": [round(float(t), 4) for t in tau_grid],
        "normalized": bool(normalize),
        "curves": curves,
    }


def aggregate_seed_stats(values: Sequence[float]) -> dict[str, float]:
    """Return mean, std, median, IQM, min, max, and 95% Bootstrap CIs over multi-seed scores."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "iqm": 0.0,
            "min": 0.0,
            "max": 0.0,
            "iqm_ci_low": 0.0,
            "iqm_ci_high": 0.0,
            "mean_ci_low": 0.0,
            "mean_ci_high": 0.0,
        }
    iqm, iqm_lo, iqm_hi = bootstrap_iqm_ci(arr.tolist())
    mean, mean_lo, mean_hi = bootstrap_mean_ci(arr.tolist())
    return {
        "mean": mean,
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "iqm": iqm,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "iqm_ci_low": iqm_lo,
        "iqm_ci_high": iqm_hi,
        "mean_ci_low": mean_lo,
        "mean_ci_high": mean_hi,
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

