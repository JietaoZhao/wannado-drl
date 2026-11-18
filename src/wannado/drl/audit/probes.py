"""Active Stage 1 MDP diagnostics: Future-Leakage Probe, numerical conditioning, and mask health.

Pure NumPy implementation (zero external SciPy / scikit-learn runtime dependencies) so
env preflight can verify Markov causality, feature scale conditioning, reward volatility,
and action-mask/saturation invariants before spending compute on DRL training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from wannado.drl.core.wrappers import (
    MultiAgentEnvAdapter,
    OfflineTransitionDataset,
    SequenceTaskEnv,
    VectorizedEnvAdapter,
    ensure_gym_env,
)


@dataclass
class MDPProbeReport:
    """Structured active probe report for MDP causality and numerical conditioning."""

    ok: bool
    future_leakage_suspected: bool = False
    leakage_signals: list[str] = field(default_factory=list)
    leakage_delta_r2: float = 0.0
    leakage_perm_p_value: float = 1.0
    shifted_column_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    obs_condition_ratio: float = 1.0
    obs_feature_stds: list[float] = field(default_factory=list)
    constant_feature_indices: list[int] = field(default_factory=list)
    reward_volatility_ratio: float = 0.0
    mask_sparsity: float | None = None
    mask_deadlock_steps: int = 0
    probed_transitions: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize probe diagnostics to a JSON-compatible dictionary."""
        data = asdict(self)
        data["shifted_column_pairs"] = [[int(j), int(k), float(score)] for j, k, score in self.shifted_column_pairs]
        return data


def compute_action_saturation_rate(
    actions: np.ndarray,
    low: np.ndarray | None,
    high: np.ndarray | None,
    *,
    tol_frac: float = 0.02,
) -> float:
    """Measure fraction of action coordinates pinned near box bounds [low, high]."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.size == 0 or low is None or high is None:
        return 0.0
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    lo = np.asarray(low, dtype=np.float64).reshape(1, -1)
    hi = np.asarray(high, dtype=np.float64).reshape(1, -1)
    finite_mask = np.isfinite(lo) & np.isfinite(hi) & ((hi - lo) > 1e-6)
    if not np.any(finite_mask):
        return 0.0
    span = np.where(finite_mask, hi - lo, 1.0)
    tol = tol_frac * span
    near_low = (arr <= lo + tol) & finite_mask
    near_high = (arr >= hi - tol) & finite_mask
    saturated = np.where(finite_mask, near_low | near_high, False)
    n_finite = int(np.sum(finite_mask)) * arr.shape[0]
    return float(np.sum(saturated) / max(1, n_finite))


def _fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    ridge_alpha: float = 1e-1,
) -> np.ndarray:
    """Fit stabilized ridge regression in closed form on standardized features."""
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    xtr = (x_train - mean) / std
    xva = (x_val - mean) / std

    y_mean = float(np.mean(y_train))
    y_centered = y_train - y_mean
    d = xtr.shape[1]
    if d == 0:
        return np.full(xva.shape[0], y_mean, dtype=np.float64)

    gram = np.einsum("ni,nj->ij", xtr, xtr) + ridge_alpha * np.eye(d, dtype=np.float64)
    rhs = np.einsum("ni,n->i", xtr, y_centered)
    coef = np.linalg.solve(gram, rhs)
    return y_mean + np.einsum("ni,i->n", xva, coef)


def _kfold_cv_r2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    ridge_alpha: float = 1e-1,
) -> tuple[float, np.ndarray]:
    """Compute blocked K-fold out-of-fold R^2 and residuals without lookahead leakage across folds."""
    n = y.shape[0]
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    y_var = float(np.var(y_arr))
    if n < 20 or y_var < 1e-12:
        return 0.0, np.zeros_like(y_arr)

    k = min(n_folds, max(2, n // 12))
    fold_bounds = np.linspace(0, n, k + 1, dtype=int)
    preds = np.empty_like(y_arr)

    for f in range(k):
        lo, hi = int(fold_bounds[f]), int(fold_bounds[f + 1])
        val_mask = np.zeros(n, dtype=np.bool_)
        val_mask[lo:hi] = True
        tr_mask = ~val_mask
        if int(np.sum(tr_mask)) < 8 or int(np.sum(val_mask)) < 2:
            preds[val_mask] = float(np.mean(y_arr))
            continue
        preds[val_mask] = _fit_ridge_predict(x[tr_mask], y_arr[tr_mask], x[val_mask], ridge_alpha=ridge_alpha)

    residuals = y_arr - preds
    mse = float(np.mean(residuals**2))
    r2 = 1.0 - (mse / max(y_var, 1e-12))
    return float(r2), residuals


def _build_polynomial_design(x: np.ndarray, *, max_linear_cols: int = 48) -> np.ndarray:
    """Build [x, x^2] features with safe dimension cap for nonlinear Markov rewards."""
    if x.shape[1] > max_linear_cols:
        # Rank columns by variance to keep ridge regression fast & well-conditioned
        stds = np.std(x, axis=0)
        top_idx = np.argsort(stds)[-max_linear_cols:]
        x = x[:, np.sort(top_idx)]
    return np.hstack([x, x**2])


def _select_active_columns(x: np.ndarray, *, max_cols: int = 48) -> np.ndarray:
    """Filter near-constant columns and cap high-dimensional inputs cleanly."""
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float64)
    stds = np.std(x, axis=0)
    keep = np.flatnonzero(stds > 1e-8)
    if keep.size == 0:
        return np.zeros((x.shape[0], 0), dtype=np.float64)
    if keep.size > max_cols:
        order = np.argsort(stds[keep])[-max_cols:]
        keep = np.sort(keep[order])
    return np.asarray(x[:, keep], dtype=np.float64)


def probe_transition_batch(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    *,
    reset_obs_samples: np.ndarray | None = None,
    masks: np.ndarray | None = None,
    n_permutations: int = 40,
    seed: int = 42,
) -> MDPProbeReport:
    """Inspect a sampled transition buffer for future-leakage, conditioning, and mask anomalies."""
    obs_arr = np.asarray(obs, dtype=np.float64)
    next_obs_arr = np.asarray(next_obs, dtype=np.float64)
    act_arr = np.asarray(actions, dtype=np.float64)
    rew_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
    done_arr = np.asarray(dones, dtype=np.bool_).reshape(-1)

    if obs_arr.ndim == 1:
        obs_arr = obs_arr.reshape(-1, 1)
    elif obs_arr.ndim > 2:
        obs_arr = obs_arr.reshape(obs_arr.shape[0], -1)
    if next_obs_arr.ndim == 1:
        next_obs_arr = next_obs_arr.reshape(-1, 1)
    elif next_obs_arr.ndim > 2:
        next_obs_arr = next_obs_arr.reshape(next_obs_arr.shape[0], -1)
    if act_arr.ndim == 1:
        act_arr = act_arr.reshape(-1, 1)
    elif act_arr.ndim > 2:
        act_arr = act_arr.reshape(act_arr.shape[0], -1)

    n_steps, obs_dim = obs_arr.shape
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []
    leakage_signals: list[str] = []
    shifted_pairs: list[tuple[int, int, float]] = []

    # 1. Numerical feature conditioning & constant-feature detection
    stds = np.std(obs_arr, axis=0) if n_steps > 1 else np.zeros(obs_dim, dtype=np.float64)
    const_idx = [int(i) for i in np.flatnonzero(stds < 1e-8)]
    nonconst_stds = stds[stds >= 1e-8]
    if nonconst_stds.size >= 2:
        min_std = float(np.min(nonconst_stds))
        max_std = float(np.max(nonconst_stds))
        cond_ratio = float(max_std / max(min_std, 1e-12))
    else:
        cond_ratio = 1.0

    if cond_ratio > 100.0:
        warnings.append(
            f"Poor observation feature conditioning (std ratio kappa={cond_ratio:.1f} > 100); "
            "high-scale features will dominate early neural gradients."
        )
        recommendations.append("Enable `normalize_obs=True` (RunningMeanStd) or scale features to O(1).")

    # 2. Reward scale & volatility diagnostic
    rew_mean_abs = float(np.mean(np.abs(rew_arr))) if n_steps > 0 else 0.0
    rew_std = float(np.std(rew_arr)) if n_steps > 1 else 0.0
    rew_vol = float(rew_std / max(rew_mean_abs, 1e-6)) if rew_std > 1e-10 else 0.0
    if rew_std > 1e-8 and rew_vol > 10.0 and rew_std > 20.0:
        warnings.append(
            f"High reward volatility (std={rew_std:.2f}, std/mean|r|={rew_vol:.1f}); "
            "heavy-tailed spikes can destabilize critic value learning."
        )
        recommendations.append("Consider `normalize_reward=True` or clip/rescale extreme reward spikes.")

    # 3. Action-mask health & sparsity
    mask_sparsity: float | None = None
    mask_deadlocks = 0
    if masks is not None and len(masks) > 0:
        mask_arr = np.asarray(masks, dtype=np.bool_)
        valid_counts = np.sum(mask_arr, axis=-1)
        mask_deadlocks = int(np.sum(valid_counts <= 0))
        mask_sparsity = float(1.0 - np.mean(mask_arr))
        if mask_deadlocks > 0:
            errors.append(f"action_mask reached 0 valid actions on {mask_deadlocks} probed step(s) (deadlock).")
        elif mask_sparsity > 0.85:
            warnings.append(f"Action mask blocks {mask_sparsity:.1%} of actions on average; verify valid exploration.")

    # 4. Future-Leakage Probe A: Non-Markov Reward Granger Test
    # For consecutive in-episode steps t and t+1:
    # Under Markov causality, r_{t+1} = R(s_{t+1}, a_{t+1}, eps_{t+1}), so adding s_t
    # MUST NOT systematically boost out-of-fold CV R^2 over nonlinear [s_{t+1}, a_{t+1}].
    delta_r2 = 0.0
    perm_p = 1.0
    if n_steps >= 48 and rew_std > 1e-8:
        valid_pair_mask = ~done_arr[:-1]
        idx_t = np.flatnonzero(valid_pair_mask)
        if idx_t.size >= 40 and float(np.var(rew_arr[idx_t + 1])) > 1e-10:
            s_t = _select_active_columns(obs_arr[idx_t], max_cols=32)
            s_tp1 = _select_active_columns(next_obs_arr[idx_t], max_cols=32)
            a_tp1 = _select_active_columns(act_arr[idx_t + 1], max_cols=16)
            y_tp1 = rew_arr[idx_t + 1]

            base_x = _build_polynomial_design(np.hstack([s_tp1, a_tp1]), max_linear_cols=48)
            full_x = np.hstack([base_x, s_t])

            r2_base, resid_base = _kfold_cv_r2(base_x, y_tp1, n_folds=5, ridge_alpha=1e-1)
            r2_full, resid_full = _kfold_cv_r2(full_x, y_tp1, n_folds=5, ridge_alpha=1e-1)
            raw_delta = float(r2_full - r2_base)
            mse_base = float(np.mean(resid_base**2))
            mse_full = float(np.mean(resid_full**2))

            if raw_delta > 0.05 and mse_full < 0.90 * max(mse_base, 1e-12):
                # Residual permutation test: verify s_t explains residual variance beyond chance
                rng = np.random.default_rng(seed)
                exceed_count = 0
                r2_resid_obs, _ = _kfold_cv_r2(s_t, resid_base, n_folds=5, ridge_alpha=1e-1)
                n_perm = max(10, int(n_permutations))
                for _ in range(n_perm):
                    perm_idx = rng.permutation(s_t.shape[0])
                    r2_perm, _ = _kfold_cv_r2(s_t[perm_idx], resid_base, n_folds=5, ridge_alpha=1e-1)
                    if r2_perm >= r2_resid_obs:
                        exceed_count += 1
                perm_p = float((exceed_count + 1) / (n_perm + 1))
                delta_r2 = max(0.0, raw_delta)
                if delta_r2 > 0.12 and r2_resid_obs > 0.10 and perm_p <= 0.05:
                    leakage_signals.append(
                        f"non_markov_future_reward_leak(delta_R2={delta_r2:.3f}, perm_p={perm_p:.3f})"
                    )

    # 5. Future-Leakage Probe B: Shifted-Column Lookahead Clone Detector
    # Detect s_t[:, j] == s_{t+1}[:, k] (j != k) while ruling out causal frame-stacking / lag buffers.
    if n_steps >= 30 and obs_dim >= 2 and obs_dim <= 64:
        in_ep = ~done_arr
        s_curr = obs_arr[in_ep]
        s_nxt = next_obs_arr[in_ep]
        a_curr = _select_active_columns(act_arr[in_ep], max_cols=16)
        r_curr = rew_arr[in_ep]
        active_cols = [c for c in range(obs_dim) if c not in const_idx]

        # Build reset disparity baseline if reset samples were provided
        reset_arr = np.asarray(reset_obs_samples, dtype=np.float64) if reset_obs_samples is not None else None
        if reset_arr is not None and reset_arr.ndim == 1:
            reset_arr = reset_arr.reshape(1, -1)

        if s_curr.shape[0] >= 25 and len(active_cols) >= 2:
            for j in active_cols:
                xj = s_curr[:, j]
                xj_norm = (xj - np.mean(xj)) / max(float(np.std(xj)), 1e-12)
                for k in active_cols:
                    if j == k:
                        continue
                    yk_next = s_nxt[:, k]
                    yk_curr = s_curr[:, k]
                    # Column k must actually evolve across step t -> t+1
                    step_diff_k = float(np.std(yk_next - yk_curr))
                    if step_diff_k < 1e-4 * max(float(np.std(yk_curr)), 1e-6):
                        continue

                    yk_next_norm = (yk_next - np.mean(yk_next)) / max(float(np.std(yk_next)), 1e-12)
                    nrmse = float(np.sqrt(np.mean((xj_norm - yk_next_norm) ** 2)))
                    corr_jk = float(np.mean(xj_norm * yk_next_norm))
                    yk_curr_norm = (yk_curr - np.mean(yk_curr)) / max(float(np.std(yk_curr)), 1e-12)
                    same_step_nrmse = float(np.sqrt(np.mean((xj_norm - yk_curr_norm) ** 2)))
                    if corr_jk <= 0.998 or nrmse >= 0.05 or same_step_nrmse <= 4.0 * max(nrmse, 1e-4):
                        continue

                    # Disambiguate causal FrameStack / history lag vs lookahead:
                    # In causal FrameStack, reset() pads or copies the lagged slot k so obs_0[k] is constant/0 or equals obs_0[j].
                    is_causal_lag = False
                    if reset_arr is not None and reset_arr.shape[0] >= 2 and reset_arr.shape[1] == obs_dim:
                        lag_zero_padded = float(np.max(np.abs(reset_arr[:, k]))) < 1e-6
                        diff_reset = float(np.max(np.abs(reset_arr[:, k] - reset_arr[:, j])))
                        if lag_zero_padded or diff_reset < 1e-5:
                            is_causal_lag = True

                    # Also check if s_{t+1}[:, j] is exogenous unpredictable while (s_t, a_t) predicts r_t
                    # vs causal frame-stacking where slot k starts equal/padded.
                    if not is_causal_lag:
                        shifted_pairs.append((int(j), int(k), round(float(corr_jk), 5)))
                        leakage_signals.append(
                            f"shifted_lookahead_clone(obs_t[{j}] == obs_{{t+1}}[{k}], corr={corr_jk:.4f})"
                        )

            # 6. Probe C: Exogenous Future-Innovation Column inside obs_t
            # Catches s_t = [state_t, future_return_{t+1}] where future_return_{t+1} is NOT copied to obs_{t+1},
            # yet deterministically dictates (obs_{t+1}[k] - obs_t[k]) and drives action-interacted reward r_t.
            if not shifted_pairs and rew_std > 1e-8 and s_curr.shape[0] >= 35:
                sa_curr = np.hstack([s_curr, a_curr])
                for j in active_cols:
                    # Exogenous innovations are unpredictable in t+1 from (s_t, a_t)
                    r2_next_j, _ = _kfold_cv_r2(sa_curr, s_nxt[:, j], n_folds=4, ridge_alpha=1e-1)
                    if r2_next_j > 0.30:
                        continue
                    xj = s_curr[:, j]
                    for k in active_cols:
                        if j == k:
                            continue
                        delta_k = s_nxt[:, k] - s_curr[:, k]
                        if float(np.std(delta_k)) < 1e-5:
                            continue
                        design_jk = np.column_stack([xj, s_curr[:, k], xj * s_curr[:, k]])
                        r2_delta_k, _ = _kfold_cv_r2(design_jk, delta_k, n_folds=4, ridge_alpha=1e-2)
                        if r2_delta_k > 0.92:
                            # Confirm feature j (or j * a_t) drives current reward r_t beyond chance
                            inter_full = np.hstack(
                                [s_curr, a_curr, s_curr[:, j : j + 1] * a_curr if a_curr.shape[1] else s_curr[:, j : j + 1]]
                            )
                            others = [c for c in range(obs_dim) if c != j]
                            inter_wo = (
                                np.hstack([s_curr[:, others], a_curr]) if others else a_curr
                            )
                            r2_r_full, _ = _kfold_cv_r2(inter_full, r_curr, n_folds=4, ridge_alpha=1e-1)
                            r2_r_wo, _ = _kfold_cv_r2(inter_wo, r_curr, n_folds=4, ridge_alpha=1e-1) if inter_wo.shape[1] else (0.0, np.zeros_like(r_curr))
                            if r2_r_full - r2_r_wo > 0.25:
                                leakage_signals.append(
                                    f"exogenous_future_innovation_in_obs(col={j} -> delta_col[{k}] R^2={r2_delta_k:.3f})"
                                )
                                break

    suspected = len(leakage_signals) > 0
    if suspected:
        errors.append(
            "Suspected future information leakage detected by active MDP probe: " + "; ".join(leakage_signals)
        )
        recommendations.append(
            "Remove future t+1 targets/prices/shocks from obs_t so the agent acts strictly on causal history <= t."
        )

    ok = len(errors) == 0
    return MDPProbeReport(
        ok=ok,
        future_leakage_suspected=suspected,
        leakage_signals=leakage_signals,
        leakage_delta_r2=round(float(delta_r2), 4),
        leakage_perm_p_value=round(float(perm_p), 4),
        shifted_column_pairs=shifted_pairs,
        obs_condition_ratio=round(float(cond_ratio), 3),
        obs_feature_stds=[round(float(x), 6) for x in stds.tolist()],
        constant_feature_indices=const_idx,
        reward_volatility_ratio=round(float(rew_vol), 4),
        mask_sparsity=round(float(mask_sparsity), 4) if mask_sparsity is not None else None,
        mask_deadlock_steps=mask_deadlocks,
        probed_transitions=int(n_steps),
        warnings=warnings,
        errors=errors,
        recommendations=recommendations,
    )


def run_mdp_probes(
    env_or_factory: Any,
    *,
    horizon_steps: int = 160,
    n_permutations: int = 40,
    seed: int = 42,
    heuristic_fn: Callable[[np.ndarray, dict[str, Any]], Any] | None = None,
) -> MDPProbeReport:
    """Actively probe an environment or factory for future leakage, numerical conditioning, and masks."""
    target = env_or_factory
    if isinstance(target, VectorizedEnvAdapter):
        target = target.raw_env
    elif (
        callable(env_or_factory)
        and (isinstance(env_or_factory, type) or not hasattr(env_or_factory, "step"))
        and not hasattr(env_or_factory, "possible_agents")
        and not isinstance(env_or_factory, (SequenceTaskEnv, MultiAgentEnvAdapter, OfflineTransitionDataset))
    ):
        target = env_or_factory()

    if isinstance(target, OfflineTransitionDataset):
        return probe_transition_batch(
            target.obs,
            target.actions,
            target.rewards,
            target.next_obs,
            target.terminateds,
            masks=target.extra.get("action_masks"),
            n_permutations=n_permutations,
            seed=seed,
        )

    if isinstance(target, (SequenceTaskEnv, MultiAgentEnvAdapter)) or hasattr(target, "possible_agents"):
        return MDPProbeReport(ok=True, probed_transitions=0)

    env = ensure_gym_env(target, heuristic_fn=heuristic_fn)
    rng = np.random.default_rng(seed)

    # Collect multi-seed initial observations to distinguish causal lag stacks from shifted lookahead
    reset_samples: list[np.ndarray] = []
    for r_i in range(4):
        obs_init, _ = env.reset(seed=seed + 1000 + r_i)
        reset_samples.append(np.asarray(obs_init, dtype=np.float64).reshape(-1))
    reset_arr = np.vstack(reset_samples)

    obs, info = env.reset(seed=seed)
    obs_list: list[np.ndarray] = []
    next_obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    rew_list: list[float] = []
    done_list: list[bool] = []
    mask_list: list[np.ndarray] = []

    for _ in range(max(16, int(horizon_steps))):
        if env.is_discrete:
            mask = info.get("action_mask")
            if mask is not None:
                m_arr = np.asarray(mask, dtype=np.bool_).reshape(-1)
                mask_list.append(m_arr)
                valid = np.flatnonzero(m_arr)
                act: Any = int(rng.choice(valid)) if valid.size > 0 else 0
            else:
                act = int(rng.integers(0, env.action_dim))
            act_vec = np.array([float(act)], dtype=np.float64)
        elif env.action_nvec is not None:
            act = np.asarray([int(rng.integers(0, k)) for k in env.action_nvec], dtype=np.int64)
            act_vec = np.asarray(act, dtype=np.float64).reshape(-1)
        else:
            lo = env.action_low if env.action_low is not None else np.full(env.action_dim, -1.0, dtype=np.float32)
            hi = env.action_high if env.action_high is not None else np.full(env.action_dim, 1.0, dtype=np.float32)
            lo_s = np.where(np.isfinite(lo), lo, -1.0)
            hi_s = np.where(np.isfinite(hi), hi, 1.0)
            act = rng.uniform(lo_s, hi_s).astype(np.float32)
            act_vec = np.asarray(act, dtype=np.float64).reshape(-1)

        nxt, rew, term, trunc, nxt_info = env.step(act)
        obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        nxt_arr = np.asarray(nxt, dtype=np.float64).reshape(-1)
        if not (np.all(np.isfinite(obs_arr)) and np.all(np.isfinite(nxt_arr)) and np.isfinite(rew)):
            break

        obs_list.append(obs_arr)
        next_obs_list.append(nxt_arr)
        act_list.append(act_vec)
        rew_list.append(float(rew))
        done = bool(term or trunc)
        done_list.append(done)

        if done:
            obs, info = env.reset(seed=int(rng.integers(0, 100_000)))
        else:
            obs, info = nxt, nxt_info

    if not obs_list:
        return MDPProbeReport(ok=True, probed_transitions=0)

    return probe_transition_batch(
        np.vstack(obs_list),
        np.vstack(act_list),
        np.asarray(rew_list, dtype=np.float64),
        np.vstack(next_obs_list),
        np.asarray(done_list, dtype=np.bool_),
        reset_obs_samples=reset_arr,
        masks=np.vstack(mask_list) if mask_list else None,
        n_permutations=n_permutations,
        seed=seed,
    )
