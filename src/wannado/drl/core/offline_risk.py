"""Offline Policy Evaluation (WIS / FQE), CVaR tail-risk metrics, and simplex portfolio wrappers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from wannado.drl.core.networks import QNetwork, TwinQCritic
from wannado.drl.core.obs_spec import obs_to_tensor, slice_obs
from wannado.drl.core.utils import hard_update, polyak_update, resolve_device, set_global_seed
from wannado.drl.core.wrappers import DictRunningMeanStd, OfflineTransitionDataset


def compute_cvar(returns: Sequence[float] | np.ndarray, *, alpha: float = 0.05) -> float:
    """Compute lower-tail Conditional Value-at-Risk (Expected Shortfall at level ``alpha``).

    Returns the mean of the worst ``alpha`` fraction of returns (higher is better; matches
    Acerbi & Tasche 2002 coherent risk ordering on reward distributions).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    sorted_vals = np.sort(arr)
    k = max(1, int(np.ceil(float(alpha) * sorted_vals.size)))
    return float(np.mean(sorted_vals[:k]))


def compute_sharpe_and_drawdown(
    returns: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute return, volatility, Sharpe-like ratio, maximum drawdown, VaR, and CVaR."""
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "var_alpha": 0.0,
            "cvar_alpha": 0.0,
        }
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=0))
    sharpe = float(mean_r / max(std_r, 1e-8))
    cum = np.cumsum(arr)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    drawdowns = np.maximum(0.0, peaks - cum)
    max_dd = float(np.max(drawdowns)) if drawdowns.size else 0.0
    q_val = float(np.quantile(arr, alpha))
    return {
        "mean_return": mean_r,
        "std_return": std_r,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "var_alpha": q_val,
        "cvar_alpha": compute_cvar(arr, alpha=alpha),
    }


def apply_simplex_projection(
    weights: np.ndarray | Sequence[float],
    *,
    min_weight: float = 0.0,
    use_softmax: bool = False,
    temperature: float = 1.0,
) -> np.ndarray:
    """Project raw actor outputs onto the probability simplex (``w >= min_weight``, ``sum(w) == 1``).

    Uses Duchi et al. (2008) exact Euclidean simplex projection by default, or temperature-scaled
    softmax normalization when ``use_softmax=True``.
    """
    v = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
    floor = float(min_weight)
    if floor < 0.0 or floor * n > 1.0 + 1e-8:
        raise ValueError(f"Infeasible simplex min_weight={min_weight} for {n} assets.")
    residual = max(0.0, 1.0 - floor * n)
    if use_softmax:
        scaled = (v - np.max(v)) / max(float(temperature), 1e-6)
        exp_v = np.exp(np.clip(scaled, -40.0, 40.0))
        probs = exp_v / max(float(np.sum(exp_v)), 1e-12)
        return (floor + residual * probs).astype(np.float32)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - residual
    ind = np.arange(1, n + 1, dtype=np.float64)
    cond = u - cssv / ind > 0.0
    rho = int(np.flatnonzero(cond)[-1]) if np.any(cond) else 0
    theta = cssv[rho] / float(rho + 1)
    proj = np.maximum(v - theta, 0.0) + floor
    s = float(np.sum(proj))
    if s > 0.0:
        proj = proj / s
    return proj.astype(np.float32)


class PortfolioSimplexWrapper(gym.Wrapper):
    """Wrap any continuous allocation env so policy actions always lie on the portfolio simplex.

    Maps raw continuous vectors onto non-negative weights summing to 1.0, appends turnover &
    concentration metrics into ``info['task_metrics']``, and optionally deducts proportional
    transaction costs from step rewards.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        use_softmax: bool = False,
        min_weight: float = 0.0,
        turnover_cost: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.use_softmax = bool(use_softmax)
        self.min_weight = float(min_weight)
        self.turnover_cost = float(turnover_cost)
        self._prev_weights: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) and len(out) == 2 else (out, {})
        self._prev_weights = None
        return obs, dict(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        w = apply_simplex_projection(action, min_weight=self.min_weight, use_softmax=self.use_softmax)
        turnover = float(np.sum(np.abs(w - self._prev_weights))) if self._prev_weights is not None else 0.0
        self._prev_weights = w.copy()

        out = self.env.step(w)
        if len(out) == 5:
            obs, rew, term, trunc, raw_info = out
            info = dict(raw_info) if isinstance(raw_info, dict) else {}
        elif len(out) == 4:
            obs, rew, done, raw_info = out
            info = dict(raw_info) if isinstance(raw_info, dict) else {}
            trunc = bool(info.get("TimeLimit.truncated", False))
            term = bool(done) and not trunc
        else:
            raise ValueError(f"Unsupported wrapped env.step() tuple length: {len(out)}")

        adj_rew = float(rew) - self.turnover_cost * turnover
        t_metrics = dict(info.get("task_metrics") or {})
        t_metrics["portfolio_turnover"] = turnover
        t_metrics["max_asset_weight"] = float(np.max(w)) if w.size else 0.0
        t_metrics["herfindahl_concentration"] = float(np.sum(w * w)) if w.size else 0.0
        info["task_metrics"] = t_metrics
        info["simplex_weights"] = w
        return obs, adj_rew, bool(term), bool(trunc), info


def _split_dataset_episodes(dataset: OfflineTransitionDataset) -> list[np.ndarray]:
    n = len(dataset)
    ep_ids_raw = dataset.extra.get("episode_ids")
    if ep_ids_raw is not None:
        ep_ids = np.asarray(ep_ids_raw).reshape(-1)
        if ep_ids.shape[0] == n:
            slices: list[np.ndarray] = []
            start = 0
            for i in range(1, n):
                if ep_ids[i] != ep_ids[i - 1]:
                    slices.append(np.arange(start, i, dtype=np.int64))
                    start = i
            slices.append(np.arange(start, n, dtype=np.int64))
            return slices

    term_idx = np.flatnonzero(dataset.terminateds > 0.5)
    if term_idx.size > 0:
        slices = []
        start = 0
        for end_i in term_idx:
            slices.append(np.arange(start, int(end_i) + 1, dtype=np.int64))
            start = int(end_i) + 1
        if start < n:
            slices.append(np.arange(start, n, dtype=np.int64))
        return slices

    # No explicit boundaries: segment fixed chunks so WIS variance remains bounded
    chunk = max(1, min(50, n))
    return [np.arange(i, min(n, i + chunk), dtype=np.int64) for i in range(0, n, chunk)]


def evaluate_offline_wis(
    dataset: OfflineTransitionDataset,
    policy_fn: Callable[..., Any],
    *,
    gamma: float = 0.99,
    clip_rho: float = 10.0,
    soft_eps: float = 0.05,
    alpha_cvar: float = 0.05,
) -> dict[str, float]:
    """Estimate Off-Policy Value via Self-Normalized Weighted Importance Sampling (WIS).

    Works without a simulator on both discrete and continuous ``OfflineTransitionDataset`` instances,
    using logged ``dataset.extra['behavior_probs']`` when supplied or empirical smoothing otherwise.
    """
    n = len(dataset)
    if n == 0:
        return {
            "behavior_return": 0.0,
            "wis_return": 0.0,
            "effective_sample_size": 0.0,
            "cvar_alpha": 0.0,
            "num_episodes": 0.0,
        }

    episodes = _split_dataset_episodes(dataset)
    b_probs_raw = dataset.extra.get("behavior_probs")
    masks_raw = dataset.extra.get("action_masks")

    if dataset.is_discrete:
        if b_probs_raw is not None:
            b_probs = np.clip(np.asarray(b_probs_raw, dtype=np.float64).reshape(-1), 1e-4, 1.0)
        else:
            counts = np.bincount(dataset.actions.astype(np.int64).reshape(-1), minlength=dataset.action_dim).astype(np.float64)
            marginal = (counts + 1.0) / float(np.sum(counts) + dataset.action_dim)
            b_probs = np.clip(marginal[dataset.actions.astype(np.int64).reshape(-1)], 1e-4, 1.0)

        eval_probs = np.zeros(n, dtype=np.float64)
        for i in range(n):
            info_i = {"action_mask": masks_raw[i]} if masks_raw is not None else {}
            pred = int(np.asarray(policy_fn(slice_obs(dataset.obs, i), info_i)).item())
            if int(dataset.actions[i]) == pred:
                eval_probs[i] = 1.0 - soft_eps
            else:
                eval_probs[i] = soft_eps / max(1, dataset.action_dim - 1)
        step_rhos = np.clip(eval_probs / b_probs, 0.0, float(clip_rho))
    else:
        act_mat = dataset.actions.astype(np.float64)
        pred_acts = np.asarray([np.asarray(policy_fn(slice_obs(dataset.obs, i), {}), dtype=np.float64).reshape(-1) for i in range(n)])
        scale = np.maximum(np.std(act_mat, axis=0), 0.1)
        dist_sq = np.mean(((pred_acts - act_mat) / scale) ** 2, axis=-1)
        if b_probs_raw is not None:
            raw_rho = np.exp(-0.5 * dist_sq) / np.clip(np.asarray(b_probs_raw, dtype=np.float64).reshape(-1), 1e-3, 1.0)
        else:
            raw_rho = np.exp(-0.5 * dist_sq)
        step_rhos = np.clip(raw_rho, 1e-3, float(clip_rho))

    beh_returns: list[float] = []
    ep_weights: list[float] = []
    ep_weighted_returns: list[float] = []
    for idx in episodes:
        rews = dataset.rewards[idx].astype(np.float64)
        discounts = float(gamma) ** np.arange(rews.size, dtype=np.float64)
        g_ret = float(np.sum(discounts * rews))
        beh_returns.append(g_ret)

        # Geometric mean per-step importance ratio prevents long-horizon weight collapse in WIS
        rhos_ep = np.clip(step_rhos[idx], 1e-4, float(clip_rho))
        w_ep = float(np.exp(np.clip(np.mean(np.log(rhos_ep)), -4.0, 4.0)))
        ep_weights.append(w_ep)
        ep_weighted_returns.append(w_ep * g_ret)

    w_arr = np.asarray(ep_weights, dtype=np.float64)
    w_sum = max(float(np.sum(w_arr)), 1e-8)
    wis_ret = float(np.sum(ep_weighted_returns) / w_sum)
    ess = float((w_sum * w_sum) / max(float(np.sum(w_arr * w_arr)), 1e-8))
    return {
        "behavior_return": float(np.mean(beh_returns)) if beh_returns else 0.0,
        "wis_return": wis_ret,
        "effective_sample_size": ess,
        "cvar_alpha": compute_cvar(beh_returns, alpha=alpha_cvar),
        "num_episodes": float(len(episodes)),
    }


def evaluate_offline_fqe(
    dataset: OfflineTransitionDataset,
    policy_fn: Callable[..., Any],
    *,
    gamma: float = 0.99,
    num_steps: int = 120,
    hidden_dims: tuple[int, ...] = (64, 64),
    lr: float = 5e-4,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "auto",
    alpha_cvar: float = 0.05,
) -> dict[str, float]:
    """Fitted Q-Evaluation (FQE; Le et al. 2019) estimating target-policy value strictly offline."""
    n = len(dataset)
    if n < 2:
        return {
            "fqe_initial_value": 0.0,
            "fqe_mean_q": 0.0,
            "fqe_cvar_alpha": 0.0,
            "fqe_bellman_mse": 0.0,
            "behavior_return": 0.0,
            "num_episodes": 0.0,
        }

    rng = set_global_seed(seed)
    dev = resolve_device(device)
    obs_spec = dataset.resolved_obs_spec
    obs_rms = DictRunningMeanStd(obs_spec)
    obs_rms.update(dataset.obs)
    obs_rms.freeze(True)
    obs_norm = obs_rms.normalize(dataset.obs)
    next_obs_norm = obs_rms.normalize(dataset.next_obs)

    episodes = _split_dataset_episodes(dataset)
    init_indices = np.asarray([int(ep[0]) for ep in episodes], dtype=np.int64)
    beh_returns = [
        float(np.sum((float(gamma) ** np.arange(ep.size, dtype=np.float64)) * dataset.rewards[ep].astype(np.float64)))
        for ep in episodes
    ]

    next_masks_raw = dataset.extra.get("next_action_masks")
    next_actions_raw = np.zeros_like(dataset.actions)
    for i in range(n):
        info_next = {"action_mask": next_masks_raw[i]} if next_masks_raw is not None else {}
        act_pred = policy_fn(slice_obs(dataset.next_obs, i), info_next)
        next_actions_raw[i] = int(np.asarray(act_pred).item()) if dataset.is_discrete else np.asarray(act_pred, dtype=np.float32)

    bs = min(batch_size, n)
    mses: list[float] = []

    if dataset.is_discrete:
        q_net = QNetwork(obs_spec, dataset.action_dim, hidden_dims=hidden_dims).to(dev)
        target_net = QNetwork(obs_spec, dataset.action_dim, hidden_dims=hidden_dims).to(dev)
        hard_update(q_net, target_net)
        opt = torch.optim.Adam(q_net.parameters(), lr=lr)

        b_na_all = next_actions_raw.astype(np.int64).reshape(-1)
        for _ in range(num_steps):
            idx = rng.integers(0, n, size=bs)
            bo = obs_to_tensor(slice_obs(obs_norm, idx), obs_spec, dev, add_batch_dim=False)
            ba = torch.as_tensor(dataset.actions[idx].astype(np.int64).reshape(-1), device=dev, dtype=torch.long)
            br = torch.as_tensor(dataset.rewards[idx], device=dev, dtype=torch.float32)
            bno = obs_to_tensor(slice_obs(next_obs_norm, idx), obs_spec, dev, add_batch_dim=False)
            bna = torch.as_tensor(b_na_all[idx], device=dev, dtype=torch.long)
            bt = torch.as_tensor(dataset.terminateds[idx], device=dev, dtype=torch.float32)

            with torch.no_grad():
                q_next = target_net(bno).gather(1, bna.unsqueeze(1)).squeeze(1)
                td_target = br + float(gamma) * (1.0 - bt) * q_next
            q_pred = q_net(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_pred, td_target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
            opt.step()
            polyak_update(q_net, target_net, 0.02)
            mses.append(float(loss.item()))

        masks_raw = dataset.extra.get("action_masks")
        init_acts = np.asarray(
            [
                int(np.asarray(policy_fn(slice_obs(dataset.obs, int(i)), {"action_mask": masks_raw[i]} if masks_raw is not None else {})).item())
                for i in init_indices
            ],
            dtype=np.int64,
        )
        all_acts = np.asarray(
            [
                int(np.asarray(policy_fn(slice_obs(dataset.obs, i), {"action_mask": masks_raw[i]} if masks_raw is not None else {})).item())
                for i in range(min(n, 512))
            ],
            dtype=np.int64,
        )
        with torch.no_grad():
            q_init = q_net(obs_to_tensor(slice_obs(obs_norm, init_indices), obs_spec, dev, add_batch_dim=False))
            init_vals = q_init.gather(1, torch.as_tensor(init_acts, device=dev).unsqueeze(1)).squeeze(1).cpu().numpy()
            q_all = q_net(obs_to_tensor(slice_obs(obs_norm, slice(0, len(all_acts))), obs_spec, dev, add_batch_dim=False))
            mean_q = float(q_all.gather(1, torch.as_tensor(all_acts, device=dev).unsqueeze(1)).squeeze(1).mean().item())
    else:
        q_net_c = TwinQCritic(obs_spec, dataset.action_dim, hidden_dims=hidden_dims).to(dev)
        target_net_c = TwinQCritic(obs_spec, dataset.action_dim, hidden_dims=hidden_dims).to(dev)
        hard_update(q_net_c, target_net_c)
        opt_c = torch.optim.Adam(q_net_c.parameters(), lr=lr)

        b_na_c_all = next_actions_raw.astype(np.float32)
        for _ in range(num_steps):
            idx = rng.integers(0, n, size=bs)
            bo = obs_to_tensor(slice_obs(obs_norm, idx), obs_spec, dev, add_batch_dim=False)
            ba = torch.as_tensor(dataset.actions[idx].astype(np.float32), device=dev, dtype=torch.float32)
            br = torch.as_tensor(dataset.rewards[idx], device=dev, dtype=torch.float32).unsqueeze(-1)
            bno = obs_to_tensor(slice_obs(next_obs_norm, idx), obs_spec, dev, add_batch_dim=False)
            bna = torch.as_tensor(b_na_c_all[idx], device=dev, dtype=torch.float32)
            bt = torch.as_tensor(dataset.terminateds[idx], device=dev, dtype=torch.float32).unsqueeze(-1)

            with torch.no_grad():
                tq1, tq2 = target_net_c(bno, bna)
                td_target = br + float(gamma) * (1.0 - bt) * torch.min(tq1, tq2)
            q1, q2 = q_net_c(bo, ba)
            loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net_c.parameters(), 5.0)
            opt_c.step()
            polyak_update(q_net_c, target_net_c, 0.02)
            mses.append(float(loss.item()))

        init_acts_c = np.asarray([np.asarray(policy_fn(slice_obs(dataset.obs, int(i)), {}), dtype=np.float32) for i in init_indices])
        all_acts_c = np.asarray([np.asarray(policy_fn(slice_obs(dataset.obs, i), {}), dtype=np.float32) for i in range(min(n, 512))])
        with torch.no_grad():
            iq1, iq2 = q_net_c(
                obs_to_tensor(slice_obs(obs_norm, init_indices), obs_spec, dev, add_batch_dim=False),
                torch.as_tensor(init_acts_c, device=dev, dtype=torch.float32),
            )
            init_vals = torch.min(iq1, iq2).squeeze(-1).cpu().numpy()
            aq1, aq2 = q_net_c(
                obs_to_tensor(slice_obs(obs_norm, slice(0, len(all_acts_c))), obs_spec, dev, add_batch_dim=False),
                torch.as_tensor(all_acts_c, device=dev, dtype=torch.float32),
            )
            mean_q = float(torch.min(aq1, aq2).mean().item())

    return {
        "fqe_initial_value": float(np.mean(init_vals)) if len(init_vals) else 0.0,
        "fqe_mean_q": mean_q,
        "fqe_cvar_alpha": compute_cvar(init_vals, alpha=alpha_cvar),
        "fqe_bellman_mse": float(np.mean(mses[-10:])) if mses else 0.0,
        "behavior_return": float(np.mean(beh_returns)) if beh_returns else 0.0,
        "num_episodes": float(len(episodes)),
    }


def evaluate_offline_ope(
    dataset: OfflineTransitionDataset,
    policy_fn: Callable[..., Any],
    *,
    gamma: float = 0.99,
    alpha_cvar: float = 0.05,
    seed: int = 42,
    fqe_steps: int = 120,
) -> dict[str, Any]:
    """Run unified Off-Policy Evaluation (WIS + FQE + optional demographic subgroup safety breakdown)."""
    wis = evaluate_offline_wis(dataset, policy_fn, gamma=gamma, alpha_cvar=alpha_cvar)
    fqe = evaluate_offline_fqe(dataset, policy_fn, gamma=gamma, num_steps=fqe_steps, seed=seed, alpha_cvar=alpha_cvar)

    subgroups: dict[str, dict[str, float]] = {}
    groups_raw = dataset.extra.get("group_labels")
    if groups_raw is not None:
        g_arr = np.asarray(groups_raw).reshape(-1)
        if g_arr.shape[0] == len(dataset):
            for g in sorted({str(x) for x in g_arr}):
                idx = np.flatnonzero(g_arr.astype(str) == g)
                if idx.size == 0:
                    continue
                sub_ds = OfflineTransitionDataset(
                    obs=dataset.obs[idx],
                    actions=dataset.actions[idx],
                    rewards=dataset.rewards[idx],
                    next_obs=dataset.next_obs[idx],
                    terminateds=dataset.terminateds[idx],
                    is_discrete=dataset.is_discrete,
                    action_dim=dataset.action_dim,
                    action_low=dataset.action_low,
                    action_high=dataset.action_high,
                    extra={
                        k: v[idx]
                        for k, v in dataset.extra.items()
                        if k != "group_labels" and isinstance(v, np.ndarray) and v.shape[0] == len(dataset)
                    },
                )
                sub_wis = evaluate_offline_wis(sub_ds, policy_fn, gamma=gamma, alpha_cvar=alpha_cvar)
                subgroups[g] = {
                    "count": float(idx.size),
                    "mean_reward": float(np.mean(sub_ds.rewards)),
                    "behavior_return": sub_wis["behavior_return"],
                    "wis_return": sub_wis["wis_return"],
                    "cvar_alpha": sub_wis["cvar_alpha"],
                }

    return {
        "wis": wis,
        "fqe": fqe,
        "primary_ope_value": fqe["fqe_initial_value"] if fqe["num_episodes"] > 0 else wis["wis_return"],
        "behavior_return": fqe["behavior_return"],
        "subgroups": subgroups,
    }
