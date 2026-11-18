"""External policy adapters for auditing SB3, PyTorch nn.Module, PolicyBundle, and custom policies.

Allows any trained policy (Stable-Baselines3 ``.predict()`` + optional ``VecNormalize``,
raw PyTorch ``nn.Module``, saved ``wannado.drl`` ``PolicyBundle``, or arbitrary Python
callable) to enter Stage 3 Rigor Auditing and Stage 4 Paper-Ready DRL Reporting without
rewriting the user's training script.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wannado.drl.audit.preflight import PreflightReport, preflight_env
from wannado.drl.audit.probes import compute_action_saturation_rate
from wannado.drl.audit.rigor_auditor import RigorAuditReport, audit_drl_rigor
from wannado.drl.core.checkpoint import PolicyBundle, load_policy
from wannado.drl.core.types import EpisodeEvalStats, TrainResult
from wannado.drl.core.wrappers import UniversalGymAdapter, ensure_gym_env
from wannado.drl.mdp.baselines import evaluate_policy_episodes


def _apply_obs_rms(obs: np.ndarray, obs_rms: Any, clip_obs: float = 10.0) -> np.ndarray:
    """Apply frozen RunningMeanStd stats (SB3 or wannado.drl shape) without mutating running estimates."""
    mean = np.asarray(getattr(obs_rms, "mean", 0.0), dtype=np.float32)
    var = np.asarray(getattr(obs_rms, "var", 1.0), dtype=np.float32)
    eps = float(getattr(obs_rms, "eps", getattr(obs_rms, "epsilon", 1e-8)))
    normed = (np.asarray(obs, dtype=np.float32) - mean) / np.sqrt(var + eps)
    return np.clip(normed, -clip_obs, clip_obs).astype(np.float32)


def _load_external_artifact(policy_or_path: Any, sb3_algo: str | None = None) -> Any:
    """Resolve disk artifacts (.pt/.pth PolicyBundle or optional SB3 .zip checkpoint)."""
    if not isinstance(policy_or_path, (str, Path)):
        return policy_or_path
    path = Path(policy_or_path)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return load_policy(path)
    if suffix == ".zip":
        try:
            import stable_baselines3 as sb3  # type: ignore[import-not-found]
        except Exception as exc:
            raise ImportError(
                f"Loading SB3 checkpoint '{path}' requires optional `stable-baselines3` to be installed."
            ) from exc

        candidates: list[str]
        if sb3_algo:
            candidates = [sb3_algo.upper()]
        else:
            candidates = ["PPO", "SAC", "TD3", "DQN", "A2C"]
        last_err: Exception | None = None
        for cls_name in candidates:
            cls = getattr(sb3, cls_name, None)
            if cls is None:
                continue
            try:
                return cls.load(str(path), device="cpu")
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as err:
                last_err = err
        raise RuntimeError(f"Unable to load SB3 archive '{path}' via {candidates}: {last_err}")
    raise ValueError(f"Unsupported policy checkpoint extension '{suffix}' for '{path}'. Expected .pt, .pth, or .zip.")


class ExternalPolicyWrapper:
    """Unified deterministic/eval callable adapter wrapping SB3, nn.Module, PolicyBundle, or callables."""

    def __init__(
        self,
        policy: Any,
        env_adapter: UniversalGymAdapter,
        *,
        deterministic: bool = True,
        vec_normalize: Any = None,
        obs_rms: Any = None,
        device: str = "cpu",
    ) -> None:
        self.raw_policy = policy
        self.env_adapter = env_adapter
        self.deterministic = bool(deterministic)
        self.device = torch.device(device)
        self.backend = "callable"
        self.obs_rms_frozen_on_eval = True
        self._vec_normalize = vec_normalize
        self._obs_rms = obs_rms
        self._clip_obs = float(getattr(vec_normalize, "clip_obs", 10.0)) if vec_normalize is not None else 10.0

        # Detect unfrozen SB3 / external VecNormalize normalization statistics during eval
        vn = vec_normalize
        if vn is None and hasattr(policy, "get_vec_normalize_env"):
            try:
                vn = policy.get_vec_normalize_env()
            except (RuntimeError, ValueError, TypeError, AttributeError):
                vn = None
        if vn is not None:
            self._vec_normalize = vn
            if getattr(vn, "training", False) is True and getattr(vn, "norm_obs", True) is True:
                self.obs_rms_frozen_on_eval = False
            if self._obs_rms is None:
                self._obs_rms = getattr(vn, "obs_rms", None)
        elif hasattr(env_adapter.raw_env, "training") and getattr(env_adapter.raw_env, "training", False) is True:
            self.obs_rms_frozen_on_eval = False

        # Identify policy backend
        self.obs_spec = getattr(policy, "obs_spec", None)
        self._sb3_accepts_masks = False
        self._callable_takes_info = False

        if isinstance(policy, PolicyBundle):
            self.backend = f"policy_bundle:{policy.algo_id}"
        elif hasattr(policy, "predict") and callable(policy.predict):
            self.backend = f"sb3:{type(policy).__name__}"
            try:
                sig = inspect.signature(policy.predict)
                self._sb3_accepts_masks = "action_masks" in sig.parameters
            except (TypeError, ValueError):
                self._sb3_accepts_masks = False
        elif isinstance(policy, torch.nn.Module):
            self.backend = f"torch_module:{type(policy).__name__}"
            policy.eval()
        elif callable(policy):
            self.backend = "callable"
            try:
                sig = inspect.signature(policy)
                params = list(sig.parameters.values())
                has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
                pos_params = [
                    p
                    for p in params
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                self._callable_takes_info = has_varargs or len(pos_params) >= 2
            except (TypeError, ValueError):
                self._callable_takes_info = False
        else:
            raise TypeError(
                f"Unsupported external policy type '{type(policy).__name__}'. "
                "Expected SB3 model (.predict), PolicyBundle, torch.nn.Module, or callable."
            )

    def reset(self) -> None:
        """Reset stateful recurrent/external policies at episode boundaries if supported."""
        hook = getattr(self.raw_policy, "reset", None)
        if callable(hook):
            hook()

    def _prepare_obs(self, obs: Any) -> Any:
        if isinstance(self.raw_policy, PolicyBundle) and isinstance(obs, dict):
            return obs
        arr = np.asarray(obs, dtype=np.float32)
        env_already_normed = hasattr(self.env_adapter.raw_env, "obs_rms")
        if self._obs_rms is not None and not env_already_normed:
            norm_fn = getattr(self._vec_normalize, "normalize_obs", None)
            if callable(norm_fn) and self.obs_rms_frozen_on_eval:
                try:
                    return np.asarray(norm_fn(arr), dtype=np.float32)
                except (RuntimeError, ValueError, TypeError):
                    return _apply_obs_rms(arr, self._obs_rms, clip_obs=self._clip_obs)
            return _apply_obs_rms(arr, self._obs_rms, clip_obs=self._clip_obs)
        return arr

    def _format_action(self, raw_act: Any) -> Any:
        if self.env_adapter.is_discrete:
            if isinstance(raw_act, np.ndarray):
                if raw_act.ndim >= 1 and raw_act.size > 1:
                    return int(np.argmax(raw_act))
                return int(raw_act.reshape(-1)[0])
            return int(raw_act)
        if self.env_adapter.action_nvec is not None:
            arr = np.asarray(raw_act, dtype=np.int64).reshape(-1)
            max_n = np.asarray(self.env_adapter.action_nvec, dtype=np.int64) - 1
            return np.clip(arr, 0, max_n)
        arr_f = np.asarray(raw_act, dtype=np.float32).reshape(self.env_adapter.action_dim)
        lo = self.env_adapter.action_low
        hi = self.env_adapter.action_high
        if lo is not None and hi is not None:
            finite = np.isfinite(lo) & np.isfinite(hi)
            if np.any(finite):
                arr_f = np.where(finite, np.clip(arr_f, lo, hi), arr_f)
        return arr_f

    def __call__(self, obs: Any, info: dict[str, Any] | None = None) -> Any:
        info_dict = info or {}
        if isinstance(self.raw_policy, PolicyBundle):
            return self.raw_policy(obs, info_dict)

        obs_in = self._prepare_obs(obs)

        if self.backend.startswith("sb3:"):
            kwargs: dict[str, Any] = {"deterministic": self.deterministic}
            if self._sb3_accepts_masks and "action_mask" in info_dict:
                kwargs["action_masks"] = np.asarray(info_dict["action_mask"], dtype=np.bool_)
            act, _ = self.raw_policy.predict(obs_in, **kwargs)
            return self._format_action(act)

        if isinstance(self.raw_policy, torch.nn.Module):
            with torch.no_grad():
                x = torch.as_tensor(obs_in, dtype=torch.float32, device=self.device)
                if x.ndim == 1 or (x.ndim == 3 and x.shape[-1] in (1, 3, 4)):
                    x = x.unsqueeze(0)
                out = self.raw_policy(x)
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, torch.distributions.Distribution):
                    out = out.mode if hasattr(out, "mode") else out.mean
                out_np = out.detach().cpu().numpy().squeeze(0)
                if self.env_adapter.is_discrete and out_np.ndim >= 1 and out_np.size > 1:
                    if "action_mask" in info_dict:
                        mask = np.asarray(info_dict["action_mask"], dtype=np.bool_).reshape(-1)
                        masked = np.where(mask, out_np.reshape(-1), -1e9)
                        return int(np.argmax(masked))
                    return int(np.argmax(out_np))
                return self._format_action(out_np)

        raw = self.raw_policy(obs_in, info_dict) if self._callable_takes_info else self.raw_policy(obs_in)
        return self._format_action(raw)


def wrap_external_policy(
    policy_or_path: Any,
    env_or_factory: Any,
    *,
    deterministic: bool = True,
    vec_normalize: Any = None,
    obs_rms: Any = None,
    sb3_algo: str | None = None,
    device: str = "cpu",
) -> ExternalPolicyWrapper:
    """Wrap an external policy (SB3, nn.Module, PolicyBundle, checkpoint path, or callable) for evaluation."""
    resolved = _load_external_artifact(policy_or_path, sb3_algo=sb3_algo)
    policy_spec = getattr(resolved, "obs_spec", None)
    needs_structured = bool(policy_spec is not None and getattr(policy_spec, "kind", "vector") != "vector")
    adapter = ensure_gym_env(env_or_factory, structured_obs=needs_structured)
    return ExternalPolicyWrapper(
        resolved,
        adapter,
        deterministic=deterministic,
        vec_normalize=vec_normalize,
        obs_rms=obs_rms,
        device=device,
    )


def evaluate_external_policy(
    env_or_factory: Any,
    policy_or_policies: Any,
    *,
    algo_id: str = "external_policy",
    eval_episodes: int = 10,
    eval_seed: int = 1000,
    deterministic: bool = True,
    vec_normalize: Any = None,
    obs_rms: Any = None,
    sb3_algo: str | None = None,
    device: str = "cpu",
) -> list[TrainResult]:
    """Evaluate one external policy or a sequence of seed checkpoints into TrainResult records."""
    if isinstance(policy_or_policies, (list, tuple)):
        items: list[Any] = list(policy_or_policies)
    else:
        items = [policy_or_policies]

    results: list[TrainResult] = []
    for idx, item in enumerate(items):
        t0 = time.perf_counter()
        wrapper = wrap_external_policy(
            item,
            env_or_factory,
            deterministic=deterministic,
            vec_normalize=vec_normalize,
            obs_rms=obs_rms,
            sb3_algo=sb3_algo,
            device=device,
        )
        recorded_actions: list[np.ndarray] = []

        def _recording_policy(
            obs: Any,
            info: dict[str, Any],
            _w: ExternalPolicyWrapper = wrapper,
            _log: list[np.ndarray] = recorded_actions,
        ) -> Any:
            act = _w(obs, info)
            if not _w.env_adapter.is_discrete and _w.env_adapter.action_nvec is None:
                _log.append(np.asarray(act, dtype=np.float64).reshape(-1))
            return act

        _recording_policy.reset = wrapper.reset  # type: ignore[attr-defined]
        if wrapper.obs_spec is not None:
            _recording_policy.obs_spec = wrapper.obs_spec  # type: ignore[attr-defined]

        eval_stats: EpisodeEvalStats = evaluate_policy_episodes(
            wrapper.env_adapter,
            _recording_policy,
            num_episodes=max(1, int(eval_episodes)),
            seed=eval_seed + idx * 100,
        )
        sat_rate = (
            compute_action_saturation_rate(
                np.vstack(recorded_actions),
                wrapper.env_adapter.action_low,
                wrapper.env_adapter.action_high,
            )
            if recorded_actions
            else 0.0
        )
        num_timesteps = int(getattr(wrapper.raw_policy, "num_timesteps", 0) or 0)
        elapsed = time.perf_counter() - t0
        is_cont = not wrapper.env_adapter.is_discrete and wrapper.env_adapter.action_nvec is None
        results.append(
            TrainResult(
                algo_id=algo_id,
                seed=idx,
                total_timesteps=num_timesteps,
                elapsed_seconds=round(float(elapsed), 4),
                policy_fn=wrapper,
                model=wrapper.raw_policy,
                config={
                    "external_backend": wrapper.backend,
                    "deterministic": deterministic,
                    "eval_episodes": eval_episodes,
                },
                learning_curve=[{"step": float(num_timesteps), "episode_return": float(eval_stats.mean_return)}],
                train_diagnostics={
                    "is_continuous": float(is_cont),
                    "action_saturation_rate": round(float(sat_rate), 4),
                    "external_backend": wrapper.backend,
                },
                eval_stats=eval_stats,
                obs_rms_frozen_on_eval=wrapper.obs_rms_frozen_on_eval,
                optimizations_used=(["obs_normalization"] if wrapper._obs_rms is not None else []),
                extra={"external": True, "eval_seed": eval_seed + idx * 100},
            )
        )
    return results


def audit_external_policy(
    env_or_factory: Any,
    policy_or_policies: Any,
    *,
    algo_id: str = "external_policy",
    eval_episodes: int = 10,
    eval_seed: int = 1000,
    min_seeds: int = 3,
    preflight_report: PreflightReport | None = None,
    heuristic_fn: Callable[[np.ndarray, dict[str, Any]], Any] | None = None,
    deterministic: bool = True,
    vec_normalize: Any = None,
    obs_rms: Any = None,
    sb3_algo: str | None = None,
    device: str = "cpu",
) -> RigorAuditReport:
    """Run Stage 1 preflight (if omitted) + external evaluation + Stage 3 Rigor Audit."""
    pf = preflight_report if preflight_report is not None else preflight_env(env_or_factory, heuristic_fn=heuristic_fn)
    runs = evaluate_external_policy(
        env_or_factory,
        policy_or_policies,
        algo_id=algo_id,
        eval_episodes=eval_episodes,
        eval_seed=eval_seed,
        deterministic=deterministic,
        vec_normalize=vec_normalize,
        obs_rms=obs_rms,
        sb3_algo=sb3_algo,
        device=device,
    )
    return audit_drl_rigor(
        runs,
        preflight_report=pf,
        random_baseline=pf.random_baseline,
        heuristic_baseline=pf.heuristic_baseline,
        min_seeds=min_seeds,
    )
