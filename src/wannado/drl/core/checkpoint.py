"""Atomic checkpointing, deterministic resume, live progress logging, and standalone PolicyBundle inference."""

from __future__ import annotations

import csv
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from wannado.drl.core.networks import (
    CategoricalQNetwork,
    ContinuousActorCritic,
    ContinuousGaussianActor,
    DeterministicActor,
    DiscreteActorCritic,
    DiscretePolicyNet,
    DuelingQNetwork,
    MultiDiscreteActorCritic,
    QNetwork,
    RecurrentActorCriticGRU,
    SquashedGaussianActor,
)
from wannado.drl.core.obs_spec import ObsSpec, format_obs_single, obs_to_tensor
from wannado.drl.core.utils import eval_mode, resolve_device
from wannado.drl.core.wrappers import DictRunningMeanStd, RewardNormalizer

_POLICY_NET_CLASSES: dict[str, type[nn.Module]] = {
    "DiscreteActorCritic": DiscreteActorCritic,
    "MultiDiscreteActorCritic": MultiDiscreteActorCritic,
    "ContinuousActorCritic": ContinuousActorCritic,
    "RecurrentActorCriticGRU": RecurrentActorCriticGRU,
    "QNetwork": QNetwork,
    "DuelingQNetwork": DuelingQNetwork,
    "CategoricalQNetwork": CategoricalQNetwork,
    "DeterministicActor": DeterministicActor,
    "SquashedGaussianActor": SquashedGaussianActor,
    "DiscretePolicyNet": DiscretePolicyNet,
    "ContinuousGaussianActor": ContinuousGaussianActor,
}


def _atomic_torch_save(payload: Any, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)
    return target_path


def _serialize_action_meta(env: Any) -> dict[str, Any]:
    return {
        "is_discrete": bool(env.is_discrete),
        "raw_action_kind": str(env.raw_action_kind),
        "action_dim": int(env.action_dim),
        "action_nvec": list(env.action_nvec) if env.action_nvec is not None else None,
        "action_low": env.action_low.tolist() if env.action_low is not None else None,
        "action_high": env.action_high.tolist() if env.action_high is not None else None,
    }


def _reconstruct_policy_module(policy_spec: dict[str, Any], obs_spec: ObsSpec, action_meta: dict[str, Any], device: torch.device) -> nn.Module:
    cls_name = str(policy_spec["net_class"])
    cls = _POLICY_NET_CLASSES.get(cls_name)
    if cls is None:
        raise ValueError(f"Unsupported policy network class in checkpoint: {cls_name!r}")
    kwargs = dict(policy_spec.get("init_kwargs", {}))
    if "action_low" in kwargs and kwargs["action_low"] is not None:
        kwargs["action_low"] = np.asarray(kwargs["action_low"], dtype=np.float32)
    elif cls_name in (
        "ContinuousActorCritic",
        "ContinuousGaussianActor",
        "DeterministicActor",
        "SquashedGaussianActor",
    ) and action_meta.get("action_low") is not None:
        kwargs.setdefault("action_low", np.asarray(action_meta["action_low"], dtype=np.float32))
    if "action_high" in kwargs and kwargs["action_high"] is not None:
        kwargs["action_high"] = np.asarray(kwargs["action_high"], dtype=np.float32)
    elif cls_name in (
        "ContinuousActorCritic",
        "ContinuousGaussianActor",
        "DeterministicActor",
        "SquashedGaussianActor",
    ) and action_meta.get("action_high") is not None:
        kwargs.setdefault("action_high", np.asarray(action_meta["action_high"], dtype=np.float32))
    if "action_nvec" in kwargs and kwargs["action_nvec"] is not None:
        kwargs["action_nvec"] = tuple(int(v) for v in kwargs["action_nvec"])
    return cls(obs_spec, **kwargs).to(device)


class PolicyBundle:
    """Self-contained inference wrapper pairing a trained policy network with frozen obs preprocessing."""

    def __init__(
        self,
        *,
        algo_id: str,
        policy_kind: str,
        model: nn.Module,
        obs_spec: ObsSpec,
        action_meta: dict[str, Any],
        obs_rms: DictRunningMeanStd | None = None,
        policy_spec: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        device: torch.device | str = "cpu",
        freeze_obs_rms: bool = True,
    ) -> None:
        self.algo_id = str(algo_id)
        self.policy_kind = str(policy_kind)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.obs_spec = obs_spec
        self.action_meta = dict(action_meta)
        self.obs_rms = obs_rms
        if self.obs_rms is not None and freeze_obs_rms:
            self.obs_rms.freeze(True)
        self.policy_spec = dict(policy_spec or {})
        self.config = dict(config or {})
        self._gru_hidden: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset recurrent hidden state (if GRU-backed) between evaluation episodes."""
        self._gru_hidden = None

    def _prepare_single_obs(self, obs_in: Any) -> Any:
        # Format raw unnormalized HWC/2D/Dict observations transparently while avoiding double-transpose
        if self.obs_spec.kind == "vector" and isinstance(obs_in, dict):
            obs_fmt: Any = format_obs_single(obs_in, self.obs_spec)
        elif self.obs_spec.kind == "dict" and isinstance(obs_in, dict):
            needs_fmt = False
            for k, leaf in self.obs_spec.leaves.items():
                if k in obs_in and leaf.kind == "image":
                    arr_sh = tuple(np.asarray(obs_in[k]).shape)
                    if (not leaf.channel_first and arr_sh == leaf.raw_shape) or (len(leaf.raw_shape) == 2 and len(arr_sh) == 2):
                        needs_fmt = True
                        break
            obs_fmt = (
                format_obs_single(obs_in, self.obs_spec)
                if needs_fmt
                else {
                    k: np.asarray(obs_in[k], dtype=np.uint8 if leaf.dtype == "uint8" else np.float32)
                    for k, leaf in self.obs_spec.leaves.items()
                    if k in obs_in
                }
            )
        elif self.obs_spec.kind == "image":
            arr = np.asarray(obs_in)
            leaf = self.obs_spec.leaves["obs"]
            if (not leaf.channel_first and arr.shape == leaf.raw_shape) or (len(leaf.raw_shape) == 2 and arr.ndim == 2):
                obs_fmt = format_obs_single(arr, self.obs_spec)
            else:
                obs_fmt = np.asarray(arr, dtype=np.uint8 if leaf.dtype == "uint8" else np.float32)
        elif self.obs_spec.kind == "entities":
            obs_fmt = np.asarray(obs_in, dtype=np.float32).reshape(self.obs_spec.shape)
        else:
            obs_fmt = np.asarray(obs_in, dtype=np.float32).reshape(-1)
        if self.obs_rms is not None:
            return self.obs_rms.normalize(obs_fmt)
        return obs_fmt

    def __call__(self, obs_in: Any, info_in: dict[str, Any] | None = None) -> Any:
        return self.predict(obs_in, info_in=info_in, deterministic=True)

    def predict(
        self,
        obs_in: Any,
        info_in: dict[str, Any] | None = None,
        *,
        deterministic: bool = True,
    ) -> Any:
        obs_proc = self._prepare_single_obs(obs_in)
        obs_t = obs_to_tensor(obs_proc, self.obs_spec, self.device, add_batch_dim=True)
        mask_in = (info_in or {}).get("action_mask")
        is_discrete = bool(self.action_meta.get("is_discrete", False))
        is_multi = self.action_meta.get("action_nvec") is not None

        with eval_mode(self.model), torch.no_grad():
            if self.policy_kind == "actor_critic":
                if is_discrete or is_multi:
                    m_t = torch.as_tensor(mask_in, device=self.device, dtype=torch.bool).unsqueeze(0) if mask_in is not None else None
                    act_t, _, _, _ = self.model.get_action_and_value(obs_t, action_mask=m_t, deterministic=deterministic)  # type: ignore[operator]
                    return int(act_t.item()) if is_discrete else act_t.squeeze(0).cpu().numpy()
                act_t, _, _, _ = self.model.get_action_and_value(obs_t, deterministic=deterministic)  # type: ignore[operator]
                return act_t.squeeze(0).cpu().numpy()

            if self.policy_kind == "gru_actor_critic":
                gru_net: RecurrentActorCriticGRU = self.model  # type: ignore[assignment]
                if self._gru_hidden is None:
                    self._gru_hidden = torch.zeros(1, 1, gru_net.hidden_dim, device=self.device)
                m_t = torch.as_tensor(mask_in, device=self.device, dtype=torch.bool).unsqueeze(0) if ((is_discrete or is_multi) and mask_in is not None) else None
                act_t, _, _, _, self._gru_hidden = gru_net.step_action_value(
                    obs_t, self._gru_hidden, action_mask=m_t, deterministic=deterministic
                )
                return int(act_t.item()) if is_discrete else act_t.squeeze(0).cpu().numpy()

            if self.policy_kind in ("q_network", "c51_q", "discrete_policy"):
                q_vals = (
                    self.model.get_q_values(obs_t).squeeze(0)  # type: ignore[operator]
                    if self.policy_kind == "c51_q"
                    else self.model(obs_t).squeeze(0)
                )
                if mask_in is not None:
                    m = torch.as_tensor(mask_in, device=self.device, dtype=torch.bool).view_as(q_vals)
                    if bool(m.any()):
                        q_vals = q_vals.masked_fill(~m, -1e9)
                if not deterministic:
                    return int(torch.distributions.Categorical(logits=q_vals).sample().item())
                return int(torch.argmax(q_vals).item())

            if self.policy_kind == "squashed_actor":
                act_t, _ = self.model.get_action(obs_t, deterministic=deterministic)  # type: ignore[operator]
                return act_t.squeeze(0).cpu().numpy()

            if self.policy_kind == "deterministic_actor":
                return self.model(obs_t).squeeze(0).cpu().numpy()

            if self.policy_kind == "gaussian_actor":
                if not deterministic and hasattr(self.model, "get_dist"):
                    return self.model.get_dist(obs_t).sample().squeeze(0).cpu().numpy()  # type: ignore[operator]
                return self.model(obs_t).squeeze(0).cpu().numpy()

        raise ValueError(f"Unsupported policy_kind={self.policy_kind!r}")

    def discrete_logits(self, obs_in: Any) -> torch.Tensor:
        """Return raw discrete action logits (or Q-values) for discrete policies on single observations."""
        obs_proc = self._prepare_single_obs(obs_in)
        obs_t = obs_to_tensor(obs_proc, self.obs_spec, self.device, add_batch_dim=True)
        if self.policy_kind == "actor_critic":
            dist = self.model.get_dist(obs_t)  # type: ignore[operator]
            if hasattr(dist, "logits"):
                return dist.logits
        elif self.policy_kind in ("q_network", "discrete_policy"):
            return self.model(obs_t)
        elif self.policy_kind == "c51_q":
            return self.model.get_q_values(obs_t)  # type: ignore[operator]
        elif self.policy_kind == "gru_actor_critic":
            gru_net: RecurrentActorCriticGRU = self.model  # type: ignore[assignment]
            h = self._gru_hidden if self._gru_hidden is not None else torch.zeros(1, 1, gru_net.hidden_dim, device=self.device)
            obs_seq: Any = {k: v.unsqueeze(0) for k, v in obs_t.items()} if isinstance(obs_t, dict) else obs_t.unsqueeze(0)
            feat, _ = gru_net.forward_seq(obs_seq, h)
            return gru_net.actor(feat.squeeze(0))
        raise ValueError(f"discrete_logits() is only supported for discrete policies, got policy_kind={self.policy_kind!r}")

    def state_payload(self) -> dict[str, Any]:
        return {
            "format": "wannado_drl_policy_bundle_v1",
            "algo_id": self.algo_id,
            "policy_kind": self.policy_kind,
            "obs_spec": self.obs_spec.to_dict(),
            "action_meta": dict(self.action_meta),
            "obs_rms_state": self.obs_rms.state_dict() if self.obs_rms is not None else None,
            "policy_spec": dict(self.policy_spec),
            "policy_state_dict": self.model.state_dict(),
            "config": dict(self.config),
        }

    def save(self, path: str | Path) -> Path:
        return _atomic_torch_save(self.state_payload(), Path(path))

    @classmethod
    def from_payload(cls, payload: dict[str, Any], device: str | torch.device = "auto") -> PolicyBundle:
        dev = resolve_device(device)
        obs_spec = ObsSpec.from_dict(payload["obs_spec"])
        action_meta = dict(payload["action_meta"])
        policy_spec = payload.get("policy_spec")
        if policy_spec is None:
            raise ValueError("Checkpoint payload does not contain 'policy_spec' for standalone inference.")
        model = _reconstruct_policy_module(policy_spec, obs_spec, action_meta, dev)
        state_dict = payload.get("policy_state_dict") or payload["models"][payload.get("primary_model_key", "policy")]
        model.load_state_dict(state_dict)
        model.eval()
        obs_rms_state = payload.get("obs_rms_state")
        obs_rms: DictRunningMeanStd | None = None
        if obs_rms_state is not None:
            obs_rms = DictRunningMeanStd(obs_spec)
            obs_rms.load_state_dict(obs_rms_state)
            obs_rms.freeze(True)
        return cls(
            algo_id=str(payload["algo_id"]),
            policy_kind=str(payload["policy_kind"]),
            model=model,
            obs_spec=obs_spec,
            action_meta=action_meta,
            obs_rms=obs_rms,
            policy_spec=policy_spec,
            config=payload.get("config"),
            device=dev,
        )


def load_policy(path: str | Path, *, device: str | torch.device = "auto") -> PolicyBundle:
    """Load any saved ``best_model.pt``, ``last_checkpoint.pt``, or exported ``PolicyBundle`` for inference."""
    dev = resolve_device(device)
    payload = torch.load(Path(path), map_location=dev, weights_only=False)
    return PolicyBundle.from_payload(payload, device=dev)


class CheckpointManager:
    """Manages atomic training checkpoints, best-model selection, deterministic resume, and progress logs."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        keep_last_k: int = 3,
        save_best_model: bool = True,
        tb_log_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = max(1, int(keep_last_k))
        self.save_best_model = bool(save_best_model)
        self.best_eval_mean: float = float("-inf")
        self.best_model_path: Path | None = None
        self.last_checkpoint_path: Path | None = None
        self.eval_history: list[dict[str, float]] = []
        self._step_ckpts: list[Path] = []
        self._csv_header_written = False
        self._tb_writer: Any = None
        if tb_log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            except Exception:  # noqa: BLE001 — tensorboard stays strictly optional
                self._tb_writer = None

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        clean = {k: float(v) for k, v in scalars.items() if isinstance(v, (int, float)) and np.isfinite(float(v))}
        if not clean:
            return
        if self.root is not None:
            rec = {"step": int(step), **clean}
            with (self.root / "progress.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            csv_path = self.root / "progress.csv"
            if not csv_path.exists():
                self._csv_header_written = False
            if not self._csv_header_written:
                with csv_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["step", *sorted(clean.keys())])
                    writer.writerow([int(step), *(clean[k] for k in sorted(clean.keys()))])
                self._csv_header_written = True
            else:
                with csv_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([int(step), *(clean[k] for k in sorted(clean.keys()))])
        if self._tb_writer is not None:
            for k, v in clean.items():
                self._tb_writer.add_scalar(k, v, int(step))

    def save_checkpoint(
        self,
        step: int,
        *,
        algo_id: str,
        policy_kind: str,
        policy_spec: dict[str, Any],
        primary_model_key: str,
        models: dict[str, Any],
        optimizers: Mapping[str, torch.optim.Optimizer],
        obs_spec: ObsSpec,
        action_meta: dict[str, Any],
        config: dict[str, Any],
        obs_rms: DictRunningMeanStd | None = None,
        ret_rms: RewardNormalizer | None = None,
        rng: np.random.Generator | None = None,
        replay_buffer: Any | None = None,
        save_replay_buffer: bool = False,
        extra_state: dict[str, Any] | None = None,
        tag: str = "last",
    ) -> Path | None:
        if self.root is None:
            return None
        model_states: dict[str, Any] = {}
        for k, obj in models.items():
            if isinstance(obj, nn.Module):
                model_states[k] = obj.state_dict()
            elif isinstance(obj, torch.Tensor):
                model_states[k] = obj.detach().cpu()

        rng_state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy_legacy": np.random.get_state(),
            "numpy_gen": rng.bit_generator.state if rng is not None else None,
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

        payload: dict[str, Any] = {
            "format": "wannado_drl_checkpoint_v1",
            "step": int(step),
            "algo_id": algo_id,
            "policy_kind": policy_kind,
            "policy_spec": dict(policy_spec),
            "primary_model_key": primary_model_key,
            "models": model_states,
            "optimizers": {k: opt.state_dict() for k, opt in optimizers.items()},
            "obs_spec": obs_spec.to_dict(),
            "action_meta": dict(action_meta),
            "config": dict(config),
            "obs_rms_state": obs_rms.state_dict() if obs_rms is not None else None,
            "ret_rms_state": ret_rms.state_dict() if ret_rms is not None else None,
            "rng_state": rng_state,
            "best_eval_mean": float(self.best_eval_mean),
            "eval_history": list(self.eval_history),
            "extra_state": dict(extra_state or {}),
        }

        if tag == "best":
            target = self.root / "best_model.pt"
            _atomic_torch_save(payload, target)
            self.best_model_path = target
            return target

        if tag == "step":
            step_path = self.root / f"step_{int(step):09d}.pt"
            _atomic_torch_save(payload, step_path)
            self._step_ckpts.append(step_path)
            while len(self._step_ckpts) > self.keep_last_k:
                old = self._step_ckpts.pop(0)
                if old.exists() and old != step_path:
                    old.unlink()

        last_path = self.root / "last_checkpoint.pt"
        _atomic_torch_save(payload, last_path)
        self.last_checkpoint_path = last_path
        if save_replay_buffer and replay_buffer is not None and hasattr(replay_buffer, "state_dict"):
            _atomic_torch_save(replay_buffer.state_dict(), self.root / "last_replay_buffer.pt")
        return last_path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        models: dict[str, Any],
        optimizers: Mapping[str, torch.optim.Optimizer],
        obs_rms: DictRunningMeanStd | None = None,
        ret_rms: RewardNormalizer | None = None,
        rng: np.random.Generator | None = None,
        replay_buffer: Any | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        ckpt_path = Path(path).resolve()
        payload = torch.load(ckpt_path, map_location=torch.device(device), weights_only=False)
        for k, obj in models.items():
            if k in payload["models"]:
                if isinstance(obj, nn.Module):
                    obj.load_state_dict(payload["models"][k])
                elif isinstance(obj, torch.Tensor):
                    with torch.no_grad():
                        obj.copy_(payload["models"][k].to(obj.device))
        for k, opt in optimizers.items():
            if k in payload.get("optimizers", {}):
                opt.load_state_dict(payload["optimizers"][k])
        if obs_rms is not None and payload.get("obs_rms_state") is not None:
            obs_rms.load_state_dict(payload["obs_rms_state"])
        if ret_rms is not None and payload.get("ret_rms_state") is not None:
            ret_rms.load_state_dict(payload["ret_rms_state"])

        rng_state = payload.get("rng_state", {})
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy_legacy") is not None:
            np.random.set_state(rng_state["numpy_legacy"])
        if rng is not None and rng_state.get("numpy_gen") is not None:
            rng.bit_generator.state = rng_state["numpy_gen"]
        if rng_state.get("torch_cpu") is not None:
            torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

        replay_sidecar = ckpt_path.with_name("last_replay_buffer.pt")
        if replay_buffer is not None and replay_sidecar.exists() and hasattr(replay_buffer, "load_state_dict"):
            rep_state = torch.load(replay_sidecar, map_location="cpu", weights_only=False)
            replay_buffer.load_state_dict(rep_state)

        self.best_eval_mean = float(payload.get("best_eval_mean", float("-inf")))
        self.eval_history = list(payload.get("eval_history", []))
        return {
            "step": int(payload.get("step", 0)),
            "extra_state": dict(payload.get("extra_state", {})),
        }

    def run_periodic_eval(
        self,
        step: int,
        *,
        eval_env: Any,
        policy_fn: Any,
        obs_rms: DictRunningMeanStd | None,
        num_episodes: int,
        seed: int,
        algo_id: str,
        policy_kind: str,
        policy_spec: dict[str, Any],
        primary_model_key: str,
        models: dict[str, Any],
        optimizers: Mapping[str, torch.optim.Optimizer],
        obs_spec: ObsSpec,
        action_meta: dict[str, Any],
        config: dict[str, Any],
        ret_rms: RewardNormalizer | None = None,
        rng: np.random.Generator | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        prev_frozen = obs_rms.frozen if obs_rms is not None else True
        if obs_rms is not None:
            obs_rms.freeze(True)
        try:
            from wannado.drl.mdp.baselines import evaluate_policy_episodes

            est = evaluate_policy_episodes(eval_env, policy_fn, num_episodes=num_episodes, seed=seed)
        finally:
            if obs_rms is not None:
                obs_rms.freeze(prev_frozen)

        rec = {
            "step": float(step),
            "eval_mean": float(est.mean_return),
            "eval_std": float(est.std_return),
            "eval_mean_length": float(est.mean_length),
        }
        self.eval_history.append(rec)
        self.log_scalars(step, {"eval/mean_return": est.mean_return, "eval/std_return": est.std_return})
        if self.save_best_model and est.mean_return >= self.best_eval_mean:
            self.best_eval_mean = float(est.mean_return)
            self.save_checkpoint(
                step,
                algo_id=algo_id,
                policy_kind=policy_kind,
                policy_spec=policy_spec,
                primary_model_key=primary_model_key,
                models=models,
                optimizers=optimizers,
                obs_spec=obs_spec,
                action_meta=action_meta,
                config=config,
                obs_rms=obs_rms,
                ret_rms=ret_rms,
                rng=rng,
                extra_state=extra_state,
                tag="best",
            )
        return rec

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None


def action_meta_from_env(env: Any) -> dict[str, Any]:
    """Extract serializable action-space metadata from UniversalGymAdapter or VectorizedEnvAdapter."""
    return _serialize_action_meta(env)
