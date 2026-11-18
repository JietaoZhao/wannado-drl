"""Vectorized & multimodal RolloutBuffer, ReplayBuffer (uint8/Dict aware), and DualDemoReplayBuffer."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from wannado.drl.core.obs_spec import (
    ObsSpec,
    concat_obs_tensors,
    obs_to_tensor,
    slice_obs,
    vector_obs_spec,
)


def _resolve_spec(obs_dim_or_spec: int | ObsSpec) -> ObsSpec:
    if isinstance(obs_dim_or_spec, ObsSpec):
        return obs_dim_or_spec
    return vector_obs_spec(int(obs_dim_or_spec))


def _alloc_obs_storage(leading_shape: tuple[int, ...], spec: ObsSpec) -> Any:
    if spec.kind == "dict":
        return {
            k: np.zeros(
                (*leading_shape, *leaf.shape),
                dtype=np.uint8 if leaf.dtype == "uint8" else np.float32,
            )
            for k, leaf in spec.leaves.items()
        }
    dt = np.uint8 if spec.dtype == "uint8" else np.float32
    return np.zeros((*leading_shape, *spec.shape), dtype=dt)


def _write_obs_slot(storage: Any, idx: Any, value: Any, spec: ObsSpec) -> None:
    if spec.kind == "dict" and isinstance(storage, dict):
        for k in storage:
            storage[k][idx] = value[k]
    else:
        storage[idx] = value


class RolloutBuffer:
    """On-policy trajectory buffer supporting both legacy 1D and vectorized multimodal ``(T, N, ...)`` rollouts."""

    def __init__(
        self,
        num_steps: int,
        obs_dim: int | ObsSpec,
        action_shape: tuple[int, ...],
        action_dtype: np.dtype[Any] | None = None,
        mask_dim: int | None = None,
        *,
        num_envs: int = 1,
        batched: bool = False,
    ) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = max(1, int(num_envs))
        self.obs_spec = _resolve_spec(obs_dim)
        self.batched = bool(batched or self.num_envs > 1 or self.obs_spec.kind != "vector")
        resolved_action_dtype = np.dtype(np.float32) if action_dtype is None else np.dtype(action_dtype)

        leading: tuple[int, ...] = (self.num_steps, self.num_envs) if self.batched else (self.num_steps,)
        self.obs: Any = _alloc_obs_storage(leading, self.obs_spec)
        self.actions = np.zeros((*leading, *action_shape), dtype=resolved_action_dtype)
        self.logprobs = np.zeros(leading, dtype=np.float32)
        self.rewards = np.zeros(leading, dtype=np.float32)
        self.terminateds = np.zeros(leading, dtype=np.float32)
        self.dones = np.zeros(leading, dtype=np.float32)
        self.values = np.zeros(leading, dtype=np.float32)
        self.advantages = np.zeros(leading, dtype=np.float32)
        self.returns = np.zeros(leading, dtype=np.float32)
        self.valid_masks = np.ones(leading, dtype=np.float32)
        self.action_masks = (
            np.ones((*leading, mask_dim), dtype=np.bool_) if mask_dim is not None else None
        )
        self.ptr = 0

    def add(
        self,
        obs: Any,
        action: np.ndarray | int,
        logprob: float,
        reward: float,
        terminated: bool,
        value: float,
        action_mask: np.ndarray | None = None,
        done: bool | None = None,
    ) -> None:
        idx = self.ptr
        if idx >= self.num_steps:
            raise IndexError("RolloutBuffer overflow; call reset() before adding more steps.")
        if self.batched:
            _write_obs_slot(self.obs, (idx, 0), obs, self.obs_spec)
            self.actions[idx, 0] = action
            self.logprobs[idx, 0] = logprob
            self.rewards[idx, 0] = reward
            self.terminateds[idx, 0] = float(terminated)
            self.dones[idx, 0] = float(terminated if done is None else done)
            self.values[idx, 0] = value
            self.valid_masks[idx, 0] = 1.0
            if self.action_masks is not None and action_mask is not None:
                self.action_masks[idx, 0] = action_mask.astype(np.bool_)
        else:
            self.obs[idx] = obs
            self.actions[idx] = action
            self.logprobs[idx] = logprob
            self.rewards[idx] = reward
            self.terminateds[idx] = float(terminated)
            self.dones[idx] = float(terminated if done is None else done)
            self.values[idx] = value
            if self.action_masks is not None and action_mask is not None:
                self.action_masks[idx] = action_mask.astype(np.bool_)
        self.ptr += 1

    def add_batch(
        self,
        obs: Any,
        actions: np.ndarray,
        logprobs: np.ndarray,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        values: np.ndarray,
        *,
        action_masks: np.ndarray | None = None,
        dones: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        idx = self.ptr
        if idx >= self.num_steps:
            raise IndexError("RolloutBuffer overflow; call reset() before adding more steps.")
        _write_obs_slot(self.obs, idx, obs, self.obs_spec)
        self.actions[idx] = actions
        self.logprobs[idx] = logprobs
        self.rewards[idx] = rewards
        self.terminateds[idx] = np.asarray(terminateds, dtype=np.float32)
        self.dones[idx] = self.terminateds[idx] if dones is None else np.asarray(dones, dtype=np.float32)
        self.values[idx] = values
        if valid_mask is not None:
            self.valid_masks[idx] = np.asarray(valid_mask, dtype=np.float32)
        else:
            self.valid_masks[idx] = 1.0
        if self.action_masks is not None and action_masks is not None:
            self.action_masks[idx] = np.asarray(action_masks, dtype=np.bool_)
        self.ptr += 1

    def compute_returns_and_advantage(
        self,
        next_value: float | np.ndarray,
        next_terminated: bool | np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute vectorized GAE across ``(T,)`` or ``(T, N)`` without cross-episode leakage."""
        n = self.ptr
        if n == 0:
            return
        if not self.batched:
            last_gae = 0.0
            nv_scalar = float(np.asarray(next_value).reshape(-1)[0])
            nt_bool = bool(np.asarray(next_terminated).reshape(-1)[0])
            for t in reversed(range(n)):
                if t == n - 1:
                    next_val = nv_scalar
                    nonterminal = 1.0 - float(bool(self.dones[t]) or nt_bool)
                else:
                    next_val = float(self.values[t + 1])
                    nonterminal = 1.0 - float(self.dones[t])
                delta = float(self.rewards[t]) + gamma * next_val * nonterminal - float(self.values[t])
                last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
                self.advantages[t] = last_gae
            self.returns[:n] = self.advantages[:n] + self.values[:n]
            return

        last_gae_vec = np.zeros((self.num_envs,), dtype=np.float32)
        nv_vec = np.asarray(next_value, dtype=np.float32).reshape(self.num_envs)
        nt_vec = np.asarray(next_terminated, dtype=np.bool_).reshape(self.num_envs).astype(np.float32)
        for t in reversed(range(n)):
            if t == n - 1:
                next_val_v = nv_vec
                nonterminal_v = 1.0 - np.maximum(self.dones[t], nt_vec)
            else:
                next_val_v = self.values[t + 1]
                nonterminal_v = 1.0 - self.dones[t]
            valid_t = self.valid_masks[t]
            delta_v = (self.rewards[t] + gamma * next_val_v * nonterminal_v - self.values[t]) * valid_t
            last_gae_vec = np.where(
                valid_t > 0.5,
                delta_v + gamma * gae_lambda * nonterminal_v * last_gae_vec,
                last_gae_vec,
            )
            self.advantages[t] = last_gae_vec
        self.returns[:n] = (self.advantages[:n] + self.values[:n]) * self.valid_masks[:n]

    def flatten_batch(self, device: torch.device) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Flatten valid rollout entries into ``(obs, actions, logprobs, advantages, returns, masks, valid)``."""
        n = self.ptr
        if self.batched:
            valid_flat = self.valid_masks[:n].reshape(-1) > 0.5
            if self.obs_spec.kind == "dict" and isinstance(self.obs, dict):
                obs_flat: Any = {
                    k: v[:n].reshape(n * self.num_envs, *v.shape[2:])[valid_flat]
                    for k, v in self.obs.items()
                }
            else:
                obs_flat = self.obs[:n].reshape(n * self.num_envs, *self.obs.shape[2:])[valid_flat]
            act_flat = self.actions[:n].reshape(n * self.num_envs, *self.actions.shape[2:])[valid_flat]
            lp_flat = self.logprobs[:n].reshape(-1)[valid_flat]
            adv_flat = self.advantages[:n].reshape(-1)[valid_flat]
            ret_flat = self.returns[:n].reshape(-1)[valid_flat]
            mask_flat = (
                self.action_masks[:n].reshape(n * self.num_envs, self.action_masks.shape[-1])[valid_flat]
                if self.action_masks is not None
                else None
            )
        else:
            obs_flat = slice_obs(self.obs, slice(0, n))
            act_flat = self.actions[:n]
            lp_flat = self.logprobs[:n]
            adv_flat = self.advantages[:n]
            ret_flat = self.returns[:n]
            mask_flat = self.action_masks[:n] if self.action_masks is not None else None

        obs_t = obs_to_tensor(obs_flat, self.obs_spec, device, add_batch_dim=False)
        act_t = torch.as_tensor(act_flat, device=device)
        lp_t = torch.as_tensor(lp_flat, device=device, dtype=torch.float32)
        adv_t = torch.as_tensor(adv_flat, device=device, dtype=torch.float32)
        ret_t = torch.as_tensor(ret_flat, device=device, dtype=torch.float32)
        mask_t = torch.as_tensor(mask_flat, device=device, dtype=torch.bool) if mask_flat is not None else None
        val_t = torch.as_tensor(self.values[:n].reshape(-1)[valid_flat] if self.batched else self.values[:n], device=device, dtype=torch.float32)
        return obs_t, act_t, lp_t, adv_t, ret_t, mask_t, val_t

    def iter_minibatches(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> Iterator[tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """Yield shuffled minibatch tensors without converting whole uint8 image buffers prematurely."""
        obs_t, act_t, lp_t, adv_t, ret_t, mask_t, _ = self.flatten_batch(device)
        total = int(lp_t.shape[0])
        if total < 2:
            return
        mb_size = min(int(batch_size), total)
        indices = np.arange(total)
        rng.shuffle(indices)
        for start in range(0, total, mb_size):
            idx = indices[start : start + mb_size]
            if len(idx) < 2:
                continue
            idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
            yield (
                slice_obs(obs_t, idx_t),
                act_t[idx_t],
                lp_t[idx_t],
                adv_t[idx_t],
                ret_t[idx_t],
                mask_t[idx_t] if mask_t is not None else None,
            )

    def reset(self) -> None:
        self.ptr = 0
        self.dones.fill(0.0)
        self.valid_masks.fill(1.0)


class ReplayBuffer:
    """Pre-allocated circular replay buffer supporting vectorized batch writes, uint8 images, and Dict obs."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int | ObsSpec,
        action_dim: int,
        discrete_action: bool = False,
        device: torch.device | str = "cpu",
        *,
        track_costs: bool = False,
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.discrete_action = discrete_action
        self.track_costs = bool(track_costs)
        self.obs_spec = _resolve_spec(obs_dim)
        self.obs: Any = _alloc_obs_storage((self.capacity,), self.obs_spec)
        self.next_obs: Any = _alloc_obs_storage((self.capacity,), self.obs_spec)
        act_dtype = np.int64 if discrete_action else np.float32
        act_shape = (self.capacity,) if discrete_action else (self.capacity, action_dim)
        self.actions = np.zeros(act_shape, dtype=act_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.terminateds = np.zeros((self.capacity,), dtype=np.float32)
        self.costs = np.zeros((self.capacity,), dtype=np.float32) if self.track_costs else None
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: Any,
        action: np.ndarray | int,
        reward: float,
        next_obs: Any,
        terminated: bool,
        cost: float = 0.0,
    ) -> None:
        _write_obs_slot(self.obs, self.ptr, obs, self.obs_spec)
        _write_obs_slot(self.next_obs, self.ptr, next_obs, self.obs_spec)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.terminateds[self.ptr] = float(terminated)
        if self.costs is not None:
            self.costs[self.ptr] = float(cost)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: Any,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: Any,
        terminateds: np.ndarray,
        valid_mask: np.ndarray | None = None,
        costs: np.ndarray | None = None,
    ) -> int:
        """Append a batch of vectorized transitions (skipping invalid autoreset dummy steps)."""
        rew_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
        term_arr = np.asarray(terminateds, dtype=np.float32).reshape(-1)
        act_arr = np.asarray(actions)
        cost_arr = np.asarray(costs, dtype=np.float32).reshape(-1) if costs is not None else None
        if valid_mask is not None:
            vmask = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
            if not np.any(vmask):
                return 0
            obs = slice_obs(obs, vmask)
            next_obs = slice_obs(next_obs, vmask)
            act_arr = act_arr[vmask]
            rew_arr = rew_arr[vmask]
            term_arr = term_arr[vmask]
            if cost_arr is not None:
                cost_arr = cost_arr[vmask]

        n = int(rew_arr.shape[0])
        if n == 0:
            return 0
        if n >= self.capacity:
            keep = slice(n - self.capacity, n)
            _write_obs_slot(self.obs, slice(None), slice_obs(obs, keep), self.obs_spec)
            _write_obs_slot(self.next_obs, slice(None), slice_obs(next_obs, keep), self.obs_spec)
            self.actions[:] = act_arr[keep]
            self.rewards[:] = rew_arr[keep]
            self.terminateds[:] = term_arr[keep]
            if self.costs is not None and cost_arr is not None:
                self.costs[:] = cost_arr[keep]
            self.ptr = 0
            self.size = self.capacity
            return n

        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            _write_obs_slot(self.obs, sl, obs, self.obs_spec)
            _write_obs_slot(self.next_obs, sl, next_obs, self.obs_spec)
            self.actions[sl] = act_arr
            self.rewards[sl] = rew_arr
            self.terminateds[sl] = term_arr
            if self.costs is not None and cost_arr is not None:
                self.costs[sl] = cost_arr
        else:
            first = self.capacity - self.ptr
            rem = n - first
            sl1 = slice(self.ptr, self.capacity)
            sl2 = slice(0, rem)
            _write_obs_slot(self.obs, sl1, slice_obs(obs, slice(0, first)), self.obs_spec)
            _write_obs_slot(self.obs, sl2, slice_obs(obs, slice(first, n)), self.obs_spec)
            _write_obs_slot(self.next_obs, sl1, slice_obs(next_obs, slice(0, first)), self.obs_spec)
            _write_obs_slot(self.next_obs, sl2, slice_obs(next_obs, slice(first, n)), self.obs_spec)
            self.actions[sl1] = act_arr[:first]
            self.actions[sl2] = act_arr[first:]
            self.rewards[sl1] = rew_arr[:first]
            self.rewards[sl2] = rew_arr[first:]
            self.terminateds[sl1] = term_arr[:first]
            self.terminateds[sl2] = term_arr[first:]
            if self.costs is not None and cost_arr is not None:
                self.costs[sl1] = cost_arr[:first]
                self.costs[sl2] = cost_arr[first:]
        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)
        return n

    def _sample_indices(self, batch_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")
        return (
            rng.integers(0, self.size, size=batch_size)
            if rng is not None
            else np.random.randint(0, self.size, size=batch_size)
        )

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        idx = self._sample_indices(batch_size, rng)
        b_obs = obs_to_tensor(slice_obs(self.obs, idx), self.obs_spec, self.device, add_batch_dim=False)
        b_next = obs_to_tensor(slice_obs(self.next_obs, idx), self.obs_spec, self.device, add_batch_dim=False)
        return (
            b_obs,
            torch.as_tensor(
                self.actions[idx],
                device=self.device,
                dtype=torch.long if self.discrete_action else torch.float32,
            ),
            torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            b_next,
            torch.as_tensor(self.terminateds[idx], device=self.device, dtype=torch.float32),
        )

    def sample_with_costs(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor]:
        """Sample transition batch alongside constraint costs."""
        idx = self._sample_indices(batch_size, rng)
        b_obs = obs_to_tensor(slice_obs(self.obs, idx), self.obs_spec, self.device, add_batch_dim=False)
        b_next = obs_to_tensor(slice_obs(self.next_obs, idx), self.obs_spec, self.device, add_batch_dim=False)
        c_arr = self.costs[idx] if self.costs is not None else np.zeros((len(idx),), dtype=np.float32)
        return (
            b_obs,
            torch.as_tensor(
                self.actions[idx],
                device=self.device,
                dtype=torch.long if self.discrete_action else torch.float32,
            ),
            torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            b_next,
            torch.as_tensor(self.terminateds[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(c_arr, device=self.device, dtype=torch.float32),
        )

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "obs": {k: v[: self.size] for k, v in self.obs.items()} if isinstance(self.obs, dict) else self.obs[: self.size],
            "next_obs": {k: v[: self.size] for k, v in self.next_obs.items()}
            if isinstance(self.next_obs, dict)
            else self.next_obs[: self.size],
            "actions": self.actions[: self.size],
            "rewards": self.rewards[: self.size],
            "terminateds": self.terminateds[: self.size],
            "ptr": self.ptr,
            "size": self.size,
        }
        if self.costs is not None:
            out["costs"] = self.costs[: self.size]
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        sz = min(int(state["size"]), self.capacity)
        if isinstance(self.obs, dict):
            for k in self.obs:
                self.obs[k][:sz] = state["obs"][k][:sz]
                self.next_obs[k][:sz] = state["next_obs"][k][:sz]
        else:
            self.obs[:sz] = state["obs"][:sz]
            self.next_obs[:sz] = state["next_obs"][:sz]
        self.actions[:sz] = state["actions"][:sz]
        self.rewards[:sz] = state["rewards"][:sz]
        self.terminateds[:sz] = state["terminateds"][:sz]
        if self.costs is not None and "costs" in state:
            self.costs[:sz] = state["costs"][:sz]
        self.size = sz
        self.ptr = int(state["ptr"]) % self.capacity


class DualDemoReplayBuffer:
    """Symmetric 50/50 Demo/Intervention + Online replay buffer (RLPD / HIL-SERL style)."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int | ObsSpec,
        action_dim: int,
        discrete_action: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.demo_buffer = ReplayBuffer(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            discrete_action=discrete_action,
            device=device,
        )
        self.online_buffer = ReplayBuffer(
            capacity=capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            discrete_action=discrete_action,
            device=device,
        )

    def add_demo(
        self,
        obs: Any,
        action: np.ndarray | int,
        reward: float,
        next_obs: Any,
        terminated: bool,
    ) -> None:
        self.demo_buffer.add(obs, action, reward, next_obs, terminated)

    def add_online(
        self,
        obs: Any,
        action: np.ndarray | int,
        reward: float,
        next_obs: Any,
        terminated: bool,
    ) -> None:
        self.online_buffer.add(obs, action, reward, next_obs, terminated)

    def add_online_batch(
        self,
        obs: Any,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: Any,
        terminateds: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> int:
        return self.online_buffer.add_batch(obs, actions, rewards, next_obs, terminateds, valid_mask=valid_mask)

    def sample_symmetric(
        self,
        batch_size: int,
        demo_fraction: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """Mix offline demo/human-intervention transitions with online rollout transitions."""
        if self.demo_buffer.size == 0 and self.online_buffer.size == 0:
            raise RuntimeError("Both demo and online buffers are empty.")
        if self.demo_buffer.size == 0:
            return self.online_buffer.sample(batch_size, rng=rng)
        if self.online_buffer.size == 0:
            return self.demo_buffer.sample(batch_size, rng=rng)

        n_demo = max(1, min(batch_size - 1, round(batch_size * demo_fraction)))
        n_online = batch_size - n_demo
        d_batch = self.demo_buffer.sample(n_demo, rng=rng)
        o_batch = self.online_buffer.sample(n_online, rng=rng)
        return (
            concat_obs_tensors(d_batch[0], o_batch[0]),
            torch.cat([d_batch[1], o_batch[1]], dim=0),
            torch.cat([d_batch[2], o_batch[2]], dim=0),
            concat_obs_tensors(d_batch[3], o_batch[3]),
            torch.cat([d_batch[4], o_batch[4]], dim=0),
        )
