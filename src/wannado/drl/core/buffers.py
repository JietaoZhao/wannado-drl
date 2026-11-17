"""Lightweight pre-allocated RolloutBuffer, ReplayBuffer, and DualDemoReplayBuffer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class RolloutBuffer:
    """On-policy trajectory buffer storing transitions, optional action masks, and GAE."""

    def __init__(
        self,
        num_steps: int,
        obs_dim: int,
        action_shape: tuple[int, ...],
        action_dtype: np.dtype[Any] | None = None,
        mask_dim: int | None = None,
    ) -> None:
        self.num_steps = int(num_steps)
        resolved_action_dtype = np.dtype(np.float32) if action_dtype is None else np.dtype(action_dtype)
        self.obs = np.zeros((num_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, *action_shape), dtype=resolved_action_dtype)
        self.logprobs = np.zeros((num_steps,), dtype=np.float32)
        self.rewards = np.zeros((num_steps,), dtype=np.float32)
        self.terminateds = np.zeros((num_steps,), dtype=np.float32)
        self.dones = np.zeros((num_steps,), dtype=np.float32)
        self.values = np.zeros((num_steps,), dtype=np.float32)
        self.advantages = np.zeros((num_steps,), dtype=np.float32)
        self.returns = np.zeros((num_steps,), dtype=np.float32)
        self.action_masks = (
            np.ones((num_steps, mask_dim), dtype=np.bool_) if mask_dim is not None else None
        )
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
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

    def compute_returns_and_advantage(
        self,
        next_value: float,
        next_terminated: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE while preventing cross-episode leakage across terminated/truncated boundaries.

        Callers handling TimeLimit truncations (``trunc and not term``) should pre-bootstrap
        ``reward += gamma * V(terminal_next_obs)`` and pass ``done=True`` so GAE recursion
        terminates cleanly at the episode boundary without double-counting or leaking next-episode values.
        """
        last_gae = 0.0
        n = self.ptr
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
                nonterminal = 1.0 - float(bool(self.dones[t]) or bool(next_terminated))
            else:
                next_val = float(self.values[t + 1])
                nonterminal = 1.0 - float(self.dones[t])
            delta = float(self.rewards[t]) + gamma * next_val * nonterminal - float(self.values[t])
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def reset(self) -> None:
        self.ptr = 0
        self.dones.fill(0.0)


class ReplayBuffer:
    """Fast pre-allocated circular replay buffer for off-policy value/actor-critic RL."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        discrete_action: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.discrete_action = discrete_action
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        act_dtype = np.int64 if discrete_action else np.float32
        act_shape = (self.capacity,) if discrete_action else (self.capacity, action_dim)
        self.actions = np.zeros(act_shape, dtype=act_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.terminateds = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.terminateds[self.ptr] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer.")
        idx = (
            rng.integers(0, self.size, size=batch_size)
            if rng is not None
            else np.random.randint(0, self.size, size=batch_size)
        )
        return (
            torch.as_tensor(self.obs[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(
                self.actions[idx],
                device=self.device,
                dtype=torch.long if self.discrete_action else torch.float32,
            ),
            torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.next_obs[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.terminateds[idx], device=self.device, dtype=torch.float32),
        )


class DualDemoReplayBuffer:
    """Symmetric 50/50 Demo/Intervention + Online replay buffer (RLPD / HIL-SERL style)."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
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
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.demo_buffer.add(obs, action, reward, next_obs, terminated)

    def add_online(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.online_buffer.add(obs, action, reward, next_obs, terminated)

    def sample_symmetric(
        self,
        batch_size: int,
        demo_fraction: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return tuple(torch.cat([d, o], dim=0) for d, o in zip(d_batch, o_batch))  # type: ignore[return-value]
