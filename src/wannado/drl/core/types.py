"""Shared metadata schemas, training records, and evaluation contracts for wannado.drl."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

FamilyType = Literal[
    "tabular_classic",
    "value_based",
    "policy_gradient",
    "actor_critic_off",
    "safe_constrained",
    "offline_rl",
    "multi_agent",
    "llm_and_discovered",
]
ParadigmType = Literal[
    "tabular_on_policy",
    "tabular_off_policy",
    "on_policy",
    "off_policy",
    "offline",
    "multi_agent_ctde",
    "group_relative",
    "meta_discovered",
]
TierType = Literal["S", "A", "B", "C"]


@dataclass(frozen=True)
class AlgoCitation:
    """Verified academic provenance record for a reinforcement learning algorithm."""

    key: str
    title: str
    authors: str
    venue: str
    year: int
    url: str
    bibtex: str
    ieee: str
    apa: str


@dataclass(frozen=True)
class AlgoMeta:
    """Single-source-of-truth metadata card embedded in every CleanRL-style algorithm file."""

    id: str
    name: str
    family: FamilyType
    paradigm: ParadigmType
    tier: TierType
    on_policy: bool
    sample_efficiency: Literal["low", "medium", "high", "very_high"]
    action_spaces: tuple[str, ...]
    env_modalities: tuple[str, ...]
    supports_action_mask: bool
    supports_recurrent: bool
    parent_id: str | None
    diff_from_parent: str
    summary_plain: str
    when_to_use: str
    when_not_to_use: str
    hyperparam_intuition: dict[str, str]
    extension_hooks: tuple[str, ...]
    citation: AlgoCitation
    module_path: str = ""
    default_optimizations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata card into a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "paradigm": self.paradigm,
            "tier": self.tier,
            "on_policy": self.on_policy,
            "sample_efficiency": self.sample_efficiency,
            "action_spaces": list(self.action_spaces),
            "env_modalities": list(self.env_modalities),
            "supports_action_mask": self.supports_action_mask,
            "supports_recurrent": self.supports_recurrent,
            "parent_id": self.parent_id,
            "diff_from_parent": self.diff_from_parent,
            "summary_plain": self.summary_plain,
            "when_to_use": self.when_to_use,
            "when_not_to_use": self.when_not_to_use,
            "hyperparam_intuition": dict(self.hyperparam_intuition),
            "extension_hooks": list(self.extension_hooks),
            "module_path": self.module_path,
            "default_optimizations": list(self.default_optimizations),
            "citation": {
                "key": self.citation.key,
                "title": self.citation.title,
                "authors": self.citation.authors,
                "venue": self.citation.venue,
                "year": self.citation.year,
                "url": self.citation.url,
                "bibtex": self.citation.bibtex,
                "ieee": self.citation.ieee,
                "apa": self.citation.apa,
            },
        }


@dataclass
class EpisodeEvalStats:
    """Deterministic evaluation metrics across episodes with sub-reward & KPI tracking."""

    returns: list[float]
    lengths: list[int]
    mean_return: float
    std_return: float
    mean_length: float
    reward_components_mean: dict[str, float] = field(default_factory=dict)
    task_metrics_mean: dict[str, float] = field(default_factory=dict)
    terminated_fraction: float = 0.0
    truncated_fraction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation stats."""
        return asdict(self)


@dataclass
class TrainResult:
    """Standardized return container produced by all wannado.drl single-file algorithms."""

    algo_id: str
    seed: int
    total_timesteps: int
    elapsed_seconds: float
    policy_fn: Callable[..., Any]
    model: Any
    config: dict[str, Any]
    learning_curve: list[dict[str, float]] = field(default_factory=list)
    train_diagnostics: dict[str, Any] = field(default_factory=dict)
    eval_stats: EpisodeEvalStats | None = None
    obs_rms_state: dict[str, Any] | None = None
    obs_rms_frozen_on_eval: bool = True
    optimizations_used: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    eval_history: list[dict[str, float]] = field(default_factory=list)
    best_model_path: str | None = None
    last_checkpoint_path: str | None = None
    policy_bundle: Any = None

    def save_policy(self, path: str | Any) -> Any:
        """Persist a standalone self-contained PolicyBundle checkpoint for deployment or auditing."""
        if self.policy_bundle is None:
            raise RuntimeError("This TrainResult does not attach a serialized PolicyBundle.")
        return self.policy_bundle.save(path)

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize JSON-safe summary (excluding non-serializable PyTorch modules/callables)."""
        return {
            "algo_id": self.algo_id,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "elapsed_seconds": self.elapsed_seconds,
            "config": dict(self.config),
            "learning_curve": list(self.learning_curve),
            "train_diagnostics": dict(self.train_diagnostics),
            "eval_stats": self.eval_stats.to_dict() if self.eval_stats is not None else None,
            "obs_rms_frozen_on_eval": self.obs_rms_frozen_on_eval,
            "optimizations_used": list(self.optimizations_used),
            "eval_history": list(self.eval_history),
            "best_model_path": self.best_model_path,
            "last_checkpoint_path": self.last_checkpoint_path,
            "extra": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.extra.items()
            },
        }
