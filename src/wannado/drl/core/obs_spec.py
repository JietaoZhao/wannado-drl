"""Structured multimodal Observation Specs (`vector`, `image`, `entities`, `dict`) and batch formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from gymnasium import spaces

ObsKind = Literal["vector", "image", "entities", "dict"]
LeafKind = Literal["vector", "image", "entities"]

_CNN_EXTRACTOR_ALIASES = {"nature_cnn", "impala_cnn", "impala", "small_cnn", "cnn"}
_ENTITY_KEY_HINTS = ("entit", "asset", "user", "agent", "node", "token", "target", "ue")


def _is_non_numeric_space(space: Any) -> bool:
    if isinstance(space, spaces.Text):
        return True
    cls_name = type(space).__name__
    return cls_name in {"Text", "MissionSpace", "Graph", "Sequence"}


def _is_dict_space(space: Any) -> bool:
    return isinstance(space, spaces.Dict) or (hasattr(space, "spaces") and isinstance(getattr(space, "spaces", None), dict))


def _is_discrete_space(space: Any) -> bool:
    return isinstance(space, spaces.Discrete) or (hasattr(space, "n") and not hasattr(space, "nvec") and not hasattr(space, "low"))


def _is_multidiscrete_space(space: Any) -> bool:
    return isinstance(space, spaces.MultiDiscrete) or hasattr(space, "nvec")


def _is_image_3d_shape(shape: tuple[int, ...], *, force_cnn: bool = False) -> tuple[bool, bool, tuple[int, int, int]]:
    """Detect whether a 3D shape is ``(C, H, W)`` or ``(H, W, C)`` and return ``(is_img, channel_first, chw)``."""
    if len(shape) != 3:
        return False, True, (0, 0, 0)
    d0, d1, d2 = (int(shape[0]), int(shape[1]), int(shape[2]))
    min_spatial = 4 if force_cnn else 8
    if d0 in (1, 3, 4) and d1 >= min_spatial and d2 >= min_spatial:
        return True, True, (d0, d1, d2)
    if 1 <= d2 <= 32 and d0 >= min_spatial and d1 >= min_spatial:
        return True, False, (d2, d0, d1)
    if force_cnn and d0 >= 1 and d1 >= 4 and d2 >= 4:
        return True, True, (d0, d1, d2)
    return False, True, (0, 0, 0)


def _space_bounds_and_dtype(space: Any, sample: Any | None = None) -> tuple[tuple[int, ...], Any, float | None, float | None]:
    shape_attr = getattr(space, "shape", None)
    shape = tuple(int(s) for s in shape_attr) if isinstance(shape_attr, tuple) else ()
    dtype = getattr(space, "dtype", None)
    low_val: float | None = None
    high_val: float | None = None
    low_arr = getattr(space, "low", getattr(space, "minimum", None))
    high_arr = getattr(space, "high", getattr(space, "maximum", None))
    try:
        if low_arr is not None:
            l_np = np.asarray(low_arr, dtype=np.float64)
            if l_np.size and np.all(np.isfinite(l_np)):
                low_val = float(np.min(l_np))
        if high_arr is not None:
            h_np = np.asarray(high_arr, dtype=np.float64)
            if h_np.size and np.all(np.isfinite(h_np)):
                high_val = float(np.max(h_np))
    except Exception:  # noqa: BLE001, S110 — bounds metadata optional on custom spaces
        pass
    if not shape and sample is not None:
        arr = np.asarray(sample)
        shape = tuple(int(s) for s in arr.shape)
        if dtype is None:
            dtype = arr.dtype
    return shape, dtype, low_val, high_val


def _infer_leaf_spec(
    key: str,
    space: Any,
    sample: Any | None = None,
    *,
    extractor_type: str = "auto",
    mask_key: str | None = None,
) -> LeafObsSpec:
    if _is_discrete_space(space):
        n = int(space.n)
        return LeafObsSpec(key=key, kind="vector", shape=(n,), raw_shape=(), dtype="float32", flat_dim=n)
    if _is_multidiscrete_space(space):
        nvec = tuple(int(v) for v in np.asarray(space.nvec, dtype=np.int64).reshape(-1))
        n_sum = int(sum(nvec))
        return LeafObsSpec(
            key=key, kind="vector", shape=(n_sum,), raw_shape=(len(nvec),), dtype="float32", flat_dim=n_sum, nvec=nvec
        )

    shape, dtype, low_val, high_val = _space_bounds_and_dtype(space, sample=sample)
    force_cnn = extractor_type in _CNN_EXTRACTOR_ALIASES
    is_uint8 = dtype is not None and np.issubdtype(np.dtype(dtype), np.uint8)
    is_pixel_range = is_uint8 or (low_val is not None and high_val is not None and low_val >= 0.0 and high_val >= 200.0)

    if len(shape) == 3:
        is_img, ch_first, chw = _is_image_3d_shape(shape, force_cnn=force_cnn)
        if is_img:
            return LeafObsSpec(
                key=key,
                kind="image",
                shape=chw,
                raw_shape=shape,
                dtype="uint8" if is_uint8 else "float32",
                channel_first=ch_first,
                normalize_pixel=is_pixel_range,
                flat_dim=int(np.prod(chw)),
            )

    if len(shape) == 2:
        h, w = shape
        if (is_pixel_range or force_cnn) and h >= 8 and w >= 8:
            chw = (1, h, w)
            return LeafObsSpec(
                key=key,
                kind="image",
                shape=chw,
                raw_shape=shape,
                dtype="uint8" if is_uint8 else "float32",
                channel_first=False,
                normalize_pixel=is_pixel_range,
                flat_dim=int(h * w),
            )
        is_entity_key = any(hint in key.lower() for hint in _ENTITY_KEY_HINTS)
        if (extractor_type in ("attention", "set_attention") or mask_key is not None or (extractor_type in ("auto", "dict") and is_entity_key)) and h >= 1 and w >= 1:
            return LeafObsSpec(
                key=key,
                kind="entities",
                shape=(h, w),
                raw_shape=shape,
                dtype="float32",
                flat_dim=int(h * w),
                mask_key=mask_key,
            )

    flat = int(np.prod(shape)) if shape else 1
    return LeafObsSpec(key=key, kind="vector", shape=(flat,), raw_shape=shape, dtype="float32", flat_dim=flat)


@dataclass(frozen=True)
class LeafObsSpec:
    """Specification for a single observation tensor (standalone or Dict entry)."""

    key: str
    kind: LeafKind
    shape: tuple[int, ...]
    raw_shape: tuple[int, ...]
    dtype: Literal["float32", "uint8"] = "float32"
    channel_first: bool = True
    normalize_pixel: bool = False
    flat_dim: int = 1
    mask_key: str | None = None
    nvec: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "shape": list(self.shape),
            "raw_shape": list(self.raw_shape),
            "dtype": self.dtype,
            "channel_first": self.channel_first,
            "normalize_pixel": self.normalize_pixel,
            "flat_dim": self.flat_dim,
            "mask_key": self.mask_key,
            "nvec": list(self.nvec) if self.nvec is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeafObsSpec:
        raw_nvec = data.get("nvec")
        return cls(
            key=str(data["key"]),
            kind=data["kind"],
            shape=tuple(int(x) for x in data["shape"]),
            raw_shape=tuple(int(x) for x in data["raw_shape"]),
            dtype=data.get("dtype", "float32"),
            channel_first=bool(data.get("channel_first", True)),
            normalize_pixel=bool(data.get("normalize_pixel", False)),
            flat_dim=int(data["flat_dim"]),
            mask_key=data.get("mask_key"),
            nvec=tuple(int(x) for x in raw_nvec) if raw_nvec is not None else None,
        )


@dataclass(frozen=True)
class ObsSpec:
    """Canonical observation schema supporting 1D vectors, CHW images, 2D entity sets, and Dicts."""

    kind: ObsKind
    flat_dim: int
    shape: tuple[int, ...]
    dtype: Literal["float32", "uint8"]
    normalize_pixel: bool
    channel_first: bool
    leaves: dict[str, LeafObsSpec]
    flatten_dict_leaves: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "flat_dim": self.flat_dim,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "normalize_pixel": self.normalize_pixel,
            "channel_first": self.channel_first,
            "leaves": {k: v.to_dict() for k, v in self.leaves.items()},
            "flatten_dict_leaves": self.flatten_dict_leaves,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObsSpec:
        return cls(
            kind=data["kind"],
            flat_dim=int(data["flat_dim"]),
            shape=tuple(int(x) for x in data["shape"]),
            dtype=data.get("dtype", "float32"),
            normalize_pixel=bool(data.get("normalize_pixel", False)),
            channel_first=bool(data.get("channel_first", True)),
            leaves={k: LeafObsSpec.from_dict(v) for k, v in data["leaves"].items()},
            flatten_dict_leaves=bool(data.get("flatten_dict_leaves", False)),
        )


def vector_obs_spec(obs_dim: int) -> ObsSpec:
    """Create a canonical 1D float32 vector ``ObsSpec``."""
    leaf = LeafObsSpec(key="obs", kind="vector", shape=(int(obs_dim),), raw_shape=(int(obs_dim),), flat_dim=int(obs_dim))
    return ObsSpec(
        kind="vector",
        flat_dim=int(obs_dim),
        shape=(int(obs_dim),),
        dtype="float32",
        normalize_pixel=False,
        channel_first=True,
        leaves={"obs": leaf},
    )


def _find_entity_mask_key(entity_key: str, all_keys: set[str], sub_spaces: dict[str, Any]) -> str | None:
    candidates = [f"{entity_key}_mask", "entity_mask", "entities_mask", "mask"]
    if entity_key.endswith("s"):
        candidates.append(f"{entity_key[:-1]}_mask")
    for cand in candidates:
        if cand in all_keys:
            sp = sub_spaces[cand]
            sh = getattr(sp, "shape", None)
            if isinstance(sh, tuple) and len(sh) == 1:
                return cand
    return None


def infer_obs_spec(
    raw_obs_space: Any,
    sample_obs: Any | None = None,
    *,
    extractor_type: str = "auto",
) -> ObsSpec:
    """Infer structured ``ObsSpec`` from a Gymnasium/Gym/dm_control observation space."""
    if _is_dict_space(raw_obs_space):
        sub_spaces: dict[str, Any] = dict(raw_obs_space.spaces)
        all_keys = set(sub_spaces.keys())
        numeric_keys = [k for k in sorted(all_keys) if not _is_non_numeric_space(sub_spaces[k])]

        # First locate paired entity mask keys so entity leaves link to them
        paired_masks: dict[str, str] = {}
        for k in numeric_keys:
            mk = _find_entity_mask_key(k, set(numeric_keys), sub_spaces)
            sh = getattr(sub_spaces[k], "shape", None)
            if mk is not None and isinstance(sh, tuple) and len(sh) == 2:
                paired_masks[k] = mk

        leaves: dict[str, LeafObsSpec] = {}
        for k in numeric_keys:
            sub_sample = sample_obs.get(k) if isinstance(sample_obs, dict) else None
            leaves[k] = _infer_leaf_spec(
                k,
                sub_spaces[k],
                sample=sub_sample,
                extractor_type=extractor_type,
                mask_key=paired_masks.get(k),
            )

        total_flat = sum(lf.flat_dim for lf in leaves.values())
        all_vectors = all(lf.kind == "vector" for lf in leaves.values())
        if all_vectors and extractor_type in ("auto", "flatten") and total_flat <= 64:
            # Preserve exact v1.0.5 1D vector behavior for tiny numeric/legacy Dict spaces
            return ObsSpec(
                kind="vector",
                flat_dim=total_flat,
                shape=(total_flat,),
                dtype="float32",
                normalize_pixel=False,
                channel_first=True,
                leaves=leaves,
                flatten_dict_leaves=True,
            )
        if extractor_type == "flatten":
            return ObsSpec(
                kind="vector",
                flat_dim=total_flat,
                shape=(total_flat,),
                dtype="float32",
                normalize_pixel=False,
                channel_first=True,
                leaves=leaves,
                flatten_dict_leaves=True,
            )
        return ObsSpec(
            kind="dict",
            flat_dim=total_flat,
            shape=(),
            dtype="float32",
            normalize_pixel=any(lf.normalize_pixel for lf in leaves.values()),
            channel_first=True,
            leaves=leaves,
            flatten_dict_leaves=False,
        )

    if extractor_type == "flatten" and raw_obs_space is not None:
        flat = int(space_flatdim(raw_obs_space, sample_obs))
        return vector_obs_spec(flat)

    if raw_obs_space is None and isinstance(sample_obs, dict):
        # Fallback inference from dict sample without declared space
        leaves_from_sample: dict[str, LeafObsSpec] = {}
        for k in sorted(sample_obs.keys()):
            val = sample_obs[k]
            if isinstance(val, (str, bytes)):
                continue
            arr = np.asarray(val)
            leaves_from_sample[k] = _infer_leaf_spec(k, None, sample=arr, extractor_type=extractor_type)
        total_flat = sum(lf.flat_dim for lf in leaves_from_sample.values())
        return ObsSpec(
            kind="dict",
            flat_dim=total_flat,
            shape=(),
            dtype="float32",
            normalize_pixel=any(lf.normalize_pixel for lf in leaves_from_sample.values()),
            channel_first=True,
            leaves=leaves_from_sample,
        )

    leaf = _infer_leaf_spec("obs", raw_obs_space, sample=sample_obs, extractor_type=extractor_type)
    return ObsSpec(
        kind=leaf.kind,
        flat_dim=leaf.flat_dim,
        shape=leaf.shape,
        dtype=leaf.dtype,
        normalize_pixel=leaf.normalize_pixel,
        channel_first=leaf.channel_first,
        leaves={"obs": leaf},
    )


def space_flatdim(obs_space: Any, sample_obs: Any | None = None) -> int:
    """Compute numeric flat observation dimension cleanly across legacy & Gymnasium spaces."""
    if _is_dict_space(obs_space):
        return sum(
            space_flatdim(sub_sp, sample_obs[k] if isinstance(sample_obs, dict) and k in sample_obs else None)
            for k, sub_sp in sorted(obs_space.spaces.items())
            if not _is_non_numeric_space(sub_sp)
        )
    if _is_discrete_space(obs_space):
        return int(obs_space.n)
    if _is_multidiscrete_space(obs_space):
        return int(np.sum(np.asarray(obs_space.nvec, dtype=np.int64)))
    if obs_space is not None:
        shape = getattr(obs_space, "shape", None)
        if isinstance(shape, tuple):
            return int(np.prod(shape))
        try:
            return int(spaces.flatdim(obs_space))
        except Exception:  # noqa: BLE001, S110 — inspect sample when flatdim fails
            pass
    if sample_obs is not None:
        if isinstance(sample_obs, dict):
            return sum(int(np.asarray(v).size) for v in sample_obs.values() if not isinstance(v, (str, bytes)))
        return int(np.asarray(sample_obs).size)
    raise ValueError("Cannot infer flat observation dimension without space or sample.")


def _format_leaf_array(raw_val: Any, leaf: LeafObsSpec, *, batched: bool) -> np.ndarray:
    """Format a single or batched leaf array into canonical ``LeafObsSpec`` shape/dtype."""
    if leaf.kind == "vector":
        if leaf.nvec is not None:
            if batched:
                arr_i = np.asarray(raw_val, dtype=np.int64).reshape(-1, len(leaf.nvec))
                out_md = np.zeros((arr_i.shape[0], leaf.shape[0]), dtype=np.float32)
                off = 0
                for j, dim_j in enumerate(leaf.nvec):
                    idx_j = np.clip(arr_i[:, j], 0, dim_j - 1)
                    out_md[np.arange(arr_i.shape[0]), off + idx_j] = 1.0
                    off += dim_j
                return out_md
            vals = np.asarray(raw_val, dtype=np.int64).reshape(-1)
            out_s_md = np.zeros(leaf.shape, dtype=np.float32)
            off = 0
            for j, dim_j in enumerate(leaf.nvec):
                v_j = int(vals[j]) if j < len(vals) else 0
                if 0 <= v_j < dim_j:
                    out_s_md[off + v_j] = 1.0
                off += dim_j
            return out_s_md
        if batched:
            arr = np.asarray(raw_val, dtype=np.float32)
            b = arr.shape[0]
            # Handle 1D Discrete scalar array (B,) -> one-hot (B, N)
            if leaf.raw_shape == () and arr.ndim == 1 and leaf.shape[0] > 1:
                out = np.zeros((b, leaf.shape[0]), dtype=np.float32)
                idx = np.clip(arr.astype(np.int64), 0, leaf.shape[0] - 1)
                out[np.arange(b), idx] = 1.0
                return out
            return arr.reshape(b, *leaf.shape).astype(np.float32, copy=False)
        if leaf.raw_shape == () and leaf.shape[0] > 1 and np.ndim(raw_val) == 0:
            out_s = np.zeros(leaf.shape, dtype=np.float32)
            idx_s = int(raw_val)
            if 0 <= idx_s < leaf.shape[0]:
                out_s[idx_s] = 1.0
            return out_s
        return np.asarray(raw_val, dtype=np.float32).reshape(leaf.shape)

    np_dtype = np.uint8 if leaf.dtype == "uint8" else np.float32
    arr = np.asarray(raw_val, dtype=np_dtype)
    if leaf.kind == "image":
        if batched:
            if len(leaf.raw_shape) == 2:
                return arr[:, None, :, :]
            if not leaf.channel_first:
                return np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))
            return np.ascontiguousarray(arr)
        if len(leaf.raw_shape) == 2:
            return np.ascontiguousarray(arr[None, :, :])
        if not leaf.channel_first:
            return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
        return np.ascontiguousarray(arr)

    # Entities: (N, F) or (B, N, F)
    if batched:
        return arr.reshape(arr.shape[0], *leaf.shape).astype(np.float32, copy=False)
    return arr.reshape(leaf.shape).astype(np.float32, copy=False)


def _format_dict_flat(raw_obs: Any, spec: ObsSpec, *, batched: bool, batch_size: int = 1) -> np.ndarray:
    parts: list[np.ndarray] = []
    for k, leaf in spec.leaves.items():
        val = raw_obs[k] if isinstance(raw_obs, dict) and k in raw_obs else np.zeros(
            (batch_size, *leaf.raw_shape) if batched else leaf.raw_shape, dtype=np.float32
        )
        formatted = _format_leaf_array(val, leaf, batched=batched)
        if batched:
            parts.append(formatted.reshape(formatted.shape[0], -1).astype(np.float32))
        else:
            parts.append(formatted.reshape(-1).astype(np.float32))
    axis = 1 if batched else 0
    return np.concatenate(parts, axis=axis).astype(np.float32) if parts else np.zeros((batch_size, 0) if batched else (0,), dtype=np.float32)


def format_obs_single(raw_obs: Any, spec: ObsSpec) -> Any:
    """Format one raw observation according to ``ObsSpec`` (``ndarray`` or ``dict[str, ndarray]``)."""
    if spec.flatten_dict_leaves:
        return _format_dict_flat(raw_obs, spec, batched=False)
    if spec.kind == "dict":
        return {
            k: _format_leaf_array(raw_obs[k], leaf, batched=False)
            for k, leaf in spec.leaves.items()
            if isinstance(raw_obs, dict) and k in raw_obs
        }
    leaf = spec.leaves["obs"]
    return _format_leaf_array(raw_obs, leaf, batched=False)


def format_obs_batch(raw_obs: Any, spec: ObsSpec, num_envs: int) -> Any:
    """Format batched observations from ``num_envs`` environments according to ``ObsSpec``."""
    if spec.kind == "dict" and isinstance(raw_obs, (list, tuple)):
        raw_obs = {
            k: np.stack([item[k] for item in raw_obs], axis=0)
            for k in spec.leaves
            if raw_obs and isinstance(raw_obs[0], dict) and k in raw_obs[0]
        }
    if spec.flatten_dict_leaves:
        return _format_dict_flat(raw_obs, spec, batched=True, batch_size=num_envs)
    if spec.kind == "dict":
        return {
            k: _format_leaf_array(raw_obs[k], leaf, batched=True)
            for k, leaf in spec.leaves.items()
            if isinstance(raw_obs, dict) and k in raw_obs
        }
    leaf = spec.leaves["obs"]
    return _format_leaf_array(raw_obs, leaf, batched=True)


def _tensorize_leaf(arr: np.ndarray, normalize_pixel: bool, device: torch.device, add_batch: bool) -> torch.Tensor:
    if add_batch:
        arr = arr[None, ...]
    if arr.dtype == np.uint8:
        t = torch.as_tensor(arr, device=device).to(torch.float32)
        return t * (1.0 / 255.0) if normalize_pixel else t
    t = torch.as_tensor(arr, device=device, dtype=torch.float32)
    return t * (1.0 / 255.0) if normalize_pixel else t


def obs_to_tensor(
    obs: Any,
    spec: ObsSpec | LeafObsSpec,
    device: torch.device,
    *,
    add_batch_dim: bool = False,
) -> Any:
    """Convert formatted NumPy observation (single or batched, array or dict) into GPU/CPU float32 tensor(s)."""
    if isinstance(spec, ObsSpec) and spec.kind == "dict" and isinstance(obs, dict):
        return {
            k: _tensorize_leaf(
                np.asarray(obs[k]),
                leaf.normalize_pixel,
                device,
                add_batch=add_batch_dim,
            )
            for k, leaf in spec.leaves.items()
            if k in obs
        }
    arr = np.asarray(obs)
    return _tensorize_leaf(arr, spec.normalize_pixel, device, add_batch=add_batch_dim)


def slice_obs(obs: Any, idx: Any) -> Any:
    """Index or slice a NumPy array or dict of NumPy/Torch arrays uniformly."""
    if isinstance(obs, dict):
        return {k: v[idx] for k, v in obs.items()}
    return obs[idx]


def concat_obs_tensors(a: Any, b: Any) -> Any:
    """Concatenate two batches of tensors (Tensor or dict[str, Tensor]) along dim=0."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: torch.cat([a[k], b[k]], dim=0) for k in a}
    return torch.cat([a, b], dim=0)
