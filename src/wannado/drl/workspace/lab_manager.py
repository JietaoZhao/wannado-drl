"""Non-intrusive external plug-in workspace (`drl_lab/`) and single-file CleanRL algorithm ejection.

Connects any user ML/simulation/RL project via ONE clean boundary (`drl_lab/interface.py`
returning `EnvInterface`) without modifying the user's core project or simulator files.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wannado.drl.algos import ALGO_ID_TO_RUNNER
from wannado.drl.catalog.build_catalog import DRL_ROOT, load_catalog
from wannado.drl.mdp.intake import build_mdp_archetype
from wannado.drl.mdp.spec import MDPSpec


@dataclass
class EnvInterface:
    """Single clean plug-in boundary connecting an external project into wannado-drl."""

    env_or_factory: Any
    mdp_spec: MDPSpec | None = None
    env_library: str = "auto"
    task_name: str = "MyDRLProject"
    domain: str = "custom_sim"
    heuristic_fn: Callable[[np.ndarray, dict[str, Any]], Any] | None = None
    action_mask_fn: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None
    demo_dataset: Any = None
    intervention_fn: Callable[..., tuple[bool, np.ndarray]] | None = None
    extra_metadata: dict[str, Any] | None = None


def _resolve_algo_source_path(algo_id: str) -> tuple[dict[str, Any], Path]:
    cards = {c["id"]: c for c in load_catalog()}
    card = cards.get(algo_id)
    mod_path = card["module_path"] if card is not None else None
    if mod_path is None and algo_id in ALGO_ID_TO_RUNNER:
        mod_path, _ = ALGO_ID_TO_RUNNER[algo_id]
        mod = importlib.import_module(mod_path)
        meta = getattr(mod, "META", None)
        if meta is not None:
            card = meta.to_dict()
    if card is None or mod_path is None:
        sample = ", ".join(sorted(cards.keys())[:10])
        raise KeyError(f"Unknown algorithm '{algo_id}'. Sample valid IDs: {sample} ...")

    if mod_path.startswith("wannado.drl."):
        rel_parts = mod_path.removeprefix("wannado.drl.").split(".")
        src_path = (DRL_ROOT / Path(*rel_parts)).with_suffix(".py")
        if src_path.exists():
            return card, src_path
    mod = importlib.import_module(mod_path)
    if mod.__file__ is None:
        raise FileNotFoundError(f"Cannot locate source file for module '{mod_path}'.")
    return card, Path(mod.__file__).resolve()


def eject_algo(
    algo_id: str,
    dest_dir: str | Path = "./drl_lab/algos",
    *,
    custom_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy ONE CleanRL-style single-file algorithm into `drl_lab/algos/` for custom research edits."""
    _, src_path = _resolve_algo_source_path(algo_id)
    target_dir = Path(dest_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = custom_name if custom_name is not None else src_path.name
    if not filename.endswith(".py"):
        filename = f"{filename}.py"
    dest_path = target_dir / filename

    if dest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing customized file at {dest_path}. Pass overwrite=True to replace."
        )

    banner = (
        f"# [wannado-drl Ejected CleanRL Single-File Algorithm: id='{algo_id}']\n"
        f"# Edit ONLY this standalone file to prototype new paper ideas without touching\n"
        f"# your project simulator or other algorithms. Update `META` if adding citations.\n\n"
    )
    dest_path.write_text(banner + src_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path


def _interface_template(mode: str, spec: MDPSpec) -> str:
    header = f'''"""Non-intrusive external DRL plug-in interface for `{spec.problem_name}`.

Generated from Stage 0 `mdp_spec.json`. Wire your project's simulator, legacy Gym/Gymnasium
environment, PettingZoo multi-agent env, or verifiable sequence task inside `load_env_interface()`
without modifying your original codebase.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wannado.drl.mdp import MDPSpec
from wannado.drl.workspace import EnvInterface

_LAB_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _LAB_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
'''

    norm_mode = mode.strip().lower()
    if norm_mode in ("gym", "gymnasium", "dm_control", "existing_gym", "legacy_gym", "existing_gym_or_gymnasium"):
        body = f'''
def heuristic_baseline(obs: np.ndarray, info: dict[str, Any]) -> Any:
    """Optional Stage 1 domain heuristic baseline (e.g., proportional rule or greedy policy)."""
    # TODO: Replace with your simple rule-based baseline, or return None from EnvInterface.
    return 0 if {spec.action_spec.kind == "discrete"!r} else np.zeros({spec.action_spec.dim}, dtype=np.float32)


def load_env_interface() -> EnvInterface:
    """Wrap any Gymnasium / legacy OpenAI Gym / dm_control environment cleanly."""
    spec = MDPSpec.load(_LAB_DIR / "mdp_spec.json")

    # TODO: Replace with your project's environment constructor (Gymnasium, legacy `gym`, or `dm_control`):
    #   import gym  # legacy 4-tuple step() works automatically via UniversalGymAdapter!
    #   def make_env(): return gym.make("MyLegacyEnv-v0")
    def make_env() -> gym.Env:
        return gym.make("CartPole-v1")

    return EnvInterface(
        env_or_factory=make_env,
        mdp_spec=spec,
        env_library="auto",
        task_name=spec.problem_name,
        domain=spec.domain,
        heuristic_fn=heuristic_baseline,
    )
'''
    elif norm_mode in ("multi_agent", "multi_agent_env"):
        body = '''
def load_env_interface() -> EnvInterface:
    """Wire a PettingZoo ParallelEnv or dict-stepped multi-agent simulator."""
    spec = MDPSpec.load(_LAB_DIR / "mdp_spec.json")
    # TODO: Replace with your PettingZoo / multi-agent env factory:
    #   from my_project.marl_env import make_parallel_env
    #   return EnvInterface(
    #       env_or_factory=make_parallel_env,
    #       mdp_spec=spec,
    #       env_library="pettingzoo",
    #       task_name=spec.problem_name,
    #       domain=spec.domain,
    #   )
    raise NotImplementedError("Wire your PettingZoo / multi-agent env factory inside drl_lab/interface.py")
'''
    elif norm_mode in ("reasoning_sequence", "reasoning_sequence_task"):
        body = '''
from wannado.drl.core import SequenceTaskEnv


def load_env_interface() -> EnvInterface:
    """Wire a verifiable prompt-to-sequence / tool-use environment for GRPO / DAPO / GSPO / ARPO."""
    spec = MDPSpec.load(_LAB_DIR / "mdp_spec.json")
    # TODO: Customize sample_prompts_fn, score_sequence_fn, and optional tool_step_fn:
    task = SequenceTaskEnv(
        vocab_size=16,
        prompt_len=6,
        max_response_len=6,
        task_name=spec.problem_name,
    )
    return EnvInterface(
        env_or_factory=task,
        mdp_spec=spec,
        env_library="sequence_task",
        task_name=spec.problem_name,
        domain=spec.domain,
    )
'''
    else:
        # Raw custom simulator -> strong-contract Gymnasium wrapper
        obs_dim = max(1, sum(f.dim for f in spec.state_features))
        is_disc = spec.action_spec.kind == "discrete"
        act_space_expr = (
            f"spaces.Discrete({spec.action_spec.dim})"
            if is_disc
            else f"spaces.Box(low=-1.0, high=1.0, shape=({spec.action_spec.dim},), dtype=np.float32)"
        )
        body = f'''
class CustomSimEnv(gym.Env):
    """Contract-enforced Gymnasium adapter wrapping your external simulator/script.

    Explicitly separates `reward_terms` (for Stage 3 Reward Hacking detection),
    `task_metrics` (independent domain KPIs), `action_mask` (if discrete invalid moves exist),
    and `terminated` vs `truncated` boundaries.
    """

    metadata = {{"render_modes": []}}

    def __init__(self, max_steps: int = {spec.termination_spec.truncated_horizon}) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=({obs_dim},), dtype=np.float32)
        self.action_space = {act_space_expr}
        self._step_count = 0
        self._state = np.zeros({obs_dim}, dtype=np.float32)
        # TODO: Initialize your project simulator here:
        #   from my_project.sim import Simulator
        #   self.sim = Simulator()

    def _build_obs(self) -> np.ndarray:
        # [Anti-Leakage Rule]: Read ONLY measurements available BEFORE choosing action_t.
        return self._state.copy()

    def valid_action_mask(self) -> np.ndarray | None:
        # Return boolean mask of shape (action_dim,) if invalid discrete choices exist, else None.
        return None

    def _compute_reward_terms(self, action: Any) -> tuple[float, dict[str, float], dict[str, float]]:
        # Return (total_scalar_reward, reward_terms_dict, independent_task_kpis_dict).
        # TODO: Replace placeholder physics with your real simulator step & KPIs.
        progress = float(-np.sum(np.square(self._state)))
        terms = {{"primary_objective": progress}}
        kpis = {{"tracking_error": float(np.linalg.norm(self._state))}}
        return progress, terms, kpis

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._state = self.np_random.uniform(-0.5, 0.5, size=({obs_dim},)).astype(np.float32)
        info: dict[str, Any] = {{}}
        mask = self.valid_action_mask()
        if mask is not None:
            info["action_mask"] = mask
        return self._build_obs(), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        # TODO: Advance your simulator by 1 control tick using `action`.
        if {is_disc!r}:
            delta = 0.1 if int(action) == 1 else -0.1
            self._state = np.clip(self._state + delta, -2.0, 2.0).astype(np.float32)
        else:
            a_vec = np.asarray(action, dtype=np.float32).reshape(-1)
            n_ctrl = min({obs_dim}, a_vec.shape[0])
            self._state[:n_ctrl] = self._state[:n_ctrl] + 0.1 * a_vec[:n_ctrl]
            self._state = np.clip(self._state, -2.0, 2.0).astype(np.float32)

        reward, reward_terms, task_metrics = self._compute_reward_terms(action)
        terminated = bool(np.linalg.norm(self._state) < 0.05)  # True ONLY on physical goal/crash!
        truncated = bool(self._step_count >= self.max_steps and not terminated)

        info: dict[str, Any] = {{
            "reward_terms": reward_terms,
            "task_metrics": task_metrics,
        }}
        mask = self.valid_action_mask()
        if mask is not None:
            info["action_mask"] = mask
        return self._build_obs(), reward, terminated, truncated, info


def heuristic_baseline(obs: np.ndarray, info: dict[str, Any]) -> Any:
    """Simple domain heuristic baseline run during Stage 1 before DRL training."""
    if {is_disc!r}:
        return 0 if float(obs[0]) > 0 else 1
    return np.clip(-obs[:{spec.action_spec.dim}], -1.0, 1.0).astype(np.float32)


def load_env_interface() -> EnvInterface:
    """Entrypoint connecting your external simulator cleanly into wannado-drl."""
    spec = MDPSpec.load(_LAB_DIR / "mdp_spec.json")
    return EnvInterface(
        env_or_factory=lambda: CustomSimEnv(),
        mdp_spec=spec,
        env_library="custom_sim",
        task_name=spec.problem_name,
        domain=spec.domain,
        heuristic_fn=heuristic_baseline,
    )
'''
    return header + body


def _runner_template() -> str:
    return '''"""Standalone non-intrusive CleanRL-style training & paper reporting runner (`drl_lab/run_train.py`).

Run directly from your project root without touching your simulator/model files:
    python drl_lab/run_train.py --algos ppo --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .interface import load_env_interface
except ImportError:
    from interface import load_env_interface

from wannado.drl.audit import preflight_env
from wannado.drl.copilot import build_post_run_debrief, estimate_runtime, recommend_algo_suite
from wannado.drl.reporter import build_algo_config, evaluate_drl_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated wannado-drl Stage 1-4 training & audit.")
    parser.add_argument("--algos", nargs="+", default=None, help="Algorithm IDs (e.g. ppo double_dqn sac).")
    parser.add_argument("--goal", default="standard", choices=["learning", "sanity", "standard", "paper_sota"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps per seed across selected algos.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override deterministic evaluation episode count.")
    parser.add_argument("--gamma", type=float, default=None, help="Override discount factor gamma.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--eps-end", type=float, default=None, help="Override final greedy epsilon (for DQN/Q-learning).")
    parser.add_argument("--num-envs", type=int, default=None, help="Parallel VectorEnv count for batched rollout/replay.")
    parser.add_argument("--extractor", default=None, help="Observation extractor (auto, flatten, nature_cnn, impala, set_attention, dict).")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for atomic best/last checkpoints & progress logs.")
    parser.add_argument("--eval-freq", type=int, default=None, help="Periodic deterministic eval & best-model selection interval (steps).")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path (.pt) to deterministically resume training.")
    parser.add_argument("--output-dir", default=None, help="Custom directory to store reports, figures, CSV, and BibTeX.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()

    iface = load_env_interface()
    pf = preflight_env(iface.env_or_factory, heuristic_fn=iface.heuristic_fn)
    print(pf.format_report())
    if not pf.ok:
        raise SystemExit(1)

    if args.algos is None:
        matched_spec = (
            iface.mdp_spec
            if (iface.mdp_spec is not None and iface.mdp_spec.action_spec.kind == pf.action_space_kind)
            else None
        )
        suite = recommend_algo_suite(
            mdp_spec=matched_spec,
            action_space=pf.action_space_kind,
            goal=args.goal,
            sim_ms_per_step=pf.ms_per_step,
        )
        algos = suite.algos
        print(f"\\n[Recommended Algorithms]: {algos}\\n  Rationale: {suite.rationale}")
    else:
        algos = args.algos

    # Example single-file customization hook (after `wannado-drl eject ppo --dest ./drl_lab/algos`):
    #   from algos.ppo_schulman2017 import train_ppo
    #   algos = [train_ppo, *algos[1:]]

    overrides = {
        "total_timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "gamma": args.gamma,
        "lr": args.lr,
        "eps_end": args.eps_end,
        "num_envs": args.num_envs,
        "extractor_type": args.extractor,
        "checkpoint_dir": args.checkpoint_dir,
        "eval_freq": args.eval_freq,
        "resume_from": args.resume_from,
        "device": args.device,
    }
    algo_configs = {}
    est_steps: list[int] = []
    for aid in algos:
        if isinstance(aid, str):
            cfg = build_algo_config(aid, **overrides)
            if cfg is not None:
                algo_configs[aid] = cfg
                if hasattr(cfg, "total_timesteps"):
                    est_steps.append(int(cfg.total_timesteps))

    avg_steps = int(sum(est_steps) / len(est_steps)) if est_steps else 15_000
    eta = estimate_runtime(
        preflight_report=pf,
        algos=[a for a in algos if isinstance(a, str)],
        total_timesteps=avg_steps,
        num_seeds=len(args.seeds),
        device=args.device,
    )
    print("\\n" + eta.format_summary())

    out_dir = Path(args.output_dir).resolve() if args.output_dir else Path(__file__).resolve().parent / "outputs"
    report = evaluate_drl_suite(
        iface.env_or_factory,
        algos=algos,
        seeds=args.seeds,
        algo_configs=algo_configs or None,
        mdp_spec=iface.mdp_spec,
        heuristic_fn=iface.heuristic_fn,
        preflight_report=pf,
        task_name=iface.task_name,
        domain=iface.domain,
        run_preflight=False,
        run_audit=(not args.no_audit),
        device=args.device,
    )
    saved = report.save(out_dir)
    debrief = build_post_run_debrief(report, output_dir=out_dir)
    print("\\n" + report.format_paper_summary(chart_filename=saved["overview_figure"].name, output_dir=out_dir))
    print("\\n[Plain Verdict]: " + debrief["plain_verdict"])
    print(f"\\nSaved paper reporting artifacts to: {out_dir} ({list(saved.keys())})")


if __name__ == "__main__":
    main()
'''


def scaffold_lab(
    target_dir: str | Path = "./drl_lab",
    *,
    archetype: str = "custom_sim",
    mode: str = "custom_sim",
    mdp_spec: MDPSpec | None = None,
    problem_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Generate an isolated `drl_lab/` external plug-in workspace for a user project."""
    root = Path(target_dir).resolve()
    algos_dir = root / "algos"
    root.mkdir(parents=True, exist_ok=True)
    algos_dir.mkdir(parents=True, exist_ok=True)

    spec = mdp_spec or build_mdp_archetype(archetype, problem_name=problem_name)
    files_to_write: dict[Path, str] = {
        root / "mdp_spec.json": spec.to_json() + "\n",
        root / "interface.py": _interface_template(mode, spec),
        root / "run_train.py": _runner_template(),
        algos_dir / "__init__.py": '"""Local customized CleanRL single-file algorithms ejected via `wannado-drl eject <algo_id>`."""\n',
    }

    written: dict[str, Path] = {}
    for fpath, content in files_to_write.items():
        if fpath.exists() and not overwrite:
            written[fpath.relative_to(root).as_posix()] = fpath
            continue
        fpath.write_text(content, encoding="utf-8")
        written[fpath.relative_to(root).as_posix()] = fpath
    return written
