"""Project traversal and RL/simulator detection for AI-Native Stage 0 & Stage 1 workflows."""

from __future__ import annotations

import ast
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".agent",
    ".agents",
    ".claude",
    ".cursor",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "egg-info",
    ".ipynb_checkpoints",
}

_DOMAIN_SIGNALS: dict[str, tuple[str, ...]] = {
    "wireless_scheduling": ("ofdm", "sinr", "snr", "cqi", "channel", "throughput", "base_station", "resource_block", "tti", "queue"),
    "continuous_control": ("torque", "velocity", "joint", "pendulum", "mujoco", "dm_control", "pid", "trajectory", "robotics", "gripper"),
    "finance_portfolio_risk": ("portfolio", "sharpe", "cvar", "drawdown", "trading", "ticker", "ohlcv", "backtest", "allocation", "anytrading", "asset_weights"),
    "clinical_offline_safety": ("clinical", "sepsis", "icu", "patient", "ehr", "dosage", "vitals", "sofa", "mortality", "treatment", "mimic", "medical"),
    "cyber_attack_defense": ("cyber", "intrusion", "firewall", "honeypot", "malware", "pentest", "vulnerability", "network_defense", "nasim", "cyborg"),
    "multi_agent": ("pettingzoo", "parallel_env", "aec_env", "possible_agents", "mappo", "qmix", "vdn", "multi_agent"),
    "reasoning_sequence": ("prompt", "tokenizer", "reasoning", "verifier", "grpo", "dapo", "gspo", "arpo", "tool_call"),
    "offline_rl": ("offline_dataset", "d4rl", "replay_dataset", "behavior_cloning", "logged_transitions", "cql", "iql"),
    "inventory_routing": ("inventory", "demand", "warehouse", "dispatch", "vehicle", "routing", "stockout", "backlog"),
}


_ARCHETYPE_GOAL_TEMPLATES: dict[str, tuple[str, str]] = {
    "wireless_scheduling": (
        "This looks like a wireless communications / radio resource allocation simulator or scheduler.",
        "Train a DRL policy to dynamically allocate channels, subcarriers, transmission slots, or power levels using live CSI/SINR and queue backlog to boost throughput/QoS.",
    ),
    "continuous_control": (
        "This looks like a physics / robotics / continuous feedback control project.",
        "Train a continuous-action DRL controller to apply smooth torques, velocities, or actuator commands for stable trajectory tracking or manipulation.",
    ),
    "finance_portfolio_risk": (
        "This looks like a quantitative finance, portfolio allocation, or algorithmic trading risk-control project.",
        "Train a risk-sensitive DRL allocation policy on simplex/portfolio weights maximizing Sharpe/return net of transaction costs while controlling CVaR drawdown risk.",
    ),
    "clinical_offline_safety": (
        "This looks like a clinical / healthcare decision support or retrospective patient trajectory project.",
        "Train a conservative offline DRL policy (CQL/IQL) with clinical safety action masks and validate via WIS/FQE Off-Policy Evaluation before live deployment.",
    ),
    "cyber_attack_defense": (
        "This looks like a cybersecurity / automated network defense or intrusion response simulation.",
        "Train an action-masked or multi-discrete DRL defender (PPO/Double-DQN) balancing threat containment against host/service availability.",
    ),
    "multi_agent": (
        "This looks like a multi-agent decision / decentralized coordination environment.",
        "Train coordinated multi-agent policies (CTDE value factorization or MAPPO) where agents collaborate or compete under shared/local rewards.",
    ),
    "reasoning_sequence": (
        "This looks like a sequence-generation / LLM reasoning or verifiable tool-use pipeline.",
        "Fine-tune a sequence policy with group-relative verifiable rewards (GRPO/DAPO/GSPO/ARPO) to improve multi-step reasoning or tool invocation accuracy.",
    ),
    "offline_rl": (
        "This project contains logged trajectory datasets for offline decision optimization without a live simulator.",
        "Train conservative offline DRL policies (CQL/IQL) strictly from logged trajectories while preventing out-of-distribution action hallucination.",
    ),
    "inventory_routing": (
        "This looks like an operations research / inventory, dispatch, or routing optimization project.",
        "Train a sequential dispatching/replenishment DRL agent balancing service level against holding, backlog, or travel costs.",
    ),
    "custom_sim": (
        "This project contains custom Python modules / RL environment hooks without a narrow preset domain tag.",
        "Wrap your simulator step logic into an explicit MDP and train a sequential decision agent balancing task performance and safety constraints.",
    ),
}


def _extract_readme_summary(root: Path) -> tuple[str, str]:
    """Extract clean title and short first-paragraph summary from README if present."""
    for name in ("README.md", "readme.md", "README.rst", "README.txt"):
        fpath = root / name
        if not fpath.is_file():
            continue
        try:
            raw = fpath.read_text(encoding="utf-8", errors="ignore")[:4096]
        except OSError:
            continue
        title = ""
        para_lines: list[str] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith(("<", "![", "[![")):
                continue
            if not title and s.startswith("#"):
                title = s.lstrip("#").strip()
                continue
            if s.startswith(("#", "---")):
                if para_lines:
                    break
                continue
            para_lines.append(s)
            if sum(len(p) for p in para_lines) >= 220:
                break
        excerpt = " ".join(para_lines)[:240].strip()
        return title, excerpt
    return "", ""


@dataclass
class ProjectScan:
    """Fresh structural snapshot & AI project-understanding hypothesis for Stage 0/1 planning."""

    root: str
    scanned_files_count: int
    total_python_files: int
    frameworks: list[str] = field(default_factory=list)
    integration_mode: str = "raw_simulator_needs_mdp"
    likely_archetype: str = "custom_sim"
    env_files: list[str] = field(default_factory=list)
    simulator_candidates: list[str] = field(default_factory=list)
    entry_candidates: list[str] = field(default_factory=list)
    existing_lab_dirs: list[str] = field(default_factory=list)
    ejected_algo_files: list[str] = field(default_factory=list)
    checkpoint_files: list[str] = field(default_factory=list)
    dataset_files: list[str] = field(default_factory=list)
    fingerprint: str = ""
    readme_title: str = ""
    readme_excerpt: str = ""
    detected_classes_and_funcs: list[str] = field(default_factory=list)
    project_summary_guess: str = ""
    understanding_confidence: str = "low"
    inferred_drl_goal: str = ""
    inferred_action_space: str = "unknown"
    candidate_goals: list[str] = field(default_factory=list)
    evidence_clues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_summary(self) -> str:
        lines = [
            f"[wannado-drl Project Scan] root={self.root}",
            f"  Python files scanned: {self.scanned_files_count} (total={self.total_python_files})",
            f"  Detected frameworks : {', '.join(self.frameworks) if self.frameworks else 'none (pure Python/custom)'}",
            f"  Integration mode    : {self.integration_mode} (suggested Stage 0 archetype='{self.likely_archetype}')",
            f"  AI Understanding    : [{self.understanding_confidence.upper()}] {self.project_summary_guess}",
            f"  Inferred DRL Goal   : {self.inferred_drl_goal}",
        ]
        if self.evidence_clues:
            lines.append(f"  Observed clues      : {'; '.join(self.evidence_clues[:6])}")
        if self.env_files:
            lines.append(f"  Existing Env classes: {', '.join(self.env_files[:8])}")
        if self.simulator_candidates:
            lines.append(f"  Simulator/Step files: {', '.join(self.simulator_candidates[:8])}")
        if self.entry_candidates:
            lines.append(f"  Train/Run scripts   : {', '.join(self.entry_candidates[:8])}")
        if self.existing_lab_dirs:
            lines.append(f"  Isolated drl_lab    : {', '.join(self.existing_lab_dirs)}")
        if self.ejected_algo_files:
            lines.append(f"  Local ejected algos : {', '.join(self.ejected_algo_files)}")
        return "\n".join(lines)


def _iter_repo_files(root: Path, max_files: int) -> tuple[list[Path], int, list[str], list[str]]:
    py_files: list[Path] = []
    ckpts: list[str] = []
    datasets: list[str] = []
    total_py = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS and not d.endswith(".egg-info"))
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            low = fname.lower()
            if fname.endswith(".py"):
                total_py += 1
                if len(py_files) < max_files:
                    py_files.append(fpath)
            elif len(ckpts) < 10 and low.endswith((".pt", ".pth", ".ckpt", ".safetensors")):
                ckpts.append(fpath.relative_to(root).as_posix())
            elif len(datasets) < 10 and low.endswith((".npz", ".hdf5", ".parquet", ".jsonl")):
                datasets.append(fpath.relative_to(root).as_posix())
    return py_files, total_py, ckpts, datasets


def scan_project(project_dir: str | Path = ".", max_files: int = 200) -> ProjectScan:
    """Inspect a project directory to identify RL libraries, existing Envs, or raw simulators needing Stage 0 MDP intake."""
    root = Path(project_dir).resolve()
    py_files, total_py, ckpts, datasets = _iter_repo_files(root, max_files=max_files)

    fw_map = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "gymnasium": "Gymnasium",
        "gym": "Legacy OpenAI Gym",
        "pettingzoo": "PettingZoo",
        "dm_control": "dm_control",
        "stable_baselines3": "Stable-Baselines3",
        "cleanrl": "CleanRL",
        "trl": "TRL",
        "verl": "verl",
        "simpy": "SimPy",
    }
    frameworks: set[str] = set()
    env_files: list[str] = []
    sim_candidates: list[str] = []
    entry_candidates: list[str] = []
    ejected_files: list[str] = []
    detected_symbols: list[str] = []
    doc_snippets: list[str] = []
    matched_kws: dict[str, list[str]] = {k: [] for k in _DOMAIN_SIGNALS}
    domain_scores: dict[str, int] = {k: 0 for k in _DOMAIN_SIGNALS}
    action_space_hits: dict[str, int] = {"continuous": 0, "discrete": 0, "multi_discrete": 0}
    fp_parts: list[str] = []

    for fpath in py_files:
        rel = fpath.relative_to(root).as_posix()
        low_rel = rel.lower()
        is_lab_path = "drl_lab/" in low_rel or "wannado_drl_lab/" in low_rel
        try:
            st = fpath.stat()
            fp_parts.append(f"{rel}:{int(st.st_mtime)}:{st.st_size}")
            text = fpath.read_text(encoding="utf-8", errors="ignore")[:131072]
        except OSError:
            continue
        low_txt = text.lower()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if alias.name.startswith("wannado.drl"):
                            frameworks.add("wannado-drl")
                        elif top in fw_map:
                            frameworks.add(fw_map[top])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if node.module.startswith("wannado.drl"):
                        frameworks.add("wannado-drl")
                    elif top in fw_map:
                        frameworks.add(fw_map[top])
                elif (
                    not is_lab_path
                    and isinstance(node, ast.ClassDef)
                    and not node.name.startswith("_")
                    and len(detected_symbols) < 15
                ):
                    detected_symbols.append(f"class {node.name} ({rel})")
            if not is_lab_path:
                mod_doc = ast.get_docstring(tree)
                if mod_doc and len(doc_snippets) < 4:
                    first_line = mod_doc.strip().splitlines()[0].strip()
                    if first_line:
                        doc_snippets.append(f"{rel}: {first_line[:140]}")
        except SyntaxError:
            pass

        if not is_lab_path:
            if re.search(r"class\s+\w+\s*\([^)]*(gym\.Env|gymnasium\.Env|ParallelEnv|AECEnv|Env)\b", text):
                env_files.append(rel)
            if "def step(" in text or "def simulate(" in text or "def schedule(" in text:
                sim_candidates.append(rel)
            if any(tok in low_rel for tok in ("train", "run", "main", "sim", "env", "sched")):
                entry_candidates.append(rel)
            if re.search(r"action_space\s*=\s*(?:spaces\.|gym\.spaces\.|gymnasium\.spaces\.)?Box\b", text) or "spaces.box" in low_txt:
                action_space_hits["continuous"] += 1
            if re.search(r"action_space\s*=\s*(?:spaces\.|gym\.spaces\.|gymnasium\.spaces\.)?MultiDiscrete\b", text):
                action_space_hits["multi_discrete"] += 1
            if re.search(r"action_space\s*=\s*(?:spaces\.|gym\.spaces\.|gymnasium\.spaces\.)?Discrete\b", text):
                action_space_hits["discrete"] += 1
            for dom, kws in _DOMAIN_SIGNALS.items():
                hits = [kw for kw in kws if kw in low_txt]
                if hits:
                    domain_scores[dom] += len(hits)
                    for kw in hits:
                        if kw not in matched_kws[dom]:
                            matched_kws[dom].append(kw)
        if "drl_lab/algos/" in low_rel or ("AlgoMeta" in text and "def train_" in text):
            ejected_files.append(rel)

    best_arch = max(domain_scores.items(), key=lambda kv: kv[1])[0]
    best_score = domain_scores[best_arch]
    if best_score == 0:
        best_arch = "custom_sim"

    if "PettingZoo" in frameworks or best_arch == "multi_agent":
        mode = "multi_agent_env"
    elif best_arch == "reasoning_sequence":
        mode = "reasoning_sequence_task"
    elif best_arch in ("offline_rl", "clinical_offline_safety") or (datasets and not env_files and not sim_candidates):
        mode = "offline_dataset"
    elif env_files or "Gymnasium" in frameworks or "Legacy OpenAI Gym" in frameworks or "dm_control" in frameworks:
        mode = "existing_gym_or_gymnasium"
    else:
        mode = "raw_simulator_needs_mdp"

    inferred_action_space = "unknown"
    if action_space_hits["multi_discrete"] > 0:
        inferred_action_space = "multi_discrete"
    elif action_space_hits["continuous"] > action_space_hits["discrete"]:
        inferred_action_space = "continuous"
    elif action_space_hits["discrete"] > 0:
        inferred_action_space = "discrete"
    elif best_arch in ("continuous_control", "finance_portfolio_risk"):
        inferred_action_space = "continuous"
    elif best_arch == "cyber_attack_defense":
        inferred_action_space = "discrete"

    readme_title, readme_excerpt = _extract_readme_summary(root)
    existing_lab_dirs = [d for d in ("drl_lab", "wannado_drl_lab") if (root / d).is_dir()]

    clues: list[str] = []
    if readme_title or readme_excerpt:
        clues.append(f"README: {readme_title or readme_excerpt}")
    if env_files:
        clues.append(f"Env classes in {', '.join(env_files[:3])}")
    if sim_candidates:
        clues.append(f"Simulator step/schedule hooks in {', '.join(sim_candidates[:3])}")
    if best_score > 0:
        clues.append(f"Domain signals ({best_arch}): {', '.join(matched_kws[best_arch][:6])}")
    if doc_snippets:
        clues.append(f"Module note: {doc_snippets[0]}")
    if datasets:
        clues.append(f"Offline trajectory datasets: {', '.join(datasets[:3])}")
    if detected_symbols and len(clues) < 4:
        clues.append(f"Symbols: {', '.join(detected_symbols[:3])}")

    has_rl_framework = bool(
        frameworks & {"Gymnasium", "Legacy OpenAI Gym", "PettingZoo", "dm_control", "Stable-Baselines3", "CleanRL", "TRL", "verl"}
    )
    base_summary, base_goal = _ARCHETYPE_GOAL_TEMPLATES.get(best_arch, _ARCHETYPE_GOAL_TEMPLATES["custom_sim"])
    if best_score >= 3 or (best_score >= 1 and (env_files or sim_candidates)):
        confidence = "high"
        summary_guess = base_summary + (f" Key clues: {clues[0]}." if clues else "")
        inferred_goal = base_goal
    elif best_score >= 1 and has_rl_framework:
        confidence = "medium"
        summary_guess = f"{base_summary} Detected RL library ({', '.join(sorted(frameworks))})" + (
            f" and clues ({clues[0]})." if clues else "."
        )
        inferred_goal = base_goal
    elif env_files or sim_candidates or readme_excerpt or datasets or has_rl_framework:
        confidence = "medium"
        detail = readme_excerpt or (doc_snippets[0] if doc_snippets else (clues[0] if clues else f"frameworks={sorted(frameworks)}"))
        summary_guess = f"{base_summary} Partial clues seen ({detail}); confirm exact objective with user."
        inferred_goal = (
            f"Train a DRL agent on your {'/'.join(sorted(frameworks))} environment ({doc_snippets[0]})"
            if has_rl_framework and doc_snippets
            else base_goal
        )
    else:
        confidence = "low"
        file_hint = f"{total_py} Python file(s)" if total_py else "no Python simulator files yet"
        summary_guess = (
            f"Scanned {file_hint}; no explicit RL Env (`step`/`reset`), simulator keywords, or README domain goal detected yet. "
            "Share what was observed first and ask the user to explain their target system."
        )
        inferred_goal = "Clarify the sequential decision problem in one sentence before designing the MDP."

    candidate_goals = [inferred_goal]
    for dom, sc in sorted(domain_scores.items(), key=lambda kv: kv[1], reverse=True):
        if sc > 0 and dom != best_arch:
            candidate_goals.append(_ARCHETYPE_GOAL_TEMPLATES[dom][1])

    return ProjectScan(
        root=str(root),
        scanned_files_count=len(py_files),
        total_python_files=total_py,
        frameworks=sorted(frameworks),
        integration_mode=mode,
        likely_archetype=best_arch,
        env_files=env_files[:15],
        simulator_candidates=sim_candidates[:15],
        entry_candidates=entry_candidates[:15],
        existing_lab_dirs=existing_lab_dirs,
        ejected_algo_files=ejected_files[:20],
        checkpoint_files=ckpts,
        dataset_files=datasets,
        fingerprint=str(hash("|".join(fp_parts))),
        readme_title=readme_title,
        readme_excerpt=readme_excerpt,
        detected_classes_and_funcs=detected_symbols,
        project_summary_guess=summary_guess,
        understanding_confidence=confidence,
        inferred_drl_goal=inferred_goal,
        inferred_action_space=inferred_action_space,
        candidate_goals=candidate_goals[:3],
        evidence_clues=clues,
    )
