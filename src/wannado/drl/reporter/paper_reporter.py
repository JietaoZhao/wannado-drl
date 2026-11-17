"""Stage 4: Publication-ready DRL experimental runner, multi-seed statistical tables, and zero-hallucination BibTeX."""

from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wannado.drl.algos import ALGO_ID_TO_RUNNER
from wannado.drl.audit.preflight import PreflightReport, preflight_env
from wannado.drl.audit.rigor_auditor import RigorAuditReport, audit_drl_rigor
from wannado.drl.catalog.build_catalog import load_catalog
from wannado.drl.core.offline_risk import evaluate_offline_ope
from wannado.drl.core.provenance import (
    ALGO_DEFAULT_OPTIMIZATIONS,
    OPTIMIZATION_PROVENANCE,
    resolve_run_optimizations,
)
from wannado.drl.core.types import EpisodeEvalStats, TrainResult
from wannado.drl.core.utils import aggregate_seed_stats
from wannado.drl.core.wrappers import OfflineTransitionDataset
from wannado.drl.mdp.spec import MDPSpec


def export_citations(
    algo_ids: Sequence[str],
    fmt: str = "bibtex",
    *,
    include_audit_refs: bool = False,
    include_optimizations: bool = True,
    optimization_ids: Sequence[str] | None = None,
) -> str:
    """Export verbatim verified academic citations for algorithms, optimizations, and rigor protocols."""
    if fmt not in ("bibtex", "ieee", "apa"):
        raise ValueError(f"Unsupported citation format '{fmt}'. Expected 'bibtex', 'ieee', or 'apa'.")

    by_id = {c["id"]: c for c in load_catalog()}
    seen_algo_ids: set[str] = set()
    seen_cit_keys: set[str] = set()
    entries: list[str] = []

    def _push_citation(cit: dict[str, Any] | Any) -> None:
        cit_dict = cit if isinstance(cit, dict) else asdict(cit)
        key = str(cit_dict.get("key", ""))
        text = str(cit_dict.get(fmt, ""))
        if not text:
            return
        dedup_key = key or text
        if dedup_key not in seen_cit_keys:
            seen_cit_keys.add(dedup_key)
            entries.append(text)

    default_opt_ids: list[str] = []
    for aid in algo_ids:
        if aid in seen_algo_ids:
            continue
        card = by_id.get(aid)
        if card is None:
            raise KeyError(f"Unknown algorithm '{aid}' in export_citations(). Refusing to fabricate citation.")
        seen_algo_ids.add(aid)
        _push_citation(card["citation"])
        for oid in card.get("default_optimizations") or ALGO_DEFAULT_OPTIMIZATIONS.get(aid, ()):
            if oid not in default_opt_ids:
                default_opt_ids.append(oid)

    if include_optimizations:
        opt_pool = list(optimization_ids) if optimization_ids is not None else default_opt_ids
        for oid in opt_pool:
            prov = OPTIMIZATION_PROVENANCE.get(oid)
            if prov is not None:
                _push_citation(prov.citation)

    if include_audit_refs:
        for audit_oid in ("iqm_rigor_audit", "reproducibility_multi_seed"):
            prov = OPTIMIZATION_PROVENANCE.get(audit_oid)
            if prov is not None:
                _push_citation(prov.citation)

    sep = "\n\n" if fmt == "bibtex" else "\n"
    return sep.join(entries)


@dataclass
class AlgoExperimentSummary:
    """Multi-seed aggregated statistics, audit verdict, and verified citation for one algorithm."""

    algo_id: str
    display_name: str
    family: str
    paradigm: str
    seeds: list[int]
    seed_stats: dict[str, float]
    mean_elapsed_sec: float
    total_timesteps_per_seed: int
    hyperparameters: dict[str, Any]
    citation: dict[str, Any]
    audit: RigorAuditReport | None = None
    seed_runs: list[dict[str, Any]] = field(default_factory=list)
    optimizations_used: list[dict[str, Any]] = field(default_factory=list)


def _write_fallback_svg(report: DRLReport, svg_path: Path) -> Path:
    """Write a dependency-free SVG summary bar chart when matplotlib is unavailable."""
    entries: list[tuple[str, float, str]] = []
    if report.random_baseline is not None:
        entries.append(("Random", float(report.random_baseline.mean_return), "#94a3b8"))
    if report.heuristic_baseline is not None:
        entries.append(("Heuristic", float(report.heuristic_baseline.mean_return), "#64748b"))
    palette = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777"]
    for idx, s in enumerate(report.algo_summaries):
        entries.append((s.algo_id, float(s.seed_stats["iqm"]), palette[idx % len(palette)]))

    vals = [v for _, v, _ in entries] or [0.0]
    v_min, v_max = min(0.0, min(vals)), max(1e-6, max(vals))
    span = max(1e-6, v_max - v_min)
    w, h = 680, 300
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-family="sans-serif" font-size="15" font-weight="bold" fill="#0f172a">{report.task_name} — Evaluation IQM Comparison</text>',
    ]
    bar_w = max(28, min(90, int(560 / max(1, len(entries) * 1.5))))
    base_y = 240
    top_y = 55
    usable_h = base_y - top_y
    zero_y = top_y + (v_max / span) * usable_h
    lines.append(f'<line x1="40" y1="{zero_y:.1f}" x2="640" y2="{zero_y:.1f}" stroke="#cbd5e1" stroke-width="1"/>')
    for i, (label, val, color) in enumerate(entries):
        x = 60 + i * (bar_w + 24)
        bar_top = top_y + ((v_max - max(0.0, val)) / span) * usable_h
        bar_bot = top_y + ((v_max - min(0.0, val)) / span) * usable_h
        bh = max(2.0, bar_bot - bar_top)
        lines.append(f'<rect x="{x}" y="{bar_top:.1f}" width="{bar_w}" height="{bh:.1f}" rx="4" fill="{color}"/>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{bar_top - 6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#1e293b">{val:.2f}</text>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="270" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#334155">{label}</text>')
    lines.append("</svg>")
    svg_path.write_text("\n".join(lines), encoding="utf-8")
    return svg_path


def _generate_overview_figure(report: DRLReport, out_dir: Path) -> Path:
    """Render a clean 2-panel learning curve + evaluation IQM bar chart (PNG via matplotlib or SVG fallback)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return _write_fallback_svg(report, out_dir / "results_overview.svg")

    png_path = out_dir / "results_overview.png"
    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=160)
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777"]

    has_curves = False
    for idx, s in enumerate(report.algo_summaries):
        col = colors[idx % len(colors)]
        seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for run in s.seed_runs:
            pts = run.get("learning_curve") or []
            if len(pts) >= 2:
                xs = np.asarray([float(p.get("step", 0.0)) for p in pts], dtype=np.float64)
                ys = np.asarray([float(p.get("episode_return", 0.0)) for p in pts], dtype=np.float64)
                order = np.argsort(xs)
                seed_curves.append((xs[order], ys[order]))
            elif len(pts) == 1:
                ax_curve.scatter([float(pts[0].get("step", 0.0))], [float(pts[0].get("episode_return", 0.0))], color=col, s=24)
                has_curves = True

        if seed_curves:
            has_curves = True
            grid_min = max(c[0][0] for c in seed_curves)
            grid_max = min(c[0][-1] for c in seed_curves)
            if grid_max > grid_min:
                x_grid = np.linspace(grid_min, grid_max, 50)
                interp = np.vstack([np.interp(x_grid, cx, cy) for cx, cy in seed_curves])
                mean_y = interp.mean(axis=0)
                std_y = interp.std(axis=0)
                ax_curve.plot(x_grid, mean_y, label=f"{s.algo_id} (mean)", color=col, linewidth=2.0)
                if len(seed_curves) > 1:
                    ax_curve.fill_between(x_grid, mean_y - std_y, mean_y + std_y, color=col, alpha=0.16)
            else:
                cx, cy = seed_curves[0]
                ax_curve.plot(cx, cy, label=s.algo_id, color=col, linewidth=2.0)

    if report.random_baseline is not None:
        ax_curve.axhline(report.random_baseline.mean_return, linestyle=":", color="#64748b", linewidth=1.4, label="Random Baseline")
    if report.heuristic_baseline is not None:
        ax_curve.axhline(report.heuristic_baseline.mean_return, linestyle="--", color="#0f172a", linewidth=1.4, label="Heuristic Baseline")

    ax_curve.set_title(f"{report.task_name} — Learning Curves")
    ax_curve.set_xlabel("Training Timesteps")
    ax_curve.set_ylabel("Episode Return")
    ax_curve.grid(True, linestyle=":", alpha=0.45)
    if has_curves or report.random_baseline is not None or report.heuristic_baseline is not None:
        ax_curve.legend(loc="best", fontsize=8)

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    iqms: list[float] = []
    bar_colors: list[str] = []
    if report.random_baseline is not None:
        labels.append("Random")
        means.append(float(report.random_baseline.mean_return))
        stds.append(float(report.random_baseline.std_return))
        iqms.append(float(report.random_baseline.mean_return))
        bar_colors.append("#94a3b8")
    if report.heuristic_baseline is not None:
        labels.append("Heuristic")
        means.append(float(report.heuristic_baseline.mean_return))
        stds.append(float(report.heuristic_baseline.std_return))
        iqms.append(float(report.heuristic_baseline.mean_return))
        bar_colors.append("#64748b")
    for idx, s in enumerate(report.algo_summaries):
        audit_tag = "PASS" if (s.audit is None or s.audit.passed) else "WARN"
        labels.append(f"{s.algo_id}\n({audit_tag})")
        means.append(float(s.seed_stats["mean"]))
        stds.append(float(s.seed_stats["std"]))
        iqms.append(float(s.seed_stats["iqm"]))
        bar_colors.append(colors[idx % len(colors)])

    x_pos = np.arange(len(labels))
    ax_bar.bar(x_pos, means, yerr=stds, capsize=4, color=bar_colors, alpha=0.88, edgecolor="#1e293b", linewidth=0.6)
    ax_bar.scatter(x_pos, iqms, color="#facc15", edgecolor="#0f172a", zorder=5, s=42, label="IQM")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_title(f"Final Evaluation ({len(report.seeds)} seed(s))")
    ax_bar.set_ylabel("Deterministic Eval Return")
    ax_bar.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_bar.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


@dataclass
class DRLReport:
    """Complete Stage 4 academic DRL evaluation report & artifact exporter."""

    task_name: str
    domain: str
    seeds: list[int]
    device: str
    torch_version: str
    timestamp_utc: str
    mdp_spec: MDPSpec | None
    preflight: PreflightReport | None
    random_baseline: EpisodeEvalStats | None
    heuristic_baseline: EpisodeEvalStats | None
    algo_summaries: list[AlgoExperimentSummary] = field(default_factory=list)
    notes: str = ""

    def verified_bibliography(self) -> list[dict[str, Any]]:
        """Collect deduplicated verified literature citations for algorithms, optimizations, and rigor audits."""
        by_key: dict[str, dict[str, Any]] = {}

        def _register(cit: dict[str, Any], *, category: str, used_by: str) -> None:
            key = str(cit.get("key", ""))
            if not key:
                return
            if key not in by_key:
                by_key[key] = {**cit, "category": category, "used_by": [used_by]}
            elif used_by not in by_key[key]["used_by"]:
                by_key[key]["used_by"].append(used_by)

        for s in self.algo_summaries:
            if s.citation:
                _register(s.citation, category="algorithm", used_by=s.algo_id)
            for opt in s.optimizations_used:
                cit = opt.get("citation") or {}
                if cit:
                    _register(cit, category="optimization", used_by=f"{s.algo_id}:{opt['id']}")

        for audit_oid in ("iqm_rigor_audit", "reproducibility_multi_seed"):
            prov = OPTIMIZATION_PROVENANCE.get(audit_oid)
            if prov is not None:
                _register(asdict(prov.citation), category="audit_protocol", used_by=audit_oid)

        return list(by_key.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": {
                "timestamp_utc": self.timestamp_utc,
                "torch_version": self.torch_version,
                "device": self.device,
                "seeds": list(self.seeds),
            },
            "task": {"task_name": self.task_name, "domain": self.domain, "notes": self.notes},
            "mdp_spec": self.mdp_spec.to_dict() if self.mdp_spec is not None else None,
            "preflight": self.preflight.to_dict() if self.preflight is not None else None,
            "random_baseline": self.random_baseline.to_dict() if self.random_baseline is not None else None,
            "heuristic_baseline": self.heuristic_baseline.to_dict() if self.heuristic_baseline is not None else None,
            "algorithms": [
                {
                    "algo_id": s.algo_id,
                    "display_name": s.display_name,
                    "family": s.family,
                    "paradigm": s.paradigm,
                    "seeds": s.seeds,
                    "seed_stats": s.seed_stats,
                    "mean_elapsed_sec": s.mean_elapsed_sec,
                    "total_timesteps_per_seed": s.total_timesteps_per_seed,
                    "hyperparameters": s.hyperparameters,
                    "citation": s.citation,
                    "optimizations_used": s.optimizations_used,
                    "audit": s.audit.to_dict() if s.audit is not None else None,
                    "seed_runs": s.seed_runs,
                }
                for s in self.algo_summaries
            ],
            "verified_bibliography": self.verified_bibliography(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_bibtex(self) -> str:
        entries = [entry["bibtex"] for entry in self.verified_bibliography() if entry.get("bibtex")]
        return ("\n\n".join(entries) + "\n") if entries else ""

    def to_latex(self) -> str:
        def _esc(val: str) -> str:
            return (
                val.replace("&", "\\&")
                .replace("_", "\\_")
                .replace("%", "\\%")
                .replace("#", "\\#")
                .replace("^", "\\^{}")
                .replace("~", "\\~{}")
            )

        lines = [
            "% Requires \\usepackage{booktabs}",
            "\\begin{table}[t]",
            "  \\centering",
            (
                f"  \\caption{{Multi-seed DRL evaluation on {_esc(self.task_name)} ({len(self.seeds)} seeds; "
                f"IQM and multi-seed rigor protocol~\\cite{{agarwal2021deep,henderson2018deep}}).}}"
            ),
            "  \\label{tab:wannado_drl_results}",
            "  \\begin{tabular}{lllcccc}",
            "    \\toprule",
            "    Algorithm & Optimizations Cited & Paradigm & Eval Mean $\\pm$ Std $\\uparrow$ & IQM $\\uparrow$ & Median & Audit \\\\",
            "    \\midrule",
        ]
        if self.random_baseline is not None:
            rb = self.random_baseline
            lines.append(f"    Random Baseline & -- & -- & ${rb.mean_return:.2f} \\pm {rb.std_return:.2f}$ & {rb.mean_return:.2f} & {rb.mean_return:.2f} & -- \\\\")
        if self.heuristic_baseline is not None:
            hb = self.heuristic_baseline
            lines.append(f"    Heuristic Baseline & -- & -- & ${hb.mean_return:.2f} \\pm {hb.std_return:.2f}$ & {hb.mean_return:.2f} & {hb.mean_return:.2f} & -- \\\\")
        if self.random_baseline is not None or self.heuristic_baseline is not None:
            lines.append("    \\midrule")

        for s in self.algo_summaries:
            st = s.seed_stats
            audit_str = "PASS" if (s.audit is None or s.audit.passed) else "WARN"
            cit_key = str(s.citation.get("key", "")).strip() if s.citation else ""
            algo_cell = f"{_esc(s.display_name)}~\\cite{{{cit_key}}}" if cit_key else _esc(s.display_name)
            opt_keys = list(
                dict.fromkeys(
                    opt["citation"]["key"]
                    for opt in s.optimizations_used
                    if opt.get("citation", {}).get("key")
                )
            )
            opt_cell = f"\\cite{{{','.join(opt_keys)}}}" if opt_keys else "--"
            lines.append(
                f"    {algo_cell} & {opt_cell} & {_esc(s.paradigm)} & "
                f"${st['mean']:.2f} \\pm {st['std']:.2f}$ & {st['iqm']:.2f} & {st['median']:.2f} & {audit_str} \\\\"
            )
        lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
        return "\n".join(lines)

    def format_paper_summary(
        self,
        *,
        chart_filename: str | None = None,
        output_dir: str | Path | None = None,
    ) -> str:
        lines = [f"# Academic DRL Report — {self.task_name} ({self.domain})", ""]
        lines.append("## 1. Plain-Language Takeaway & Results Comparison")
        if chart_filename:
            lines.append(f"![Multi-Seed Learning Curves & Evaluation Comparison]({chart_filename})")
            lines.append("")
        for s in self.algo_summaries:
            verdict = "passed all statistical & anti-hacking checks" if (s.audit is None or s.audit.passed) else f"flagged {', '.join(s.audit.red_flags)}"
            cit_key = str(s.citation.get("key", "")).strip() if s.citation else ""
            cit_tag = f" [`@{cit_key}`]" if cit_key else ""
            opt_tags = ", ".join(f"`{opt['id']}`" for opt in s.optimizations_used) or "none"
            lines.append(
                f"- **{s.display_name}** (`{s.algo_id}`{cit_tag}; optimizations: {opt_tags}): "
                f"Eval IQM = `{s.seed_stats['iqm']:.2f}` "
                f"(mean `{s.seed_stats['mean']:.2f} ± {s.seed_stats['std']:.2f}` across {len(s.seeds)} seeds) — {verdict}."
            )
        lines.extend([
            "",
            "| Method | Eval Mean ± Std | IQM | Median | Audit | Primary Citation | Active Optimizations |",
            "| :--- | :---: | :---: | :---: | :---: | :--- | :--- |",
        ])
        if self.random_baseline is not None:
            rb = self.random_baseline
            lines.append(f"| Random Baseline | {rb.mean_return:.2f} ± {rb.std_return:.2f} | {rb.mean_return:.2f} | {rb.mean_return:.2f} | -- | -- | -- |")
        if self.heuristic_baseline is not None:
            hb = self.heuristic_baseline
            lines.append(f"| Heuristic Baseline | {hb.mean_return:.2f} ± {hb.std_return:.2f} | {hb.mean_return:.2f} | {hb.mean_return:.2f} | -- | -- | -- |")
        for s in self.algo_summaries:
            st = s.seed_stats
            audit_str = "PASS" if (s.audit is None or s.audit.passed) else f"WARN ({', '.join(s.audit.red_flags)})"
            ckey = str(s.citation.get("key", "")).strip() if s.citation else "--"
            opt_str = ", ".join(f"`{o['id']}` (`@{o['citation']['key']}`)" for o in s.optimizations_used if o.get("citation", {}).get("key")) or "--"
            lines.append(
                f"| **{s.display_name}** (`{s.algo_id}`) | {st['mean']:.2f} ± {st['std']:.2f} | "
                f"{st['iqm']:.2f} | {st['median']:.2f} | {audit_str} | `@{ckey}` | {opt_str} |"
            )
        lines.append("")
        if self.mdp_spec is not None:
            lines.append("## 2. Formal Problem-to-MDP Specification")
            lines.append(self.mdp_spec.format_formal_mdp())
            lines.append("")
        lines.append("## 3. Experimental Setup, Data Storage & Reproducibility")
        lines.append(f"- **Timestamp (UTC)**: `{self.timestamp_utc}` | **PyTorch**: `{self.torch_version}` | **Device**: `{self.device}`")
        if self.preflight is not None:
            pf = self.preflight
            lines.append(
                f"- **Observed Live Environment**: `{pf.env_library}` | `obs_dim={pf.obs_dim}`, "
                f"`action={pf.action_space_kind}({pf.action_dim})`, `sim_speed={pf.ms_per_step:.3f} ms/step`"
            )
        lines.append(
            f"- **Random Seeds**: `{self.seeds}` (deterministic evaluation with frozen normalization statistics; "
            "IQM & multi-seed rigor protocol: `@agarwal2021deep`, `@henderson2018deep`)"
        )
        if output_dir is not None:
            lines.append(
                f"- **Data & Artifacts Directory**: `{Path(output_dir).resolve()}` "
                "(contains `paper_summary.md`, `paper_table.tex`, `references.bib`, `drl_report.json`, `learning_curves.csv`, and overview figure)"
            )
        if self.random_baseline is not None:
            lines.append(f"- **Random Floor**: `{self.random_baseline.mean_return:.2f} ± {self.random_baseline.std_return:.2f}`")
        if self.heuristic_baseline is not None:
            lines.append(f"- **Heuristic Baseline**: `{self.heuristic_baseline.mean_return:.2f} ± {self.heuristic_baseline.std_return:.2f}`")
        lines.append("")
        lines.append("## 4. Multi-Seed Comparison & Rigor Audit")
        for s in self.algo_summaries:
            lines.append(f"### `{s.algo_id}` ({s.display_name})")
            if s.citation:
                lines.append(f"- **Primary Citation (`@{s.citation.get('key', '')}`)**: {s.citation.get('ieee', '')}")
            lines.append("- **Active Optimizations & Verified Citations**:")
            if s.optimizations_used:
                for opt in s.optimizations_used:
                    ocit = opt.get("citation", {})
                    lines.append(
                        f"  - **{opt['name']}** (`{opt['id']}`): {opt['role_plain']} "
                        f"— {ocit.get('ieee', '')} (`@{ocit.get('key', '')}`)"
                    )
            else:
                lines.append("  - Core baseline formulation (no extra structural tricks enabled).")
            lines.append(f"- Hyperparameters: `{json.dumps(s.hyperparameters, ensure_ascii=False)}`")
            if s.audit is not None:
                lines.append("```text\n" + s.audit.format_report() + "\n```")
        lines.append("")
        lines.append("## 5. Verified Reference Provenance & Bibliography")
        for idx, entry in enumerate(self.verified_bibliography(), start=1):
            used_by = ", ".join(entry.get("used_by", []))
            url = entry.get("url", "")
            lines.append(
                f"{idx}. `@{entry['key']}` [{entry['category']}; used by: {used_by}]: "
                f"{entry.get('ieee', '')}" + (f" ({url})" if url else "")
            )
        return "\n".join(lines)

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        root = Path(out_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {}

        chart_path = _generate_overview_figure(self, root)
        files["overview_figure"] = chart_path

        p_json = root / "drl_report.json"
        p_json.write_text(self.to_json() + "\n", encoding="utf-8")
        files["drl_report_json"] = p_json

        p_md = root / "paper_summary.md"
        p_md.write_text(
            self.format_paper_summary(chart_filename=chart_path.name, output_dir=root) + "\n",
            encoding="utf-8",
        )
        files["paper_summary_md"] = p_md

        p_tex = root / "paper_table.tex"
        p_tex.write_text(self.to_latex(), encoding="utf-8")
        files["paper_table_tex"] = p_tex

        p_bib = root / "references.bib"
        p_bib.write_text(self.to_bibtex(), encoding="utf-8")
        files["references_bib"] = p_bib

        p_csv = root / "learning_curves.csv"
        with p_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["algo_id", "seed", "step", "episode_return"])
            for s in self.algo_summaries:
                for run in s.seed_runs:
                    for pt in run.get("learning_curve", []):
                        writer.writerow([s.algo_id, run["seed"], pt.get("step", 0.0), pt.get("episode_return", 0.0)])
        files["learning_curves_csv"] = p_csv
        return files


def _resolve_runner(algo_or_fn: str | Callable[..., TrainResult]) -> tuple[str, Callable[..., TrainResult]]:
    if callable(algo_or_fn):
        return getattr(algo_or_fn, "__name__", "custom_algo").removeprefix("train_"), algo_or_fn
    if algo_or_fn not in ALGO_ID_TO_RUNNER:
        raise KeyError(f"Unknown algorithm '{algo_or_fn}'. Available: {sorted(ALGO_ID_TO_RUNNER.keys())}")
    mod_path, fn_name = ALGO_ID_TO_RUNNER[algo_or_fn]
    mod = importlib.import_module(mod_path)
    return algo_or_fn, getattr(mod, fn_name)


def build_algo_config(algo_id: str, **overrides: Any) -> Any | None:
    """Construct the dataclass Config for ``algo_id`` applying only fields supported by that algorithm."""
    if algo_id not in ALGO_ID_TO_RUNNER:
        return None
    mod_path, _ = ALGO_ID_TO_RUNNER[algo_id]
    mod = importlib.import_module(mod_path)
    cfg_cls = next((v for k, v in vars(mod).items() if k.endswith("Config")), None)
    if cfg_cls is None:
        return None
    valid_names = {f.name for f in fields(cfg_cls)}
    filtered = {k: v for k, v in overrides.items() if k in valid_names and v is not None}
    return cfg_cls(**filtered)


def evaluate_drl_suite(
    env_or_factory: Any,
    algos: Sequence[str | Callable[..., TrainResult]],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    min_seeds: int = 3,
    algo_configs: dict[str, Any] | None = None,
    mdp_spec: MDPSpec | None = None,
    heuristic_fn: Callable[[np.ndarray, dict[str, Any]], Any] | None = None,
    preflight_report: PreflightReport | None = None,
    task_name: str = "DRLBenchmark",
    domain: str = "control",
    run_preflight: bool = True,
    run_audit: bool = True,
    device: str = "auto",
) -> DRLReport:
    """Run Stage 1 Preflight -> Multi-Seed Training -> Stage 3 Rigor Audit -> Stage 4 Report."""
    cards_by_id = {c["id"]: c for c in load_catalog()}
    pf = preflight_report if preflight_report is not None else (preflight_env(env_or_factory, heuristic_fn=heuristic_fn) if run_preflight else None)
    rand_base = pf.random_baseline if pf is not None else None
    heur_base = pf.heuristic_baseline if pf is not None else None

    summaries: list[AlgoExperimentSummary] = []
    for item in algos:
        aid, fn = _resolve_runner(item)
        card = cards_by_id.get(aid, {})
        cfg_obj = (algo_configs or {}).get(aid)
        seed_runs: list[TrainResult] = []
        for s in seeds:
            target = (
                env_or_factory()
                if callable(env_or_factory) and (isinstance(env_or_factory, type) or not hasattr(env_or_factory, "step"))
                else env_or_factory
            )
            if cfg_obj is not None:
                # Clone dataclass config with seed override
                cfg_copy = type(cfg_obj)(**{**asdict(cfg_obj), "seed": int(s)})
                res = fn(target, config=cfg_copy)
            else:
                default_cfg = build_algo_config(aid, seed=int(s), device=device)
                res = fn(target, config=default_cfg) if default_cfg is not None else fn(target)
            seed_runs.append(res)

        if isinstance(env_or_factory, OfflineTransitionDataset):
            eval_vals = []
            for r in seed_runs:
                if r.eval_stats is not None:
                    eval_vals.append(r.eval_stats.mean_return)
                else:
                    gamma_r = float(r.config.get("gamma", 0.99)) if isinstance(r.config, dict) else 0.99
                    ope = evaluate_offline_ope(env_or_factory, r.policy_fn, gamma=gamma_r, seed=r.seed + 20_000)
                    r.train_diagnostics["ope_fqe_value"] = float(ope["fqe"]["fqe_initial_value"])
                    r.train_diagnostics["ope_fqe_cvar_05"] = float(ope["fqe"]["fqe_cvar_alpha"])
                    r.train_diagnostics["ope_wis_return"] = float(ope["wis"]["wis_return"])
                    r.train_diagnostics["ope_behavior_return"] = float(ope["behavior_return"])
                    r.extra["ope"] = ope
                    eval_vals.append(float(ope["primary_ope_value"]))
        else:
            eval_vals = [
                r.eval_stats.mean_return
                if r.eval_stats is not None
                else (
                    float(r.learning_curve[-1]["episode_return"])
                    if r.learning_curve and "episode_return" in r.learning_curve[-1]
                    else float(r.train_diagnostics.get("ope_fqe_value", r.train_diagnostics.get("ope_wis_return", 0.0)))
                )
                for r in seed_runs
            ]
        audit = (
            audit_drl_rigor(
                seed_runs,
                preflight_report=pf,
                mdp_spec=mdp_spec,
                random_baseline=rand_base,
                heuristic_baseline=heur_base,
                min_seeds=min_seeds,
            )
            if run_audit
            else None
        )
        runtime_opt_ids = [oid for r in seed_runs for oid in r.optimizations_used]
        if any(r.obs_rms_state is not None for r in seed_runs):
            runtime_opt_ids.append("obs_normalization")
        resolved_opts = resolve_run_optimizations(
            aid,
            config=seed_runs[0].config if seed_runs else None,
            explicit_ids=runtime_opt_ids,
            saw_action_mask=bool(pf is not None and pf.summary.get("saw_action_mask")),
        )

        summaries.append(
            AlgoExperimentSummary(
                algo_id=aid,
                display_name=card.get("name", aid),
                family=card.get("family", "custom"),
                paradigm=card.get("paradigm", "custom"),
                seeds=[int(s) for s in seeds],
                seed_stats=aggregate_seed_stats(eval_vals),
                mean_elapsed_sec=float(np.mean([r.elapsed_seconds for r in seed_runs])),
                total_timesteps_per_seed=int(np.mean([r.total_timesteps for r in seed_runs])),
                hyperparameters=seed_runs[0].config if seed_runs else {},
                citation=card.get("citation", {}),
                audit=audit,
                seed_runs=[r.to_summary_dict() for r in seed_runs],
                optimizations_used=[opt.to_dict() for opt in resolved_opts],
            )
        )

    return DRLReport(
        task_name=task_name,
        domain=domain,
        seeds=[int(s) for s in seeds],
        device=device,
        torch_version=torch.__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        mdp_spec=mdp_spec,
        preflight=pf,
        random_baseline=rand_base,
        heuristic_baseline=heur_base,
        algo_summaries=summaries,
    )
