"""Command-line interface for the wannado-drl AI-Native DRL platform."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

from wannado.drl.agent_kit import install_agent_kit
from wannado.drl.audit import preflight_env
from wannado.drl.catalog.build_catalog import build_registry_json, query_catalog
from wannado.drl.copilot import (
    build_arsenal_showcase,
    build_interactive_brief,
    explain_parameters,
    format_arsenal_overview,
    get_sota_lineage,
    recommend_algo_suite,
    scan_project,
)
from wannado.drl.mdp import audit_mdp_spec, build_mdp_archetype
from wannado.drl.reporter import export_citations
from wannado.drl.workspace import eject_algo, scaffold_lab


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wannado-drl",
        description="AI-Native CleanRL-style DRL platform with Stage 0 MDP co-design, multi-lib adapters, and anti-fake-convergence auditing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. inspect
    p_inspect = sub.add_parser("inspect", help="Scan a project directory, infer RL/simulator mode, and surface Stage 0 MDP brief.")
    p_inspect.add_argument("project_dir", nargs="?", default=".")
    p_inspect.add_argument("--goal", default="standard", choices=["learning", "sanity", "standard", "paper_sota"])
    p_inspect.add_argument("--json", action="store_true", dest="as_json")

    # 2. intake
    p_intake = sub.add_parser("intake", help="Generate & audit a Stage 0 MDP Spec (mdp_spec.json) from a domain archetype.")
    p_intake.add_argument(
        "--archetype",
        default="custom_sim",
        choices=[
            "custom_sim",
            "wireless_scheduling",
            "continuous_control",
            "multi_agent_coop",
            "reasoning_sequence",
            "offline_control",
        ],
    )
    p_intake.add_argument("--name", default=None, help="Optional problem name.")
    p_intake.add_argument("--out", default="./drl_lab/mdp_spec.json", help="Output path for mdp_spec.json.")
    p_intake.add_argument("--json", action="store_true", dest="as_json")

    # 3. catalog
    p_cat = sub.add_parser("catalog", help="Search the 27-algorithm DRL metadata catalog across 7 families.")
    p_cat.add_argument("--family", default=None)
    p_cat.add_argument("--paradigm", default=None)
    p_cat.add_argument("--tier", default=None)
    p_cat.add_argument("--action-space", default=None)
    p_cat.add_argument("--rebuild", action="store_true")
    p_cat.add_argument("--json", action="store_true", dest="as_json")

    # 4. lineage
    p_lin = sub.add_parser("lineage", help="Inspect algorithm evolution chains from tabular classics to frontier RL.")
    p_lin.add_argument("--family", default=None)
    p_lin.add_argument("--action-space", default=None)
    p_lin.add_argument("--json", action="store_true", dest="as_json")

    # 5. recommend
    p_rec = sub.add_parser("recommend", help="Recommend an algorithm suite and novice parameter guide.")
    p_rec.add_argument("--action-space", default="discrete")
    p_rec.add_argument("--goal", default="standard", choices=["learning", "sanity", "standard", "paper_sota"])
    p_rec.add_argument("--partial-obs", action="store_true")
    p_rec.add_argument("--action-mask", action="store_true")
    p_rec.add_argument("--offline", action="store_true")
    p_rec.add_argument("--human-demos", action="store_true")
    p_rec.add_argument("--num-agents", type=int, default=1)
    p_rec.add_argument("--json", action="store_true", dest="as_json")

    # 6. cite
    p_cite = sub.add_parser("cite", help="Export verified BibTeX/IEEE/APA citations for algorithms and optimizations.")
    p_cite.add_argument("algo_ids", nargs="+")
    p_cite.add_argument("--format", default="bibtex", choices=["bibtex", "ieee", "apa"])
    p_cite.add_argument("--include-audit-refs", action="store_true")
    p_cite.add_argument("--opt", nargs="*", default=None, help="Explicit optimization IDs (from OPTIMIZATION_PROVENANCE).")
    p_cite.add_argument("--no-optimizations", action="store_true", help="Export primary algorithm citations only.")

    # 7. eject
    p_eject = sub.add_parser("eject", help="Copy ONE CleanRL-style single-file algorithm into drl_lab/algos/ for custom edits.")
    p_eject.add_argument("algo_id")
    p_eject.add_argument("--dest", default="./drl_lab/algos")
    p_eject.add_argument("--name", default=None)
    p_eject.add_argument("--overwrite", action="store_true")

    # 8. scaffold
    p_scaf = sub.add_parser("scaffold", help="Generate non-intrusive external drl_lab/ plug-in harness.")
    p_scaf.add_argument("--dir", default="./drl_lab", dest="target_dir")
    p_scaf.add_argument("--archetype", default="custom_sim")
    p_scaf.add_argument("--mode", default="custom_sim")
    p_scaf.add_argument("--name", default=None)
    p_scaf.add_argument("--overwrite", action="store_true")

    # 9. preflight
    p_pf = sub.add_parser("preflight", help="Run Stage 1 MDP Preflight & Random/Heuristic Baselines on an interface.py.")
    p_pf.add_argument("--interface", default="./drl_lab/interface.py", dest="interface_path")
    p_pf.add_argument("--json", action="store_true", dest="as_json")

    # 10. init-agent
    p_init = sub.add_parser("init-agent", help="Install universal AGENTS.md / Agent Skills rules + IDE adapters.")
    p_init.add_argument("--dir", default=".", dest="project_dir")
    p_init.add_argument("--ide", default="all", choices=["all", "universal", "cursor", "antigravity", "claude", "codex"])

    # 11. arsenal
    p_ars = sub.add_parser("arsenal", help="Display versioned wannado-drl RL arsenal (Classics -> 2025 Frontier SOTA).")
    p_ars.add_argument("--json", action="store_true", dest="as_json")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "inspect":
        scan = scan_project(args.project_dir)
        brief = build_interactive_brief(project_scan=scan, goal=args.goal)
        if args.as_json:
            print(json.dumps(brief, indent=2, ensure_ascii=False))
        else:
            print(scan.format_summary())
            ars = brief["arsenal_showcase"]
            print(f"\n[wannado-drl v{ars['version']} Arsenal]")
            print(f"  {ars['headline']}")
            print(f"  Classic Foundations : {', '.join(ars['era_breakdown']['classic_foundations'])}")
            print(f"  Frontier SOTA (2025): {', '.join(ars['era_breakdown']['frontier_sota'])}")

            dlg = brief["goal_confirmation_dialog"]
            print("\n[Goal Confirmation Dialog]")
            print(f"  AI Understanding : {dlg['ai_understanding_summary']}")
            print(f"  Question         : {dlg['dialog_1_confirm_goal']['question']}")
            for idx, opt in enumerate(dlg["dialog_1_confirm_goal"]["options"], start=1):
                print(f"    [{idx}] {opt['label']}")
            print(f"  If [2] chosen    : {dlg['dialog_2_one_sentence_prompt']['prompt']}")

            rec = brief["recommended_suite"]
            print(f"\n[Recommended '{args.goal}' Suite ({rec['paradigm']}, action_space='{rec['action_space']}')]")
            print(f"  Run Lineup   : {', '.join(rec['algos'])}")
            print(f"  Classic Pool : {', '.join(rec['classic_baselines'])} | Frontier SOTA Pool: {', '.join(rec['frontier_sota'])}")
            print(f"  Rationale    : {rec['rationale']}")

            print("\n[Stage 0 MDP Intake Questions]")
            for q in brief["stage0_mdp_intake"]["mdp_questions"]:
                print(f"  - {q['prompt']}")
                print(f"    Plain idea: {q['plain_analogy']}")
                print(f"    Pitfall   : {q['pitfall_warning']}")

            print("\n[Stage 2 Zero-Jargon Settings & Data Storage Choices]")
            for q in brief["questions_to_ask_user"]:
                print(f"  - {q['parameter']} ({q.get('terminology', q['parameter'])})")
                print(f"    Plain analogy : {q['plain_analogy']}")
                if q.get("equivalent_to"):
                    print(f"    Equivalent to : {q['equivalent_to']}")
                print(f"    Trade-off     : {q['if_higher_vs_lower']}")
                for opt in q.get("options", []):
                    print(f"      * {opt.get('label', str(opt))}")

            eta = brief["runtime_estimate"]
            print(f"\n[Hardware ETA Forecast] device={eta['device']}, ~{eta['total_seconds']:.1f}s total — {eta['advisory']}")
        return 0

    if args.command == "arsenal":
        if args.as_json:
            print(json.dumps(build_arsenal_showcase(), indent=2, ensure_ascii=False))
        else:
            print(format_arsenal_overview())
        return 0

    if args.command == "intake":
        spec = build_mdp_archetype(args.archetype, problem_name=args.name)
        audit = audit_mdp_spec(spec)
        out_path = spec.save(args.out)
        payload = {"saved_to": str(out_path), "mdp_spec": spec.to_dict(), "audit": audit.to_dict()}
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(spec.format_formal_mdp())
            print(f"\nSaved MDP Spec to: {out_path} (audit ok={audit.ok})")
        return 0

    if args.command == "catalog":
        if args.rebuild:
            build_registry_json()
        cards = query_catalog(
            family=args.family,
            paradigm=args.paradigm,
            tier=args.tier,
            action_space=args.action_space,
        )
        if args.as_json:
            print(json.dumps(cards, indent=2, ensure_ascii=False))
        else:
            print(f"Matched {len(cards)} algorithms:")
            for c in cards:
                print(f"  [{c['tier']}][{c['family']}] {c['id']:<16s} ({c['paradigm']}) — {c['summary_plain']}")
        return 0

    if args.command == "lineage":
        entries = get_sota_lineage(family=args.family, action_space=args.action_space)
        if args.as_json:
            print(json.dumps(entries, indent=2, ensure_ascii=False))
        else:
            for e in entries:
                print(f"  [{e['year']}][{e['tier']}][{e['family']}] {e['id']:<15s} | {e['evolution_path']}")
                print(f"      -> {e['diff_from_parent']}")
        return 0

    if args.command == "recommend":
        suite = recommend_algo_suite(
            action_space=args.action_space,
            goal=args.goal,
            partial_observability=args.partial_obs,
            has_invalid_actions=args.action_mask,
            is_offline_only=args.offline,
            has_human_demos=args.human_demos,
            num_agents=args.num_agents,
        )
        payload = {"recommended_suite": suite.to_dict(), "parameter_guide": explain_parameters()}
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Suite ({suite.goal} | {suite.paradigm}): {suite.algos}")
            print(f"Rationale: {suite.rationale}")
        return 0

    if args.command == "cite":
        print(
            export_citations(
                args.algo_ids,
                fmt=args.format,
                include_audit_refs=args.include_audit_refs,
                include_optimizations=not args.no_optimizations,
                optimization_ids=args.opt,
            )
        )
        return 0

    if args.command == "eject":
        dest = eject_algo(args.algo_id, dest_dir=args.dest, custom_name=args.name, overwrite=args.overwrite)
        print(f"Ejected CleanRL single-file algorithm to: {dest}")
        return 0

    if args.command == "scaffold":
        written = scaffold_lab(
            target_dir=args.target_dir,
            archetype=args.archetype,
            mode=args.mode,
            problem_name=args.name,
            overwrite=args.overwrite,
        )
        print(f"Scaffolded isolated drl_lab workspace under {Path(args.target_dir).resolve()}:")
        for rel, path in written.items():
            print(f"  - {rel}: {path}")
        return 0

    if args.command == "preflight":
        iface_path = Path(args.interface_path).resolve()
        mod_spec = importlib.util.spec_from_file_location("_wannado_user_iface", iface_path)
        if mod_spec is None or mod_spec.loader is None:
            raise FileNotFoundError(f"Cannot load interface module at {iface_path}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        iface = mod.load_env_interface()
        pf = preflight_env(iface.env_or_factory, heuristic_fn=iface.heuristic_fn)
        if args.as_json:
            print(json.dumps(pf.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(pf.format_report())
        return 0 if pf.ok else 1

    if args.command == "init-agent":
        written = install_agent_kit(project_dir=args.project_dir, ide=args.ide)
        print(f"Installed wannado-drl AI Co-Pilot skill/rules for ide='{args.ide}':")
        for role, path in written.items():
            print(f"  - {role}: {path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
