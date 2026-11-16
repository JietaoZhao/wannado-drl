"""Build and query the wannado.drl algorithm taxonomy and citation registry from single-file META headers."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from wannado.drl.core.provenance import ALGO_DEFAULT_OPTIMIZATIONS, OPTIMIZATION_PROVENANCE
from wannado.drl.core.types import AlgoMeta

DRL_ROOT = Path(__file__).resolve().parent.parent
ALGOS_DIR = DRL_ROOT / "algos"
REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


def discover_algo_metas() -> list[dict[str, Any]]:
    """Scan all CleanRL-style algorithm files and extract verified ``META`` cards."""
    cards: list[dict[str, Any]] = []
    for file_path in sorted(ALGOS_DIR.rglob("*.py")):
        if file_path.name.startswith("_"):
            continue
        rel = file_path.relative_to(DRL_ROOT).with_suffix("")
        module_name = "wannado.drl." + ".".join(rel.parts)
        module = importlib.import_module(module_name)
        meta = getattr(module, "META", None)
        if not isinstance(meta, AlgoMeta):
            raise TypeError(f"Algorithm module '{module_name}' must define META as an AlgoMeta instance.")
        card = meta.to_dict()
        if not card.get("module_path"):
            card["module_path"] = module_name
        if not card.get("default_optimizations"):
            card["default_optimizations"] = list(ALGO_DEFAULT_OPTIMIZATIONS.get(card["id"], ()))
        card["optimization_citations"] = [
            OPTIMIZATION_PROVENANCE[oid].to_dict()
            for oid in card["default_optimizations"]
            if oid in OPTIMIZATION_PROVENANCE
        ]
        cards.append(card)

    ids = {c["id"] for c in cards}
    for c in cards:
        pid = c.get("parent_id")
        if pid is not None and pid not in ids:
            raise ValueError(f"Algorithm '{c['id']}' declares unknown parent_id='{pid}'.")

    cards.sort(key=lambda c: (c["family"], c["citation"]["year"], c["id"]))
    return cards


def build_registry_json(output_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Generate ``registry.json`` directly from source module metadata."""
    cards = discover_algo_metas()
    payload = {
        "schema_version": "1.0",
        "count": len(cards),
        "algorithms": cards,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_catalog(rebuild_if_missing: bool = True) -> list[dict[str, Any]]:
    """Load the serialized algorithm knowledge graph."""
    if REGISTRY_PATH.exists():
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return list(data.get("algorithms", []))
    if rebuild_if_missing:
        return discover_algo_metas()
    raise FileNotFoundError(f"Catalog registry not found at {REGISTRY_PATH}.")


def query_catalog(
    *,
    family: str | None = None,
    paradigm: str | None = None,
    tier: str | None = None,
    on_policy: bool | None = None,
    action_space: str | None = None,
    env_modality: str | None = None,
    supports_action_mask: bool | None = None,
    supports_recurrent: bool | None = None,
) -> list[dict[str, Any]]:
    """Filter algorithm metadata cards by taxonomy, paradigm, action space, or modality."""
    cards = load_catalog()
    results: list[dict[str, Any]] = []
    for card in cards:
        if family is not None and card["family"] != family:
            continue
        if paradigm is not None and card["paradigm"] != paradigm:
            continue
        if tier is not None and card["tier"] != tier:
            continue
        if on_policy is not None and card["on_policy"] != on_policy:
            continue
        if supports_action_mask is not None and card["supports_action_mask"] != supports_action_mask:
            continue
        if supports_recurrent is not None and card["supports_recurrent"] != supports_recurrent:
            continue
        if action_space is not None and action_space not in card["action_spaces"]:
            continue
        if env_modality is not None and env_modality not in card["env_modalities"]:
            continue
        results.append(card)
    return results


def get_taxonomy_tree() -> dict[str, list[dict[str, Any]]]:
    """Group all algorithms cleanly by family in pedagogical-to-SOTA progression."""
    tree: dict[str, list[dict[str, Any]]] = {}
    for card in load_catalog():
        tree.setdefault(card["family"], []).append(card)
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the wannado.drl algorithm catalog.")
    parser.add_argument("--build", action="store_true", help="Rebuild registry.json from algorithm modules.")
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--paradigm", type=str, default=None)
    parser.add_argument("--tier", type=str, default=None)
    parser.add_argument("--action-space", type=str, default=None)
    args = parser.parse_args()

    if args.build or not REGISTRY_PATH.exists():
        payload = build_registry_json()
        print(f"Built {REGISTRY_PATH} with {payload['count']} algorithms.")

    matches = query_catalog(
        family=args.family,
        paradigm=args.paradigm,
        tier=args.tier,
        action_space=args.action_space,
    )
    print(json.dumps(matches, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
