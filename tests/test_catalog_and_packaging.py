"""Contract tests for PEP 420 namespace isolation, 27-algorithm catalog, and zero-hallucination citations."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from wannado.drl import (
    ALGO_DEFAULT_OPTIMIZATIONS,
    OPTIMIZATION_PROVENANCE,
    build_registry_json,
    export_citations,
    get_sota_lineage,
    get_taxonomy_tree,
    load_catalog,
    query_catalog,
    resolve_run_optimizations,
)
from wannado.drl import (
    __version__ as drl_version,
)
from wannado.drl.catalog.build_catalog import REGISTRY_PATH, discover_algo_metas

REPO_ROOT = Path(__file__).resolve().parent.parent
SIBLING_ADV_SRC = REPO_ROOT.parent / "wannado-adv" / "src"


class TestPackagingAndCatalogContracts(unittest.TestCase):
    def test_pep420_namespace_has_no_top_init(self) -> None:
        forbidden = REPO_ROOT / "src" / "wannado" / "__init__.py"
        self.assertFalse(forbidden.exists(), "src/wannado/__init__.py must NOT exist (PEP 420).")

    def test_pep420_coexists_with_wannado_adv(self) -> None:
        if not SIBLING_ADV_SRC.exists():
            self.skipTest("Sibling wannado-adv not present.")
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{SIBLING_ADV_SRC}"
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import wannado.drl, wannado.adv; print(wannado.drl.__version__, wannado.adv.__version__)",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        parts = proc.stdout.strip().split()
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], drl_version)
        self.assertTrue(len(parts[1]) > 0)

    def test_agent_kit_coexists_with_wannado_adv_without_prompt_collisions(self) -> None:
        if not SIBLING_ADV_SRC.exists():
            self.skipTest("Sibling wannado-adv not present.")
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{SIBLING_ADV_SRC}"
            script = (
                "from pathlib import Path\n"
                "from wannado.drl.agent_kit import install_agent_kit as install_drl\n"
                "from wannado.adv.agent_kit import install_agent_kit as install_adv\n"
                f"root = Path({tmp!r})\n"
                "(root / 'AGENTS.md').write_text('# Project Custom Rules\\n- Stay deterministic.\\n', encoding='utf-8')\n"
                "install_drl(root, ide='all')\n"
                "install_adv(root, ide='all')\n"
                "install_drl(root, ide='all')\n"
                "install_adv(root, ide='all')\n"
            )
            subprocess.run([sys.executable, "-c", script], env=env, check=True)

            root = Path(tmp)
            agents_text = (root / "AGENTS.md").read_text(encoding="utf-8")
            claude_text = (root / "CLAUDE.md").read_text(encoding="utf-8")
            drl_skill = (root / ".agents" / "skills" / "wannado-drl" / "SKILL.md").read_text(encoding="utf-8")
            adv_skill = (root / ".agents" / "skills" / "wannado-adv" / "SKILL.md").read_text(encoding="utf-8")

            self.assertIn("Stay deterministic.", agents_text)
            for marker in (
                "<!-- wannado-drl:start -->",
                "<!-- wannado-drl:end -->",
                "<!-- wannado-adv:start -->",
                "<!-- wannado-adv:end -->",
            ):
                self.assertEqual(agents_text.count(marker), 1, marker)
                self.assertEqual(claude_text.count(marker), 1, marker)

            self.assertTrue((root / ".cursor" / "rules" / "wannado-drl.mdc").exists())
            self.assertTrue((root / ".cursor" / "rules" / "wannado-adv.mdc").exists())
            self.assertIn("obs_rms_state", agents_text)
            self.assertIn("./adv_eval/", drl_skill)
            self.assertIn("./drl_lab/", adv_skill)

    def test_all_27_algorithms_across_7_families_in_catalog(self) -> None:
        discovered = discover_algo_metas()
        self.assertEqual(len(discovered), 27)
        tree = get_taxonomy_tree()
        expected_families = {
            "tabular_classic",
            "value_based",
            "policy_gradient",
            "actor_critic_off",
            "offline_rl",
            "multi_agent",
            "llm_and_discovered",
        }
        self.assertEqual(set(tree.keys()), expected_families)

        # Ensure serialized registry.json is synchronized with live META discovery
        if not REGISTRY_PATH.exists():
            build_registry_json()
        loaded = load_catalog()
        self.assertEqual([c["id"] for c in discovered], [c["id"] for c in loaded])

    def test_verified_citations_and_acyclic_lineage(self) -> None:
        cards = load_catalog()
        ids = {c["id"] for c in cards}
        for c in cards:
            cit = c["citation"]
            self.assertTrue(cit["bibtex"].startswith("@"))
            self.assertTrue(cit["url"].startswith("http"))
            self.assertGreaterEqual(cit["year"], 1992)
            self.assertTrue(cit["ieee"])
            self.assertTrue(cit["apa"])
            if c["parent_id"] is not None:
                self.assertIn(c["parent_id"], ids)

        # Check specific requested frontier & classic entries exist and cite accurately
        for required_id in ("monte_carlo", "q_learning", "ppo", "sac", "hil_serl", "qmix", "mappo", "grpo", "dapo", "gspo", "arpo", "discorl"):
            self.assertIn(required_id, ids)

        lineage = get_sota_lineage()
        self.assertEqual(len(lineage), 27)
        bib = export_citations(["grpo", "dapo", "gspo", "arpo", "discorl", "hil_serl"], fmt="bibtex", include_audit_refs=True)
        self.assertIn("shao2024deepseekmath", bib)
        self.assertIn("yu2025dapo", bib)
        self.assertIn("zheng2025group", bib)
        self.assertIn("dong2025arpo", bib)
        self.assertIn("oh2025discovering", bib)
        self.assertIn("luo2025hilserl", bib)
        self.assertIn("agarwal2021deep", bib)
        self.assertIn("henderson2018deep", bib)

        with self.assertRaises(KeyError):
            export_citations(["nonexistent_fake_algo"])

    def test_query_catalog_filters(self) -> None:
        marl = query_catalog(family="multi_agent")
        self.assertEqual({c["id"] for c in marl}, {"vdn", "qmix", "mappo", "maddpg"})
        masked = query_catalog(supports_action_mask=True)
        self.assertIn("ppo", {c["id"] for c in masked})
        self.assertIn("dqn", {c["id"] for c in masked})

    def test_optimization_provenance_registry_and_resolution(self) -> None:
        for aid, opt_ids in ALGO_DEFAULT_OPTIMIZATIONS.items():
            for oid in opt_ids:
                self.assertIn(oid, OPTIMIZATION_PROVENANCE, f"Unknown opt '{oid}' mapped for '{aid}'.")

        for oid, prov in OPTIMIZATION_PROVENANCE.items():
            self.assertTrue(prov.citation.bibtex.startswith("@"), oid)
            self.assertTrue(prov.citation.url.startswith("http"), oid)
            self.assertTrue(prov.citation.ieee, oid)
            self.assertTrue(prov.citation.apa, oid)
            self.assertGreaterEqual(prov.citation.year, 2014, oid)

        for card in load_catalog():
            self.assertEqual(
                [c["id"] for c in card.get("optimization_citations", [])],
                list(card.get("default_optimizations", [])),
            )

        opts = resolve_run_optimizations("ppo", config={"normalize_obs": True}, saw_action_mask=True)
        self.assertEqual([o.id for o in opts], ["gae", "orthogonal_init_and_clip", "obs_normalization", "action_masking"])

        bib = export_citations(
            ["ppo"],
            fmt="bibtex",
            optimization_ids=[o.id for o in opts],
            include_audit_refs=True,
        )
        for key in (
            "schulman2017proximal",
            "schulman2016high",
            "engstrom2020implementation",
            "huang2022closer",
            "agarwal2021deep",
            "henderson2018deep",
        ):
            self.assertIn(key, bib)
        # obs_normalization & orthogonal_init_and_clip share engstrom2020implementation -> must be deduplicated
        self.assertEqual(bib.count("@inproceedings{engstrom2020implementation"), 1)


if __name__ == "__main__":
    unittest.main()
