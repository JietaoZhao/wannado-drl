"""Tests for AI Co-Pilot inspection, Stage 1 preflight, Stage 3 Anti-Fake-Convergence Auditor, scaffolding, ejection & CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wannado.drl import (
    EpisodeEvalStats,
    TrainResult,
    audit_drl_rigor,
    build_arsenal_showcase,
    build_interactive_brief,
    build_post_run_debrief,
    eject_algo,
    estimate_runtime,
    explain_parameters,
    format_arsenal_overview,
    install_agent_kit,
    preflight_env,
    recommend_algo_suite,
    scan_project,
    scaffold_lab,
)
from wannado.drl.cli import main as cli_main
from wannado.drl.reporter import evaluate_drl_suite


class _KPIRewardHackingEnv(gym.Env):
    """Toy env where agent can farm reward while degrading the true KPI."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        trunc = self.steps >= 5
        return np.zeros(2, dtype=np.float32), 0.1, False, trunc, {"task_metrics": {"success_rate": 0.5}}


class TestCopilotAuditAndCLI(unittest.TestCase):
    def test_scan_project_distinguishes_raw_sim_vs_existing_gym(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# 5G OFDMA Resource Allocator\nSchedules resource blocks across users.\n", encoding="utf-8")
            (root / "wireless_sim.py").write_text(
                "def step(action):\n    # simulate ofdm sinr cqi queue throughput\n    return None\n",
                encoding="utf-8",
            )
            scan = scan_project(root)
            self.assertEqual(scan.integration_mode, "raw_simulator_needs_mdp")
            self.assertEqual(scan.likely_archetype, "wireless_scheduling")
            self.assertEqual(scan.understanding_confidence, "high")
            self.assertIn("wireless", scan.project_summary_guess.lower())
            self.assertIn("wireless_sim.py", scan.simulator_candidates)

            brief = build_interactive_brief(project_scan=scan, goal="standard")
            self.assertIn("arsenal_showcase", brief)
            self.assertIn("goal_confirmation_dialog", brief)
            self.assertIn("stage0_mdp_intake", brief)
            self.assertIn("recommended_suite", brief)
            self.assertIn("questions_to_ask_user", brief)
            self.assertEqual(brief["arsenal_showcase"]["total_algorithms"], 27)
            self.assertGreaterEqual(len(brief["recommended_suite"]["classic_baselines"]), 1)
            self.assertGreaterEqual(len(brief["recommended_suite"]["frontier_sota"]), 1)

            dlg = brief["goal_confirmation_dialog"]
            self.assertTrue(dlg["must_explain_understanding_before_asking"])
            self.assertEqual(len(dlg["dialog_1_confirm_goal"]["options"]), 2)
            self.assertIn("one sentence", dlg["dialog_2_one_sentence_prompt"]["prompt"].lower())

            # Context-aware parameter guides (epsilon for DQN vs group_size for GRPO/DAPO)
            dqn_guide = explain_parameters(algos=["double_dqn"])
            self.assertIn("epsilon_exploration", dqn_guide["algorithm_specific"])
            self.assertIn("storage_and_outputs", dqn_guide)
            llm_guide = explain_parameters(algos=["grpo", "dapo"], action_space="autoregressive_sequence")
            self.assertIn("group_size_and_clip_high", llm_guide["algorithm_specific"])

            # Ambiguous empty/generic repo -> low confidence so AI explains partial scan before asking
            empty_root = root / "ambiguous_repo"
            empty_root.mkdir()
            (empty_root / "helper.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            low_scan = scan_project(empty_root)
            self.assertEqual(low_scan.understanding_confidence, "low")
            self.assertIn("Share what was observed first", low_scan.project_summary_guess)

            self.assertIn("27 Algorithms", format_arsenal_overview())
            self.assertEqual(build_arsenal_showcase()["total_families"], 7)

            # Continuous Gymnasium project -> confidence >= medium and continuous action suite
            pend_root = root / "pendulum_proj"
            pend_root.mkdir()
            (pend_root / "app.py").write_text(
                '"""Pendulum torque swingup with Gymnasium."""\nimport gymnasium as gym\nenv = gym.make("Pendulum-v1")\n',
                encoding="utf-8",
            )
            pend_scan = scan_project(pend_root)
            self.assertEqual(pend_scan.understanding_confidence, "medium")
            pend_brief = build_interactive_brief(project_scan=pend_scan)
            self.assertEqual(pend_brief["recommended_suite"]["action_space"], "continuous")
            self.assertEqual(pend_brief["questions_to_ask_user"][0]["terminology"], "Algorithm Portfolio (Classic Foundations -> Frontier SOTA)")

    def test_preflight_and_runtime_estimate(self) -> None:
        pf = preflight_env(lambda: gym.make("CartPole-v1"), heuristic_fn=lambda obs, _info: 1 if obs[2] > 0 else 0)
        self.assertTrue(pf.ok)
        self.assertGreater(pf.steps_per_sec, 100.0)
        self.assertIsNotNone(pf.random_baseline)
        self.assertIsNotNone(pf.heuristic_baseline)
        self.assertGreater(pf.heuristic_baseline.mean_return, pf.random_baseline.mean_return)

        eta = estimate_runtime(preflight_report=pf, algos=["ppo", "double_dqn", "q_learning"], total_timesteps=10_000, num_seeds=3)
        self.assertGreater(eta.total_seconds, 0.0)
        self.assertIn("ppo", eta.per_algo_seconds)
        self.assertLess(eta.per_algo_seconds["q_learning"], eta.per_algo_seconds["double_dqn"])

    def test_rigor_auditor_catches_all_fake_convergence_traps(self) -> None:
        rand_base = EpisodeEvalStats([1.0], [10], 1.0, 0.0, 10.0, task_metrics_mean={"success_rate": 0.50})
        heur_base = EpisodeEvalStats([5.0], [10], 5.0, 0.0, 10.0, task_metrics_mean={"success_rate": 0.80})

        # Trap 1: Single seed + failed to beat heuristic + reward hacking KPI divergence + unfrozen obs_rms + entropy collapse + negative EV
        bad_run = TrainResult(
            algo_id="ppo",
            seed=0,
            total_timesteps=5000,
            elapsed_seconds=1.0,
            policy_fn=lambda o, _: 0,
            model=None,
            config={},
            learning_curve=[{"step": 1000.0, "episode_return": 50.0}, {"step": 2000.0, "episode_return": 50.0}, {"step": 3000.0, "episode_return": 50.0}],
            train_diagnostics={"final_entropy": 1e-6, "mean_explained_variance": -1.2},
            eval_stats=EpisodeEvalStats(
                returns=[4.0],
                lengths=[5],
                mean_return=4.0,
                std_return=0.0,
                mean_length=5.0,
                task_metrics_mean={"success_rate": 0.10},  # Dropped from 0.50 -> 0.10 while return rose 1.0 -> 4.0!
            ),
            obs_rms_frozen_on_eval=False,
        )
        audit = audit_drl_rigor([bad_run], random_baseline=rand_base, heuristic_baseline=heur_base, min_seeds=3)
        self.assertFalse(audit.passed)
        for expected_flag in (
            "single_seed_survivorship_risk",
            "below_heuristic_baseline",
            "obs_rms_unfrozen_on_eval",
            "suspected_reward_hacking_kpi_divergence",
            "policy_entropy_collapse",
            "critic_negative_explained_variance",
            "train_eval_collapse_mismatch",
        ):
            self.assertIn(expected_flag, audit.red_flags)

        # Clean multi-seed runs should pass
        good_runs = [
            TrainResult(
                algo_id="ppo",
                seed=s,
                total_timesteps=5000,
                elapsed_seconds=1.0,
                policy_fn=lambda o, _: 0,
                model=None,
                config={},
                learning_curve=[{"step": 5000.0, "episode_return": 9.0}],
                train_diagnostics={"final_entropy": 0.4, "mean_explained_variance": 0.7},
                eval_stats=EpisodeEvalStats([9.0 + 0.1 * s], [10], 9.0 + 0.1 * s, 0.0, 10.0, task_metrics_mean={"success_rate": 0.95}),
                obs_rms_frozen_on_eval=True,
            )
            for s in (0, 1, 2)
        ]
        good_audit = audit_drl_rigor(good_runs, random_baseline=rand_base, heuristic_baseline=heur_base, min_seeds=3)
        self.assertTrue(good_audit.passed, f"Unexpected red flags: {good_audit.red_flags}")

        # Normalized [0, 1] reward train-vs-eval collapse (0.92 -> 0.00) must also trigger train_eval_collapse_mismatch
        small_scale_collapse = [
            TrainResult(
                algo_id="double_dqn",
                seed=s,
                total_timesteps=2000,
                elapsed_seconds=0.5,
                policy_fn=lambda o, _: 0,
                model=None,
                config={},
                learning_curve=[{"step": 500.0, "episode_return": 0.90}, {"step": 1000.0, "episode_return": 0.92}, {"step": 1500.0, "episode_return": 0.94}],
                eval_stats=EpisodeEvalStats([0.0], [100], 0.0, 0.0, 100.0),
                obs_rms_frozen_on_eval=True,
            )
            for s in (0, 1, 2)
        ]
        small_audit = audit_drl_rigor(small_scale_collapse, min_seeds=3)
        self.assertFalse(small_audit.passed)
        self.assertIn("train_eval_collapse_mismatch", small_audit.red_flags)

    def test_scaffold_eject_agent_kit_reporter_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lab_dir = root / "drl_lab"
            written = scaffold_lab(lab_dir, archetype="wireless_scheduling", mode="custom_sim")
            self.assertIn("interface.py", written)
            self.assertIn("mdp_spec.json", written)
            self.assertIn("run_train.py", written)

            ejected = eject_algo("ppo", dest_dir=lab_dir / "algos")
            self.assertTrue(ejected.exists())
            self.assertIn("[AI-Prompt Hook]", ejected.read_text(encoding="utf-8"))

            agent_files = install_agent_kit(root, ide="all")
            self.assertTrue((root / "AGENTS.md").exists())
            self.assertTrue((root / ".agents" / "skills" / "wannado-drl" / "SKILL.md").exists())
            self.assertTrue((root / ".cursor" / "rules" / "wannado-drl.mdc").exists())
            self.assertIn("claude_md", agent_files)

            # End-to-end report export test using CartPole-v1 with tiny step budget
            from wannado.drl import PPOConfig

            rep = evaluate_drl_suite(
                lambda: gym.make("CartPole-v1"),
                algos=["ppo"],
                seeds=[0, 1],
                algo_configs={"ppo": PPOConfig(total_timesteps=512, n_steps=128, eval_episodes=3)},
                task_name="CartPoleSmoke",
                domain="discrete_control",
            )
            saved = rep.save(root / "outputs")
            self.assertTrue(saved["overview_figure"].exists())
            self.assertTrue(saved["drl_report_json"].exists())
            self.assertTrue(saved["paper_summary_md"].exists())
            self.assertTrue(saved["paper_table_tex"].exists())
            self.assertTrue(saved["references_bib"].exists())
            self.assertTrue(saved["learning_curves_csv"].exists())

            bib_text = saved["references_bib"].read_text(encoding="utf-8")
            md_text = saved["paper_summary_md"].read_text(encoding="utf-8")
            tex_text = saved["paper_table_tex"].read_text(encoding="utf-8")
            report_payload = json.loads(saved["drl_report_json"].read_text(encoding="utf-8"))

            self.assertIn(f"![Multi-Seed Learning Curves & Evaluation Comparison]({saved['overview_figure'].name})", md_text)
            self.assertIn("| Method | Eval Mean ± Std |", md_text)
            self.assertIn("Data & Artifacts Directory", md_text)
            debrief = build_post_run_debrief(rep, output_dir=root / "outputs")
            self.assertTrue(debrief["plain_verdict"])
            self.assertGreaterEqual(debrief["verified_bibliography_count"], 5)

            for key in ("schulman2017proximal", "schulman2016high", "engstrom2020implementation", "agarwal2021deep", "henderson2018deep"):
                self.assertIn(key, bib_text)
                self.assertIn(f"`@{key}`", md_text)
            self.assertIn("Active Optimizations & Verified Citations", md_text)
            self.assertIn("Verified Reference Provenance & Bibliography", md_text)
            self.assertIn("Optimizations Cited", tex_text)
            self.assertIn("\\cite{schulman2017proximal}", tex_text)
            self.assertIn("\\cite{schulman2016high,engstrom2020implementation}", tex_text)
            self.assertIn("verified_bibliography", report_payload)
            self.assertEqual(
                [o["id"] for o in report_payload["algorithms"][0]["optimizations_used"]],
                ["gae", "orthogonal_init_and_clip"],
            )

            # End-to-end masked discrete env + ObsNorm provenance verification
            class _MaskedTinyEnv(gym.Env):
                def __init__(self) -> None:
                    super().__init__()
                    self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
                    self.action_space = spaces.Discrete(2)
                    self.t = 0

                def reset(self, *, seed: int | None = None, options: dict | None = None):
                    super().reset(seed=seed)
                    self.t = 0
                    return np.zeros(3, dtype=np.float32), {"action_mask": np.array([True, False], dtype=np.bool_)}

                def step(self, action: int):
                    self.t += 1
                    trunc = self.t >= 4
                    return (
                        np.full(3, float(action), dtype=np.float32),
                        1.0 if action == 0 else -1.0,
                        False,
                        trunc,
                        {"action_mask": np.array([True, False], dtype=np.bool_)},
                    )

            from wannado.drl import train_ppo

            # Bare env class (without parentheses) should auto-instantiate cleanly too
            ppo_masked = train_ppo(
                _MaskedTinyEnv,
                config=PPOConfig(total_timesteps=64, n_steps=32, eval_episodes=2, normalize_obs=True, seed=0),
            )
            self.assertEqual(
                ppo_masked.optimizations_used,
                ["gae", "orthogonal_init_and_clip", "obs_normalization", "action_masking"],
            )
            masked_rep = evaluate_drl_suite(
                _MaskedTinyEnv,
                algos=["ppo"],
                seeds=[0],
                algo_configs={"ppo": PPOConfig(total_timesteps=64, n_steps=32, eval_episodes=2, normalize_obs=True)},
                task_name="MaskedSmoke",
            )
            self.assertIn("huang2022closer", masked_rep.to_bibtex())
            self.assertIn("huang2022closer", masked_rep.to_latex())

            # CLI preflight, arsenal & inspect smoke checks
            self.assertEqual(cli_main(["arsenal"]), 0)
            self.assertEqual(cli_main(["inspect", str(root)]), 0)
            self.assertEqual(cli_main(["preflight", "--interface", str(lab_dir / "interface.py")]), 0)
            self.assertEqual(cli_main(["cite", "grpo", "dapo", "gspo", "arpo", "discorl", "hil_serl"]), 0)
            rec = recommend_algo_suite(action_space="autoregressive_sequence", goal="paper_sota")
            self.assertEqual(rec.algos, ["grpo", "dapo", "gspo", "arpo"])

            # Standalone generated run_train.py smoke test for continuous_control (with --timesteps override + Random Baseline preserved!)
            cont_lab = root / "cont_lab"
            scaffold_lab(cont_lab, archetype="continuous_control", mode="custom_sim")
            gym_lab = root / "gym_alias_lab"
            scaffold_lab(gym_lab, archetype="custom_sim", mode="gymnasium")
            self.assertIn("make_env()", (gym_lab / "interface.py").read_text(encoding="utf-8"))
            proc = subprocess.run(
                [sys.executable, str(cont_lab / "run_train.py"), "--algos", "ppo", "--seeds", "0", "--timesteps", "512"],
                capture_output=True,
                text=True,
                check=True,
            )
            tex_cont = (cont_lab / "outputs" / "paper_table.tex").read_text(encoding="utf-8")
            self.assertIn("Random Baseline", tex_cont, proc.stderr)


if __name__ == "__main__":
    unittest.main()
