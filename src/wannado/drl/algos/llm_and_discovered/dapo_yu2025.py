"""DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization (Yu et al., 2025).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : llm_and_discovered / S (Scaled Open-Source Reasoning RL)
- Parent        : `grpo` (Shao et al., 2024)
- Key Diff      : Fixes four failure modes of GRPO in long-CoT reasoning:
                  1. Clip-Higher: asymmetric [1 - eps_low, 1 + eps_high] prevents entropy collapse.
                  2. Dynamic Sampling: discards zero-variance prompt groups (all right/all wrong).
                  3. Token-Level Policy Gradient Loss: averages over total valid tokens, not per-sequence.
                  4. Overlong Reward Shaping: soft linear length penalty near max_response_len.
- Extension Hook [AI-Prompt Hook]:
    1. Replace token-level ratio with length-normalized sequence ratio -> `gspo`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.drl.core.networks import AutoregressiveSeqPolicy
from wannado.drl.core.provenance import resolve_run_optimizations
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, resolve_device, set_global_seed
from wannado.drl.core.wrappers import SequenceTaskEnv
from wannado.drl.mdp.baselines import evaluate_sequence_task

META = AlgoMeta(
    id="dapo",
    name="Decoupled Clip & Dynamic Sampling PO (DAPO)",
    family="llm_and_discovered",
    paradigm="group_relative",
    tier="S",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("autoregressive_sequence",),
    env_modalities=("reasoning_sequence", "llm_rl"),
    supports_action_mask=False,
    supports_recurrent=True,
    parent_id="grpo",
    diff_from_parent="Adds asymmetric Clip-Higher, Dynamic Sampling filtering, token-level loss normalization, and overlong reward shaping.",
    summary_plain="Upgrades GRPO to stop entropy collapse and gradient starvation on reasoning sequence tasks.",
    when_to_use="Reasoning sequence RL where GRPO suffers premature entropy collapse or sparse/unbalanced prompt difficulties.",
    when_not_to_use="MoE or ultra-long sequence tasks where token-level ratios destabilize (`gspo`), or dense control (`ppo`).",
    hyperparam_intuition={
        "clip_low": "Lower clip bound preventing excessive probability suppression.",
        "clip_high": "Higher upper clip bound (0.28 > 0.2) letting low-probability creative tokens rise.",
        "dynamic_sampling": "Filters out prompt groups with std(reward)==0 so every update carries learning signal.",
    },
    extension_hooks=(
        "Adjust asymmetric `clip_high` vs `clip_low` ratio to tune exploration vs convergence speed",
        "Switch token ratio to geometric sequence ratio -> `gspo`",
    ),
    module_path="wannado.drl.algos.llm_and_discovered.dapo_yu2025",
    citation=AlgoCitation(
        key="yu2025dapo",
        title="DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
        authors="Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, et al.",
        venue="Advances in Neural Information Processing Systems (NeurIPS) / arXiv:2503.14476",
        year=2025,
        url="https://arxiv.org/abs/2503.14476",
        bibtex=(
            "@article{yu2025dapo,\n"
            "  title={DAPO: An Open-Source LLM Reinforcement Learning System at Scale},\n"
            "  author={Yu, Qiying and Zhang, Zheng and Zhu, Ruofei and Yuan, Yufeng and Zuo, Xiaochen and Yue, Yu and others},\n"
            "  journal={arXiv preprint arXiv:2503.14476},\n"
            "  year={2025},\n"
            "  url={https://arxiv.org/abs/2503.14476}\n"
            "}"
        ),
        ieee='Q. Yu et al., "DAPO: An open-source LLM reinforcement learning system at scale," arXiv:2503.14476, 2025.',
        apa="Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., et al. (2025). DAPO: An open-source LLM reinforcement learning system at scale. arXiv:2503.14476.",
    ),
)


@dataclass
class DAPOConfig:
    total_updates: int = 80
    prompts_per_batch: int = 8
    group_size: int = 6
    update_epochs: int = 2
    lr: float = 2e-3
    clip_low: float = 0.20
    clip_high: float = 0.28
    dynamic_sampling: bool = True
    max_resample_rounds: int = 3
    soft_overlong_cache: int = 2
    overlong_penalty: float = 0.1
    temperature: float = 1.0
    embed_dim: int = 48
    hidden_dim: int = 64
    eval_prompts: int = 24
    device: str = "auto"
    seed: int = 42


def _apply_overlong_shaping(raw_rew: float, seq_len: int, max_len: int, cache_len: int, penalty: float) -> float:
    soft_start = max(1, max_len - cache_len)
    if seq_len <= soft_start:
        return raw_rew
    excess = min(1.0, float(seq_len - soft_start) / max(1, cache_len))
    return raw_rew - penalty * excess


def train_dapo(
    task: SequenceTaskEnv,
    config: DAPOConfig | None = None,
    *,
    policy: AutoregressiveSeqPolicy | None = None,
) -> TrainResult:
    """Train DAPO (Clip-Higher + Dynamic Sampling + Token-Level Loss + Overlong Shaping)."""
    cfg = config or DAPOConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    max_seq_len = max(128, int(task.prompt_len) + int(task.max_response_len) + 8)
    model = policy if policy is not None else AutoregressiveSeqPolicy(
        task.vocab_size, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim, max_seq_len=max_seq_len
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    t0 = time.perf_counter()
    learning_curve: list[dict[str, float]] = []
    ent_hist: list[float] = []
    kept_fraction_hist: list[float] = []
    total_rollouts = 0

    for step in range(1, cfg.total_updates + 1):
        kept_prompts: list[Any] = []
        kept_resps: list[Any] = []
        kept_old_lps: list[Any] = []
        kept_advs: list[Any] = []
        all_raw_rews: list[float] = []
        attempts = 0
        while len(kept_prompts) < cfg.prompts_per_batch and attempts < cfg.max_resample_rounds:
            attempts += 1
            prompts_np = task.sample_prompts(cfg.prompts_per_batch, rng)
            prompts_rep = np.repeat(prompts_np, cfg.group_size, axis=0)
            p_t = torch.as_tensor(prompts_rep, device=dev, dtype=torch.long)
            with torch.no_grad():
                resp_t, old_lps = model.generate(p_t, task.max_response_len, temperature=cfg.temperature)
            resp_np = resp_t.cpu().numpy()
            total_rollouts += int(resp_np.shape[0])

            for p_idx in range(cfg.prompts_per_batch):
                sl = slice(p_idx * cfg.group_size, (p_idx + 1) * cfg.group_size)
                grp_rews = []
                for j in range(sl.start, sl.stop):
                    r_raw, _ = task.score_sequence(prompts_rep[j], resp_np[j])
                    if task.eos_token_id is not None:
                        eos_pos = np.flatnonzero(resp_np[j] == task.eos_token_id)
                        eff_len = int(eos_pos[0]) + 1 if eos_pos.size > 0 else len(resp_np[j])
                    else:
                        eff_len = len(resp_np[j])
                    r_shaped = _apply_overlong_shaping(
                        r_raw, eff_len, task.max_response_len, cfg.soft_overlong_cache, cfg.overlong_penalty
                    )
                    grp_rews.append(r_shaped)
                    all_raw_rews.append(r_raw)

                grp_arr = np.asarray(grp_rews, dtype=np.float32)
                std = float(np.std(grp_arr))
                # [AI-Prompt Hook]: Dynamic Sampling skips degenerate zero-variance groups.
                if cfg.dynamic_sampling and std < 1e-6 and attempts < cfg.max_resample_rounds:
                    continue
                adv_grp = (grp_arr - np.mean(grp_arr)) / (std + 1e-6)
                kept_prompts.append(p_t[sl])
                kept_resps.append(resp_t[sl])
                kept_old_lps.append(old_lps[sl])
                kept_advs.append(torch.as_tensor(adv_grp, device=dev, dtype=torch.float32).unsqueeze(1))

        if not kept_prompts:
            continue
        b_prompts = torch.cat(kept_prompts, dim=0)
        b_resps = torch.cat(kept_resps, dim=0)
        b_old_lps = torch.cat(kept_old_lps, dim=0)
        b_adv = torch.cat(kept_advs, dim=0)
        kept_fraction_hist.append(min(1.0, float(len(kept_prompts)) / max(1, cfg.prompts_per_batch * attempts)))

        valid_mask = AutoregressiveSeqPolicy.eos_valid_mask(b_resps, task.eos_token_id)
        token_denom = valid_mask.sum().clamp(min=1.0)

        for _ in range(cfg.update_epochs):
            new_lps, entropy = model.sequence_logprobs_and_entropy(b_prompts, b_resps, temperature=cfg.temperature)
            ratio = torch.exp(new_lps - b_old_lps)
            # [AI-Prompt Hook]: Clip-Higher asymmetric bounds + Token-Level Policy Gradient Loss.
            clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_low, 1.0 + cfg.clip_high)
            token_obj = torch.min(ratio * b_adv, clipped_ratio * b_adv) * valid_mask
            loss = -token_obj.sum() / token_denom

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ent_hist.append(float(((entropy * valid_mask).sum() / token_denom).item()))

        if step % max(1, cfg.total_updates // 10) == 0 or step == cfg.total_updates:
            learning_curve.append({"step": float(step), "episode_return": float(np.mean(all_raw_rews))})

    def generate_fn(prompts_in: np.ndarray) -> np.ndarray:
        p_t = torch.as_tensor(prompts_in, device=dev, dtype=torch.long)
        with eval_mode(model), torch.no_grad():
            resp, _ = model.generate(p_t, task.max_response_len, deterministic=True)
        return resp.cpu().numpy()

    eval_stats = evaluate_sequence_task(task, generate_fn, num_prompts=cfg.eval_prompts, seed=cfg.seed + 10_000)
    cfg_dict = asdict(cfg)
    opts = resolve_run_optimizations(META.id, config=cfg_dict)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=total_rollouts,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=generate_fn,
        model=model,
        config=cfg_dict,
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
            "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
            "mean_dynamic_kept_fraction": float(np.mean(kept_fraction_hist)) if kept_fraction_hist else 1.0,
        },
        eval_stats=eval_stats,
        optimizations_used=[o.id for o in opts],
    )
