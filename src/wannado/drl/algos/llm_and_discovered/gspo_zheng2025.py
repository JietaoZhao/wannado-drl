"""GSPO: Group Sequence Policy Optimization (Zheng et al., Qwen Team, arXiv:2507.18071).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : llm_and_discovered / S (Sequence-Level Importance Ratio SOTA)
- Parent        : `grpo` (Shao et al., 2024)
- Key Diff      : Fixes the granularity mismatch between sequence-level rewards and
                  token-level importance ratios in GRPO. Defines the importance ratio as
                  the length-normalized sequence likelihood geometric mean:
                  s_i(theta) = exp((1 / |y_i|) * sum_t log(pi_theta(y_{i,t}) / pi_old(y_{i,t})))
                  and applies clipping and policy optimization directly at the sequence level.
- Extension Hook [AI-Prompt Hook]:
    1. Combine sequence-level ratio with token-wise entropy bonus or MoE routing stability hooks.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wannado.drl.core.networks import AutoregressiveSeqPolicy
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, resolve_device, set_global_seed
from wannado.drl.core.wrappers import SequenceTaskEnv
from wannado.drl.mdp.baselines import evaluate_sequence_task

META = AlgoMeta(
    id="gspo",
    name="Group Sequence Policy Optimization (GSPO)",
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
    diff_from_parent="Replaces token-level importance ratios with length-normalized sequence-level likelihood ratios and sequence-level clipping.",
    summary_plain="Aligns RL clipping granularity with sequence-level rewards via geometric-mean sequence probability ratios.",
    when_to_use="Long-sequence or MoE reasoning RL where per-token off-policy ratios cause high-variance instability.",
    when_not_to_use="Step-level dense reward control (`ppo`) or token-step branching tool agents (`arpo`).",
    hyperparam_intuition={
        "clip_coef": "Sequence-level geometric ratio clip width (often tighter than token-level PPO).",
        "group_size": "Rollouts sampled per prompt for relative sequence normalization.",
        "kl_coef": "Optional KL anchor toward reference policy.",
    },
    extension_hooks=(
        "Swap sequence-level clipping to asymmetric bounds [1-eps_low, 1+eps_high]",
        "Add length-weighted sequence advantage calibration",
    ),
    module_path="wannado.drl.algos.llm_and_discovered.gspo_zheng2025",
    citation=AlgoCitation(
        key="zheng2025group",
        title="Group Sequence Policy Optimization",
        authors="Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin",
        venue="arXiv preprint arXiv:2507.18071",
        year=2025,
        url="https://arxiv.org/abs/2507.18071",
        bibtex=(
            "@article{zheng2025group,\n"
            "  title={Group Sequence Policy Optimization},\n"
            "  author={Zheng, Chujie and Liu, Shixuan and Li, Mingze and Chen, Xiong-Hui and Yu, Bowen and Gao, Chang and Dang, Kai and Liu, Yuqiong and Men, Rui and Yang, An and Zhou, Jingren and Lin, Junyang},\n"
            "  journal={arXiv preprint arXiv:2507.18071},\n"
            "  year={2025},\n"
            "  url={https://arxiv.org/abs/2507.18071}\n"
            "}"
        ),
        ieee='C. Zheng et al., "Group sequence policy optimization," arXiv:2507.18071, 2025.',
        apa="Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., Zhou, J., & Lin, J. (2025). Group sequence policy optimization. arXiv:2507.18071.",
    ),
)


@dataclass
class GSPOConfig:
    total_updates: int = 80
    prompts_per_batch: int = 8
    group_size: int = 6
    update_epochs: int = 2
    lr: float = 2e-3
    clip_coef: float = 0.1
    kl_coef: float = 0.02
    temperature: float = 1.0
    embed_dim: int = 48
    hidden_dim: int = 64
    eval_prompts: int = 24
    device: str = "auto"
    seed: int = 42


def train_gspo(
    task: SequenceTaskEnv,
    config: GSPOConfig | None = None,
    *,
    policy: AutoregressiveSeqPolicy | None = None,
) -> TrainResult:
    """Train Group Sequence Policy Optimization (GSPO) with length-normalized sequence ratios."""
    cfg = config or GSPOConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    model = policy if policy is not None else AutoregressiveSeqPolicy(
        task.vocab_size, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim
    ).to(dev)
    ref_model = copy.deepcopy(model).to(dev)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t0 = time.perf_counter()
    learning_curve: list[dict[str, float]] = []
    ent_hist, seq_ratio_hist = [], []
    total_rollouts = 0

    for step in range(1, cfg.total_updates + 1):
        prompts_np = task.sample_prompts(cfg.prompts_per_batch, rng)
        prompts_rep = np.repeat(prompts_np, cfg.group_size, axis=0)
        prompts_t = torch.as_tensor(prompts_rep, device=dev, dtype=torch.long)

        with torch.no_grad():
            responses_t, old_lps = model.generate(prompts_t, task.max_response_len, temperature=cfg.temperature)
            ref_lps, _ = ref_model.sequence_logprobs_and_entropy(prompts_t, responses_t)

        resp_np = responses_t.cpu().numpy()
        total_rollouts += int(resp_np.shape[0])
        rews_np = np.array(
            [task.score_sequence(prompts_rep[i], resp_np[i])[0] for i in range(resp_np.shape[0])],
            dtype=np.float32,
        )
        rews_grp = rews_np.reshape(cfg.prompts_per_batch, cfg.group_size)
        adv_grp = (rews_grp - rews_grp.mean(axis=1, keepdims=True)) / (rews_grp.std(axis=1, keepdims=True) + 1e-6)
        adv_t = torch.as_tensor(adv_grp.reshape(-1), device=dev, dtype=torch.float32)

        valid_mask = (
            (responses_t != task.eos_token_id).float()
            if task.eos_token_id is not None
            else torch.ones_like(responses_t, dtype=torch.float32)
        )
        seq_lengths = valid_mask.sum(dim=1).clamp(min=1.0)

        for _ in range(cfg.update_epochs):
            new_lps, entropy = model.sequence_logprobs_and_entropy(prompts_t, responses_t)
            # [AI-Prompt Hook]: Length-normalized sequence importance ratio s_i(theta) and sequence-level clipping.
            mean_log_ratio = ((new_lps - old_lps) * valid_mask).sum(dim=1) / seq_lengths
            seq_ratio = torch.exp(mean_log_ratio.clamp(-5.0, 5.0))
            clipped_seq_ratio = torch.clamp(seq_ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            seq_obj = torch.min(seq_ratio * adv_t, clipped_seq_ratio * adv_t)

            mean_ref_diff = ((ref_lps - new_lps) * valid_mask).sum(dim=1) / seq_lengths
            kl_seq = torch.exp(mean_ref_diff.clamp(-5.0, 5.0)) - mean_ref_diff - 1.0
            loss = -(seq_obj - cfg.kl_coef * kl_seq).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ent_hist.append(float(entropy.mean().item()))
            seq_ratio_hist.append(float(seq_ratio.mean().item()))

        if step % max(1, cfg.total_updates // 10) == 0 or step == cfg.total_updates:
            learning_curve.append({"step": float(step), "episode_return": float(np.mean(rews_np))})

    def generate_fn(prompts_in: np.ndarray) -> np.ndarray:
        p_t = torch.as_tensor(prompts_in, device=dev, dtype=torch.long)
        with eval_mode(model), torch.no_grad():
            resp, _ = model.generate(p_t, task.max_response_len, deterministic=True)
        return resp.cpu().numpy()

    eval_stats = evaluate_sequence_task(task, generate_fn, num_prompts=cfg.eval_prompts, seed=cfg.seed + 10_000)

    return TrainResult(
        algo_id=META.id,
        seed=cfg.seed,
        total_timesteps=total_rollouts,
        elapsed_seconds=time.perf_counter() - t0,
        policy_fn=generate_fn,
        model=model,
        config=asdict(cfg),
        learning_curve=learning_curve,
        train_diagnostics={
            "mean_entropy": float(np.mean(ent_hist)) if ent_hist else 0.0,
            "mean_seq_ratio": float(np.mean(seq_ratio_hist)) if seq_ratio_hist else 1.0,
        },
        eval_stats=eval_stats,
    )
