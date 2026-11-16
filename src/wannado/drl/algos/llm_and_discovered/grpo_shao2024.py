"""GRPO: Group Relative Policy Optimization (Shao et al., DeepSeekMath, arXiv:2402.03300).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : llm_and_discovered / S (Critic-Free Reasoning & Verifiable Sequence RL)
- Parent        : `ppo` (Schulman et al., 2017)
- Key Diff      : Drops the Critic value network entirely! For each prompt q, samples a
                  group of G candidate sequences, normalizes each sequence's verifiable
                  reward relative to its group mean/std, and optimizes clipped token-level
                  policy ratios plus an unbiased k3 KL penalty against a frozen reference policy.
- Extension Hook [AI-Prompt Hook]:
    1. Decouple clip bounds (Clip-Higher) + Dynamic Sampling + Token-level loss -> `dapo`.
    2. Replace token-level ratio with length-normalized sequence ratio -> `gspo`.
    3. Add entropy-driven tool-call branching rollouts -> `arpo`.
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
    id="grpo",
    name="Group Relative Policy Optimization (GRPO)",
    family="llm_and_discovered",
    paradigm="group_relative",
    tier="S",
    on_policy=True,
    sample_efficiency="medium",
    action_spaces=("autoregressive_sequence",),
    env_modalities=("reasoning_sequence", "llm_rl"),
    supports_action_mask=False,
    supports_recurrent=True,
    parent_id="ppo",
    diff_from_parent="Removes the Critic value model; estimates baseline advantages across G sampled rollouts per prompt + k3 KL regularization.",
    summary_plain="Critic-free sequence RL comparing G parallel attempts per prompt against each other; foundation of DeepSeek-R1 style RL.",
    when_to_use="Sequence/reasoning/code/symbolic generation tasks with verifiable final outcome scores.",
    when_not_to_use="Dense step-by-step continuous control (`sac`/`ppo`) or when you need DAPO's anti-entropy-collapse fixes (`dapo`).",
    hyperparam_intuition={
        "group_size": "Number of candidate responses sampled per prompt (G) to compute group baseline.",
        "kl_coef": "Weight of k3 KL divergence keeping policy close to frozen reference model.",
        "clip_coef": "PPO-style probability ratio clipping threshold.",
    },
    extension_hooks=(
        "Add Clip-Higher + Dynamic Sampling + Token-level loss -> `dapo`",
        "Switch to sequence-level geometric mean ratio -> `gspo`",
        "Add post-tool entropy branching rollouts -> `arpo`",
    ),
    module_path="wannado.drl.algos.llm_and_discovered.grpo_shao2024",
    citation=AlgoCitation(
        key="shao2024deepseekmath",
        title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        authors="Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo",
        venue="arXiv preprint arXiv:2402.03300",
        year=2024,
        url="https://arxiv.org/abs/2402.03300",
        bibtex=(
            "@article{shao2024deepseekmath,\n"
            "  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},\n"
            "  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},\n"
            "  journal={arXiv preprint arXiv:2402.03300},\n"
            "  year={2024},\n"
            "  url={https://arxiv.org/abs/2402.03300}\n"
            "}"
        ),
        ieee='Z. Shao et al., "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," arXiv:2402.03300, 2024.',
        apa="Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300.",
    ),
)


@dataclass
class GRPOConfig:
    total_updates: int = 80
    prompts_per_batch: int = 8
    group_size: int = 6
    update_epochs: int = 2
    lr: float = 2e-3
    clip_coef: float = 0.2
    kl_coef: float = 0.04
    temperature: float = 1.0
    embed_dim: int = 48
    hidden_dim: int = 64
    eval_prompts: int = 24
    device: str = "auto"
    seed: int = 42


def train_grpo(
    task: SequenceTaskEnv,
    config: GRPOConfig | None = None,
    *,
    policy: AutoregressiveSeqPolicy | None = None,
) -> TrainResult:
    """Train Group Relative Policy Optimization (GRPO) on a verifiable SequenceTaskEnv."""
    cfg = config or GRPOConfig()
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
    ent_hist, kl_hist = [], []
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
        adv_t = torch.as_tensor(adv_grp.reshape(-1, 1), device=dev, dtype=torch.float32)

        for _ in range(cfg.update_epochs):
            new_lps, entropy = model.sequence_logprobs_and_entropy(prompts_t, responses_t)
            ratio = torch.exp(new_lps - old_lps)
            # [AI-Prompt Hook]: GRPO clipped token surrogate + unbiased k3 KL divergence to frozen reference.
            pg_1 = ratio * adv_t
            pg_2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_t
            pg_obj = torch.min(pg_1, pg_2)

            log_ratio_ref = ref_lps - new_lps
            kl_k3 = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
            loss = -(pg_obj - cfg.kl_coef * kl_k3).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ent_hist.append(float(entropy.mean().item()))
            kl_hist.append(float(kl_k3.mean().item()))

        if step % max(1, cfg.total_updates // 10) == 0 or step == cfg.total_updates:
            learning_curve.append({"step": float(step), "episode_return": float(np.mean(rews_np)), "kl": kl_hist[-1]})

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
            "mean_kl_ref": float(np.mean(kl_hist)) if kl_hist else 0.0,
        },
        eval_stats=eval_stats,
    )
