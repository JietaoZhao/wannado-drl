"""ARPO: Agentic Reinforced Policy Optimization (Dong et al., arXiv:2507.19849).

Original single-file modular implementation for wannado-drl.

[Lineage & Intuition]
- Family / Tier : llm_and_discovered / S (Adaptive Rollout Tool-Use Agentic RL)
- Parent        : `grpo` (Shao et al., 2024)
- Key Diff      : Multi-turn tool-using agents experience sharp policy entropy spikes
                  right after receiving external tool observations. ARPO introduces:
                  1. Entropy-based Adaptive Rollout: monitors post-tool entropy H_t and
                     forks local branch continuations ONLY at high-uncertainty tool steps
                     to focus exploration budget where it matters.
                  2. Advantage Attribution: contrasts branched sub-trajectories locally and
                     blends step-level credit with global group-relative advantages.
- Note on Naming:
    Implements Dong et al. (2025) "Agentic Reinforced Policy Optimization" (arXiv:2507.19849)
    for tool-use reasoning agents (distinct from Lu et al., arXiv:2505.16282 GUI replay ARPO).
- Extension Hook [AI-Prompt Hook]:
    1. Supply custom `SequenceTaskEnv(tool_step_fn=...)` for multi-turn Python/search/DB tool loops.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from wannado.drl.core.networks import AutoregressiveSeqPolicy
from wannado.drl.core.types import AlgoCitation, AlgoMeta, TrainResult
from wannado.drl.core.utils import eval_mode, resolve_device, set_global_seed
from wannado.drl.core.wrappers import SequenceTaskEnv
from wannado.drl.mdp.baselines import evaluate_sequence_task

META = AlgoMeta(
    id="arpo",
    name="Agentic Reinforced Policy Optimization (ARPO)",
    family="llm_and_discovered",
    paradigm="group_relative",
    tier="S",
    on_policy=True,
    sample_efficiency="high",
    action_spaces=("autoregressive_sequence",),
    env_modalities=("agentic_tool_use", "reasoning_sequence", "llm_rl"),
    supports_action_mask=False,
    supports_recurrent=True,
    parent_id="grpo",
    diff_from_parent="Adds entropy-triggered local branching after tool observations and stepwise advantage attribution.",
    summary_plain="Focuses rollout branching on uncertain post-tool steps, halving tool budget relative to naive full-trajectory sampling.",
    when_to_use="Multi-turn agentic / tool-calling sequence policies where external tool feedback spikes policy uncertainty.",
    when_not_to_use="Single-turn non-tool tasks (`dapo` / `gspo`), GUI experience replay (Lu et al. 2025), or continuous control (`sac`).",
    hyperparam_intuition={
        "entropy_branch_ratio": "Multiplier over initial entropy triggering partial local rollouts after tool calls.",
        "branch_width": "Number of local continuations spawned at high-entropy tool steps.",
        "group_size": "Global trajectory rollouts per prompt.",
    },
    extension_hooks=(
        "Plug in external simulator/API callback via `SequenceTaskEnv.tool_step_fn`",
        "Combine ARPO's adaptive branching with GSPO sequence-level ratios",
    ),
    module_path="wannado.drl.algos.llm_and_discovered.arpo_dong2025",
    citation=AlgoCitation(
        key="dong2025arpo",
        title="Agentic Reinforced Policy Optimization",
        authors="Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, et al.",
        venue="arXiv preprint arXiv:2507.19849",
        year=2025,
        url="https://arxiv.org/abs/2507.19849",
        bibtex=(
            "@article{dong2025arpo,\n"
            "  title={Agentic Reinforced Policy Optimization},\n"
            "  author={Dong, Guanting and Mao, Hangyu and Ma, Kai and Bao, Licheng and Chen, Yifei and Wang, Zhongyuan and others},\n"
            "  journal={arXiv preprint arXiv:2507.19849},\n"
            "  year={2025},\n"
            "  url={https://arxiv.org/abs/2507.19849}\n"
            "}"
        ),
        ieee='G. Dong et al., "Agentic reinforced policy optimization," arXiv:2507.19849, 2025.',
        apa="Dong, G., Mao, H., Ma, K., Bao, L., Chen, Y., Wang, Z., et al. (2025). Agentic reinforced policy optimization. arXiv:2507.19849.",
    ),
)


@dataclass
class ARPOConfig:
    total_updates: int = 80
    prompts_per_batch: int = 6
    group_size: int = 4
    branch_width: int = 2
    max_branches_per_prompt: int = 2
    entropy_branch_ratio: float = 0.85
    branch_adv_weight: float = 0.5
    update_epochs: int = 2
    lr: float = 2e-3
    clip_coef: float = 0.2
    kl_coef: float = 0.03
    temperature: float = 1.0
    embed_dim: int = 48
    hidden_dim: int = 64
    eval_prompts: int = 24
    device: str = "auto"
    seed: int = 42


def _rollout_with_entropy(
    model: AutoregressiveSeqPolicy,
    prompts_t: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    tool_step_fn: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate tokens while tracking step-level policy entropies and optional tool rewards."""
    cur = prompts_t
    toks, lps, ents = [], [], []
    tool_rewards = torch.zeros(prompts_t.shape[0], device=prompts_t.device, dtype=torch.float32)

    with torch.no_grad():
        for t in range(max_new_tokens):
            logits = model.forward_logits(cur)[:, -1, :] / max(temperature, 1e-4)
            dist = Categorical(logits=logits)
            tok = dist.sample()
            toks.append(tok.unsqueeze(1))
            lps.append(dist.log_prob(tok).unsqueeze(1))
            ents.append(dist.entropy().unsqueeze(1))
            cur = torch.cat([cur, tok.unsqueeze(1)], dim=1)

            if tool_step_fn is not None and t == max_new_tokens // 2 - 1:
                for b in range(prompts_t.shape[0]):
                    _, step_r = tool_step_fn(prompts_t[b].cpu().numpy(), cur[b, prompts_t.shape[1] :].cpu().tolist())
                    tool_rewards[b] += float(step_r)

    return torch.cat(toks, dim=1), torch.cat(lps, dim=1), torch.cat(ents, dim=1), tool_rewards


def train_arpo(
    task: SequenceTaskEnv,
    config: ARPOConfig | None = None,
    *,
    policy: AutoregressiveSeqPolicy | None = None,
) -> TrainResult:
    """Train ARPO with entropy-based adaptive step branching and advantage attribution."""
    cfg = config or ARPOConfig()
    rng = set_global_seed(cfg.seed)
    dev = resolve_device(cfg.device)

    max_seq_len = max(128, int(task.prompt_len) + int(task.max_response_len) + 8)
    model = policy if policy is not None else AutoregressiveSeqPolicy(
        task.vocab_size, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim, max_seq_len=max_seq_len
    ).to(dev)
    ref_model = copy.deepcopy(model).to(dev)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t0 = time.perf_counter()
    learning_curve: list[dict[str, float]] = []
    ent_hist: list[float] = []
    branch_counts = 0
    total_rollouts = 0

    for step in range(1, cfg.total_updates + 1):
        prompts_np = task.sample_prompts(cfg.prompts_per_batch, rng)
        prompts_rep = np.repeat(prompts_np, cfg.group_size, axis=0)
        prompts_t = torch.as_tensor(prompts_rep, device=dev, dtype=torch.long)

        resp_t, old_lps, step_ents, tool_rews = _rollout_with_entropy(
            model, prompts_t, task.max_response_len, cfg.temperature, tool_step_fn=task.tool_step_fn
        )
        resp_np = resp_t.cpu().numpy()
        total_rollouts += int(resp_np.shape[0])

        rews_np = np.array(
            [task.score_sequence(prompts_rep[i], resp_np[i])[0] + float(tool_rews[i].item()) for i in range(resp_np.shape[0])],
            dtype=np.float32,
        )
        rews_grp = rews_np.reshape(cfg.prompts_per_batch, cfg.group_size)
        global_adv = (rews_grp - rews_grp.mean(axis=1, keepdims=True)) / (rews_grp.std(axis=1, keepdims=True) + 1e-6)
        token_adv = np.repeat(global_adv.reshape(-1, 1), task.max_response_len, axis=1)

        # [AI-Prompt Hook]: Entropy-based adaptive local branching at uncertain mid-trajectory/tool steps.
        extra_prompts, extra_resps, extra_old_lps, extra_advs = [], [], [], []
        split_t = max(1, task.max_response_len // 2)
        if task.max_response_len >= 2:
            for p_idx in range(cfg.prompts_per_batch):
                base_i = p_idx * cfg.group_size
                h_init = float(step_ents[base_i, 0].item())
                h_mid = float(step_ents[base_i, split_t].item())
                if h_mid >= cfg.entropy_branch_ratio * max(h_init, 1e-4):
                    prefix_t = torch.cat([prompts_t[base_i : base_i + 1], resp_t[base_i : base_i + 1, :split_t]], dim=1)
                    b_prefix = prefix_t.repeat(cfg.branch_width, 1)
                    rem_len = task.max_response_len - split_t
                    b_tail, b_tail_lp, _, _ = _rollout_with_entropy(model, b_prefix, rem_len, cfg.temperature)
                    branch_resps = torch.cat([resp_t[base_i : base_i + 1, :split_t].repeat(cfg.branch_width, 1), b_tail], dim=1)
                    branch_old_lp = torch.cat([old_lps[base_i : base_i + 1, :split_t].repeat(cfg.branch_width, 1), b_tail_lp], dim=1)

                    b_rews = np.array(
                        [task.score_sequence(prompts_np[p_idx], branch_resps[k].cpu().numpy())[0] for k in range(cfg.branch_width)],
                        dtype=np.float32,
                    )
                    b_adv = (b_rews - b_rews.mean()) / (b_rews.std() + 1e-6)
                    adv_mat = np.zeros((cfg.branch_width, task.max_response_len), dtype=np.float32)
                    adv_mat[:, :split_t] = global_adv[p_idx, 0] * (1.0 - cfg.branch_adv_weight)
                    adv_mat[:, split_t:] = b_adv[:, None]

                    extra_prompts.append(prompts_t[base_i : base_i + 1].repeat(cfg.branch_width, 1))
                    extra_resps.append(branch_resps)
                    extra_old_lps.append(branch_old_lp)
                    extra_advs.append(adv_mat)
                    branch_counts += 1
                    total_rollouts += cfg.branch_width

        all_prompts = torch.cat([prompts_t, *extra_prompts], dim=0) if extra_prompts else prompts_t
        all_resps = torch.cat([resp_t, *extra_resps], dim=0) if extra_resps else resp_t
        all_old_lps = torch.cat([old_lps, *extra_old_lps], dim=0) if extra_old_lps else old_lps
        all_adv = torch.as_tensor(
            np.concatenate([token_adv, *extra_advs], axis=0) if extra_advs else token_adv,
            device=dev,
            dtype=torch.float32,
        )

        valid_mask = AutoregressiveSeqPolicy.eos_valid_mask(all_resps, task.eos_token_id)
        mask_denom = valid_mask.sum().clamp(min=1.0)

        with torch.no_grad():
            ref_lps, _ = ref_model.sequence_logprobs_and_entropy(all_prompts, all_resps, temperature=cfg.temperature)

        for _ in range(cfg.update_epochs):
            new_lps, entropy = model.sequence_logprobs_and_entropy(all_prompts, all_resps, temperature=cfg.temperature)
            ratio = torch.exp(new_lps - all_old_lps)
            pg_obj = torch.min(ratio * all_adv, torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * all_adv)
            log_ref = ref_lps - new_lps
            kl_k3 = torch.exp(log_ref) - log_ref - 1.0
            loss = -((pg_obj - cfg.kl_coef * kl_k3) * valid_mask).sum() / mask_denom

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ent_hist.append(float(((entropy * valid_mask).sum() / mask_denom).item()))

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
            "final_entropy": float(ent_hist[-1]) if ent_hist else 0.0,
            "adaptive_branches_triggered": branch_counts,
        },
        eval_stats=eval_stats,
    )
