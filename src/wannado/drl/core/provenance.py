"""Hardwired zero-hallucination provenance registry for DRL optimizations & audit protocols.

Every algorithmic optimization (GAE, Action Masking, ObsNorm, Double-Q, Clipped Double-Q,
RLPD Symmetric Replay, LayerNorm Critic, GRU Memory, Clip-Higher, IQM Rigor Audit, etc.)
maps strictly to its verified academic paper so Stage 4 reports cite BOTH the main algorithm
AND every active optimization/trick without LLM memory guessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from wannado.drl.core.types import AlgoCitation


@dataclass(frozen=True)
class OptimizationProvenance:
    """Verified academic source card for an RL optimization or audit technique."""

    id: str
    name: str
    role_plain: str
    citation: AlgoCitation

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role_plain": self.role_plain,
            "citation": asdict(self.citation),
        }


OPTIMIZATION_PROVENANCE: dict[str, OptimizationProvenance] = {
    "gae": OptimizationProvenance(
        id="gae",
        name="Generalized Advantage Estimation (GAE)",
        role_plain="Exponentially weighted lambda-return advantage estimator balancing TD bias vs Monte Carlo variance.",
        citation=AlgoCitation(
            key="schulman2016high",
            title="High-Dimensional Continuous Control Using Generalized Advantage Estimation",
            authors="John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel",
            venue="International Conference on Learning Representations (ICLR)",
            year=2016,
            url="https://arxiv.org/abs/1506.02438",
            bibtex=(
                "@inproceedings{schulman2016high,\n"
                "  title={High-Dimensional Continuous Control Using Generalized Advantage Estimation},\n"
                "  author={Schulman, John and Moritz, Philipp and Levine, Sergey and Jordan, Michael I. and Abbeel, Pieter},\n"
                "  booktitle={International Conference on Learning Representations (ICLR)},\n"
                "  year={2016},\n"
                "  url={https://arxiv.org/abs/1506.02438}\n"
                "}"
            ),
            ieee='J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," in Proc. ICLR, 2016.',
            apa="Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. ICLR.",
        ),
    ),
    "action_masking": OptimizationProvenance(
        id="action_masking",
        name="Invalid Action Masking",
        role_plain="Masks out infeasible discrete actions by setting logits/Q-values to -inf while preserving valid policy gradients.",
        citation=AlgoCitation(
            key="huang2022closer",
            title="A Closer Look at Invalid Action Masking in Policy Gradient Algorithms",
            authors="Shengyi Huang and Santiago Ontanon",
            venue="Proceedings of the 35th International FLAIRS Conference",
            year=2022,
            url="https://arxiv.org/abs/2006.14171",
            bibtex=(
                "@inproceedings{huang2022closer,\n"
                "  title={A Closer Look at Invalid Action Masking in Policy Gradient Algorithms},\n"
                "  author={Huang, Shengyi and Onta{\\~n}{\\'o}n, Santiago},\n"
                "  booktitle={Proceedings of the International Florida Artificial Intelligence Research Society Conference (FLAIRS)},\n"
                "  volume={35},\n"
                "  year={2022},\n"
                "  doi={10.32473/flairs.v35i.130584},\n"
                "  url={https://arxiv.org/abs/2006.14171}\n"
                "}"
            ),
            ieee='S. Huang and S. Ontanon, "A closer look at invalid action masking in policy gradient algorithms," in Proc. FLAIRS, vol. 35, 2022.',
            apa="Huang, S., & Ontanon, S. (2022). A closer look at invalid action masking in policy gradient algorithms. Proceedings of FLAIRS, 35.",
        ),
    ),
    "obs_normalization": OptimizationProvenance(
        id="obs_normalization",
        name="Running Observation Normalization (with Eval-Time Freeze)",
        role_plain="Welford running standardization of state vectors during training, frozen strictly during deterministic evaluation.",
        citation=AlgoCitation(
            key="engstrom2020implementation",
            title="Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO",
            authors="Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry",
            venue="International Conference on Learning Representations (ICLR)",
            year=2020,
            url="https://arxiv.org/abs/2005.12729",
            bibtex=(
                "@inproceedings{engstrom2020implementation,\n"
                "  title={Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO},\n"
                "  author={Engstrom, Logan and Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Janoos, Firdaus and Rudolph, Larry and M{\\k{a}}dry, Aleksander},\n"
                "  booktitle={International Conference on Learning Representations (ICLR)},\n"
                "  year={2020},\n"
                "  url={https://arxiv.org/abs/2005.12729}\n"
                "}"
            ),
            ieee='L. Engstrom et al., "Implementation matters in deep policy gradients: A case study on PPO and TRPO," in Proc. ICLR, 2020.',
            apa="Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation matters in deep policy gradients: A case study on PPO and TRPO. ICLR.",
        ),
    ),
    "orthogonal_init_and_clip": OptimizationProvenance(
        id="orthogonal_init_and_clip",
        name="Orthogonal Layer Initialization, LR Annealing & Global Grad Clipping",
        role_plain="Code-level policy-gradient stabilization practices (orthogonal init, gradient clipping, linear LR decay).",
        citation=AlgoCitation(
            key="engstrom2020implementation",
            title="Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO",
            authors="Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry",
            venue="International Conference on Learning Representations (ICLR)",
            year=2020,
            url="https://arxiv.org/abs/2005.12729",
            bibtex=(
                "@inproceedings{engstrom2020implementation,\n"
                "  title={Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO},\n"
                "  author={Engstrom, Logan and Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Janoos, Firdaus and Rudolph, Larry and M{\\k{a}}dry, Aleksander},\n"
                "  booktitle={International Conference on Learning Representations (ICLR)},\n"
                "  year={2020},\n"
                "  url={https://arxiv.org/abs/2005.12729}\n"
                "}"
            ),
            ieee='L. Engstrom et al., "Implementation matters in deep policy gradients: A case study on PPO and TRPO," in Proc. ICLR, 2020.',
            apa="Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation matters in deep policy gradients: A case study on PPO and TRPO. ICLR.",
        ),
    ),
    "double_q_target": OptimizationProvenance(
        id="double_q_target",
        name="Double Q-Learning Target Decoupling",
        role_plain="Decouples greedy next-action selection (online Q) from evaluation (target Q) to suppress maximization bias.",
        citation=AlgoCitation(
            key="vanhasselt2016deep",
            title="Deep Reinforcement Learning with Double Q-learning",
            authors="Hado van Hasselt, Arthur Guez, and David Silver",
            venue="Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)",
            year=2016,
            url="https://arxiv.org/abs/1509.06461",
            bibtex=(
                "@inproceedings{vanhasselt2016deep,\n"
                "  title={Deep Reinforcement Learning with Double Q-learning},\n"
                "  author={van Hasselt, Hado and Guez, Arthur and Silver, David},\n"
                "  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},\n"
                "  year={2016},\n"
                "  url={https://arxiv.org/abs/1509.06461}\n"
                "}"
            ),
            ieee='H. van Hasselt, A. Guez, and D. Silver, "Deep reinforcement learning with double Q-learning," in Proc. AAAI, 2016.',
            apa="van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. AAAI.",
        ),
    ),
    "clipped_double_q": OptimizationProvenance(
        id="clipped_double_q",
        name="Clipped Double-Q Critic Target min(Q1, Q2)",
        role_plain="Takes the minimum over twin target Q-networks to prevent continuous actor-critic Q-overestimation.",
        citation=AlgoCitation(
            key="fujimoto2018addressing",
            title="Addressing Function Approximation Error in Actor-Critic Methods",
            authors="Scott Fujimoto, Herke van Hoof, and David Meger",
            venue="Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80",
            year=2018,
            url="https://proceedings.mlr.press/v80/fujimoto18a.html",
            bibtex=(
                "@inproceedings{fujimoto2018addressing,\n"
                "  title={Addressing Function Approximation Error in Actor-Critic Methods},\n"
                "  author={Fujimoto, Scott and van Hoof, Herke and Meger, David},\n"
                "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
                "  volume={80},\n"
                "  year={2018},\n"
                "  url={https://proceedings.mlr.press/v80/fujimoto18a.html}\n"
                "}"
            ),
            ieee='S. Fujimoto, H. van Hoof, and D. Meger, "Addressing function approximation error in actor-critic methods," in Proc. ICML, PMLR 80, 2018.',
            apa="Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. ICML (PMLR 80).",
        ),
    ),
    "rlpd_symmetric_replay": OptimizationProvenance(
        id="rlpd_symmetric_replay",
        name="RLPD 50/50 Symmetric Offline-Online Replay & High-UTD",
        role_plain="Mixes batches 50/50 between offline demos/human interventions and online rollouts under high update-to-data ratios.",
        citation=AlgoCitation(
            key="ball2023efficient",
            title="Efficient Online Reinforcement Learning with Offline Data",
            authors="Philip J. Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine",
            venue="Proceedings of the 40th International Conference on Machine Learning (ICML), PMLR 202",
            year=2023,
            url="https://arxiv.org/abs/2302.02948",
            bibtex=(
                "@inproceedings{ball2023efficient,\n"
                "  title={Efficient Online Reinforcement Learning with Offline Data},\n"
                "  author={Ball, Philip J. and Smith, Laura and Kostrikov, Ilya and Levine, Sergey},\n"
                "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
                "  year={2023},\n"
                "  url={https://arxiv.org/abs/2302.02948}\n"
                "}"
            ),
            ieee='P. J. Ball, L. Smith, I. Kostrikov, and S. Levine, "Efficient online reinforcement learning with offline data," in Proc. ICML, 2023.',
            apa="Ball, P. J., Smith, L., Kostrikov, I., & Levine, S. (2023). Efficient online reinforcement learning with offline data. ICML.",
        ),
    ),
    "layer_norm_critic": OptimizationProvenance(
        id="layer_norm_critic",
        name="Layer Normalization in Critic Networks",
        role_plain="Stabilizes Q-network activations and prevents catastrophic Q-divergence under high UTD and OOD data.",
        citation=AlgoCitation(
            key="ba2016layer",
            title="Layer Normalization",
            authors="Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton",
            venue="arXiv preprint arXiv:1607.06450",
            year=2016,
            url="https://arxiv.org/abs/1607.06450",
            bibtex=(
                "@article{ba2016layer,\n"
                "  title={Layer Normalization},\n"
                "  author={Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E.},\n"
                "  journal={arXiv preprint arXiv:1607.06450},\n"
                "  year={2016},\n"
                "  url={https://arxiv.org/abs/1607.06450}\n"
                "}"
            ),
            ieee='J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," arXiv:1607.06450, 2016.',
            apa="Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. arXiv:1607.06450.",
        ),
    ),
    "gru_memory": OptimizationProvenance(
        id="gru_memory",
        name="Gated Recurrent Unit (GRU) Temporal Memory",
        role_plain="Recurrent gated state encoder tracking partial observability and backward meta-update rollouts.",
        citation=AlgoCitation(
            key="cho2014learning",
            title="Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
            authors="Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio",
            venue="Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
            year=2014,
            url="https://arxiv.org/abs/1406.1078",
            bibtex=(
                "@inproceedings{cho2014learning,\n"
                "  title={Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation},\n"
                "  author={Cho, Kyunghyun and van Merri{\\\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},\n"
                "  booktitle={Proceedings of EMNLP},\n"
                "  year={2014},\n"
                "  url={https://arxiv.org/abs/1406.1078}\n"
                "}"
            ),
            ieee='K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," in Proc. EMNLP, 2014.',
            apa="Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP.",
        ),
    ),
    "iqm_rigor_audit": OptimizationProvenance(
        id="iqm_rigor_audit",
        name="Interquartile Mean (IQM) Statistical Aggregation",
        role_plain="Drops top/bottom 25% outlier seed scores for statistically reliable multi-seed DRL comparison.",
        citation=AlgoCitation(
            key="agarwal2021deep",
            title="Deep Reinforcement Learning at the Edge of the Statistical Precipice",
            authors="Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, and Marc G. Bellemare",
            venue="Advances in Neural Information Processing Systems (NeurIPS)",
            year=2021,
            url="https://arxiv.org/abs/2108.13264",
            bibtex=(
                "@inproceedings{agarwal2021deep,\n"
                "  title={Deep Reinforcement Learning at the Edge of the Statistical Precipice},\n"
                "  author={Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and Courville, Aaron C. and Bellemare, Marc G.},\n"
                "  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},\n"
                "  year={2021},\n"
                "  url={https://arxiv.org/abs/2108.13264}\n"
                "}"
            ),
            ieee='R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare, "Deep reinforcement learning at the edge of the statistical precipice," in Proc. NeurIPS, 2021.',
            apa="Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. NeurIPS.",
        ),
    ),
    "reproducibility_multi_seed": OptimizationProvenance(
        id="reproducibility_multi_seed",
        name="Multi-Seed Variance & Reproducibility Protocol",
        role_plain="Prevents single-seed survivorship bias and enforces deterministic multi-seed baseline comparison.",
        citation=AlgoCitation(
            key="henderson2018deep",
            title="Deep Reinforcement Learning that Matters",
            authors="Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger",
            venue="Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)",
            year=2018,
            url="https://arxiv.org/abs/1709.06560",
            bibtex=(
                "@inproceedings{henderson2018deep,\n"
                "  title={Deep Reinforcement Learning that Matters},\n"
                "  author={Henderson, Peter and Islam, Riashat and Bachman, Philip and Pineau, Joelle and Precup, Doina and Meger, David},\n"
                "  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},\n"
                "  year={2018},\n"
                "  url={https://arxiv.org/abs/1709.06560}\n"
                "}"
            ),
            ieee='P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, "Deep reinforcement learning that matters," in Proc. AAAI, 2018.',
            apa="Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. AAAI.",
        ),
    ),
    "primal_dual_lagrangian": OptimizationProvenance(
        id="primal_dual_lagrangian",
        name="Primal-Dual Constrained MDP Lagrangian Ascent",
        role_plain="Pairs reward policy optimization with an adaptive dual multiplier lambda enforcing E[cost] <= limit.",
        citation=AlgoCitation(
            key="ray2019benchmarking",
            title="Benchmarking Safe Exploration in Deep Reinforcement Learning",
            authors="Alex Ray, Joshua Achiam, and Dario Amodei",
            venue="OpenAI Technical Report",
            year=2019,
            url="https://cdn.openai.com/safexp-short.pdf",
            bibtex=(
                "@article{ray2019benchmarking,\n"
                "  title={Benchmarking Safe Exploration in Deep Reinforcement Learning},\n"
                "  author={Ray, Alex and Achiam, Joshua and Amodei, Dario},\n"
                "  journal={OpenAI Technical Report},\n"
                "  year={2019},\n"
                "  url={https://cdn.openai.com/safexp-short.pdf}\n"
                "}"
            ),
            ieee='A. Ray, J. Achiam, and D. Amodei, "Benchmarking safe exploration in deep reinforcement learning," OpenAI Technical Report, 2019.',
            apa="Ray, A., Achiam, J., & Amodei, D. (2019). Benchmarking safe exploration in deep reinforcement learning. OpenAI Technical Report.",
        ),
    ),
    "pid_lagrangian": OptimizationProvenance(
        id="pid_lagrangian",
        name="PID Lagrangian Constraint Control",
        role_plain="Adds proportional, integral, and non-negative derivative damping to dual multiplier updates to stop safety oscillation.",
        citation=AlgoCitation(
            key="stooke2020responsive",
            title="Responsive Safety in Reinforcement Learning by PID Lagrangian Methods",
            authors="Adam Stooke, Joshua Achiam, and Pieter Abbeel",
            venue="Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR Vol. 119, pp. 9133-9143",
            year=2020,
            url="https://arxiv.org/abs/2007.03964",
            bibtex=(
                "@inproceedings{stooke2020responsive,\n"
                "  title={Responsive Safety in Reinforcement Learning by PID Lagrangian Methods},\n"
                "  author={Stooke, Adam and Achiam, Joshua and Abbeel, Pieter},\n"
                "  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},\n"
                "  volume={119},\n"
                "  pages={9133--9143},\n"
                "  year={2020},\n"
                "  url={https://arxiv.org/abs/2007.03964}\n"
                "}"
            ),
            ieee='A. Stooke, J. Achiam, and P. Abbeel, "Responsive safety in reinforcement learning by PID Lagrangian methods," in Proc. ICML, PMLR 119, pp. 9133-9143, 2020.',
            apa="Stooke, A., Achiam, J., & Abbeel, P. (2020). Responsive safety in reinforcement learning by PID Lagrangian methods. ICML (PMLR 119).",
        ),
    ),
    "cvar_tail_risk": OptimizationProvenance(
        id="cvar_tail_risk",
        name="Rockafellar-Uryasev CVaR Percentile Tail-Risk Surrogate",
        role_plain="Optimizes lower-alpha Conditional Value-at-Risk instead of risk-neutral expectation to suppress catastrophic worst-case crashes.",
        citation=AlgoCitation(
            key="chow2018risk",
            title="Risk-Constrained Reinforcement Learning with Percentile Risk Criteria",
            authors="Yinlam Chow, Mohammad Ghavamzadeh, Lucas Janson, and Marco Pavone",
            venue="Journal of Machine Learning Research (JMLR), Vol. 18, No. 167, pp. 1-51",
            year=2018,
            url="https://arxiv.org/abs/1512.01629",
            bibtex=(
                "@article{chow2018risk,\n"
                "  title={Risk-Constrained Reinforcement Learning with Percentile Risk Criteria},\n"
                "  author={Chow, Yinlam and Ghavamzadeh, Mohammad and Janson, Lucas and Pavone, Marco},\n"
                "  journal={Journal of Machine Learning Research},\n"
                "  volume={18},\n"
                "  number={167},\n"
                "  pages={1--51},\n"
                "  year={2018},\n"
                "  url={https://arxiv.org/abs/1512.01629}\n"
                "}"
            ),
            ieee='Y. Chow, M. Ghavamzadeh, L. Janson, and M. Pavone, "Risk-constrained reinforcement learning with percentile risk criteria," JMLR, vol. 18, no. 167, pp. 1-51, 2018.',
            apa="Chow, Y., Ghavamzadeh, M., Janson, L., & Pavone, M. (2018). Risk-constrained reinforcement learning with percentile risk criteria. JMLR, 18(167).",
        ),
    ),
}

# Static mapping from built-in algo_id -> default structural optimizations baked into that implementation
ALGO_DEFAULT_OPTIMIZATIONS: dict[str, tuple[str, ...]] = {
    "monte_carlo": (),
    "sarsa": (),
    "q_learning": (),
    "dqn": (),
    "double_dqn": ("double_q_target",),
    "dueling_dqn": ("double_q_target",),
    "c51": (),
    "reinforce": (),
    "a2c": ("gae", "orthogonal_init_and_clip"),
    "trpo": ("gae", "orthogonal_init_and_clip"),
    "ppo": ("gae", "orthogonal_init_and_clip"),
    "ppo_gru": ("gae", "orthogonal_init_and_clip", "gru_memory"),
    "ppo_lagrangian": ("gae", "orthogonal_init_and_clip", "primal_dual_lagrangian"),
    "cvar_ppo": ("gae", "orthogonal_init_and_clip", "cvar_tail_risk"),
    "ddpg": (),
    "td3": ("clipped_double_q",),
    "sac": ("clipped_double_q",),
    "sac_lagrangian": ("clipped_double_q", "pid_lagrangian"),
    "hil_serl": ("clipped_double_q", "rlpd_symmetric_replay", "layer_norm_critic"),
    "cql": (),
    "iql_offline": (),
    "vdn": (),
    "qmix": (),
    "mappo": ("gae", "orthogonal_init_and_clip"),
    "maddpg": (),
    "grpo": (),
    "dapo": (),
    "gspo": (),
    "arpo": (),
    "discorl": ("gru_memory",),
}


def resolve_run_optimizations(
    algo_id: str,
    *,
    config: dict[str, Any] | None = None,
    explicit_ids: list[str] | None = None,
    saw_action_mask: bool = False,
) -> list[OptimizationProvenance]:
    """Resolve every optimization technique genuinely active in an algorithm run."""
    ordered_ids: list[str] = []

    def _add(oid: str) -> None:
        if oid in OPTIMIZATION_PROVENANCE and oid not in ordered_ids:
            ordered_ids.append(oid)

    for oid in ALGO_DEFAULT_OPTIMIZATIONS.get(algo_id, ()):
        _add(oid)
    for oid in (explicit_ids or ()):
        _add(oid)
    if config and config.get("normalize_obs"):
        _add("obs_normalization")
    if saw_action_mask:
        _add("action_masking")

    return [OPTIMIZATION_PROVENANCE[oid] for oid in ordered_ids]
