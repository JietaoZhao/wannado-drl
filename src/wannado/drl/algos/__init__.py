"""CleanRL-style single-file DRL algorithms organized into 8 authoritative families.

Supports PEP 562 lazy imports such as:
    from wannado.drl.algos import train_ppo, train_sac, train_ppo_lagrangian
"""

from __future__ import annotations

import importlib
from typing import Any

ALGO_EXPORTS: dict[str, tuple[str, str]] = {
    # 1. Tabular classic foundations
    "train_monte_carlo": ("wannado.drl.algos.tabular_classic.monte_carlo_sutton2018", "train_monte_carlo"),
    "MonteCarloConfig": ("wannado.drl.algos.tabular_classic.monte_carlo_sutton2018", "MonteCarloConfig"),
    "train_sarsa": ("wannado.drl.algos.tabular_classic.sarsa_rummery1994", "train_sarsa"),
    "SarsaConfig": ("wannado.drl.algos.tabular_classic.sarsa_rummery1994", "SarsaConfig"),
    "train_q_learning": ("wannado.drl.algos.tabular_classic.q_learning_watkins1992", "train_q_learning"),
    "QLearningConfig": ("wannado.drl.algos.tabular_classic.q_learning_watkins1992", "QLearningConfig"),
    # 2. Value-based DRL
    "train_dqn": ("wannado.drl.algos.value_based.dqn_mnih2015", "train_dqn"),
    "DQNConfig": ("wannado.drl.algos.value_based.dqn_mnih2015", "DQNConfig"),
    "train_double_dqn": ("wannado.drl.algos.value_based.double_dqn_hasselt2016", "train_double_dqn"),
    "DoubleDQNConfig": ("wannado.drl.algos.value_based.double_dqn_hasselt2016", "DoubleDQNConfig"),
    "train_dueling_dqn": ("wannado.drl.algos.value_based.dueling_dqn_wang2016", "train_dueling_dqn"),
    "DuelingDQNConfig": ("wannado.drl.algos.value_based.dueling_dqn_wang2016", "DuelingDQNConfig"),
    "train_c51": ("wannado.drl.algos.value_based.c51_bellemare2017", "train_c51"),
    "C51Config": ("wannado.drl.algos.value_based.c51_bellemare2017", "C51Config"),
    # 3. Policy-gradient & on-policy actor-critic
    "train_reinforce": ("wannado.drl.algos.policy_gradient.reinforce_williams1992", "train_reinforce"),
    "ReinforceConfig": ("wannado.drl.algos.policy_gradient.reinforce_williams1992", "ReinforceConfig"),
    "train_a2c": ("wannado.drl.algos.policy_gradient.a2c_mnih2016", "train_a2c"),
    "A2CConfig": ("wannado.drl.algos.policy_gradient.a2c_mnih2016", "A2CConfig"),
    "train_trpo": ("wannado.drl.algos.policy_gradient.trpo_schulman2015", "train_trpo"),
    "TRPOConfig": ("wannado.drl.algos.policy_gradient.trpo_schulman2015", "TRPOConfig"),
    "train_ppo": ("wannado.drl.algos.policy_gradient.ppo_schulman2017", "train_ppo"),
    "PPOConfig": ("wannado.drl.algos.policy_gradient.ppo_schulman2017", "PPOConfig"),
    "train_ppo_gru": ("wannado.drl.algos.policy_gradient.ppo_gru_schulman2017", "train_ppo_gru"),
    "PPOGRUConfig": ("wannado.drl.algos.policy_gradient.ppo_gru_schulman2017", "PPOGRUConfig"),
    # 4. Off-policy continuous actor-critic & HIL
    "train_ddpg": ("wannado.drl.algos.actor_critic_off.ddpg_lillicrap2016", "train_ddpg"),
    "DDPGConfig": ("wannado.drl.algos.actor_critic_off.ddpg_lillicrap2016", "DDPGConfig"),
    "train_td3": ("wannado.drl.algos.actor_critic_off.td3_fujimoto2018", "train_td3"),
    "TD3Config": ("wannado.drl.algos.actor_critic_off.td3_fujimoto2018", "TD3Config"),
    "train_sac": ("wannado.drl.algos.actor_critic_off.sac_haarnoja2018", "train_sac"),
    "SACConfig": ("wannado.drl.algos.actor_critic_off.sac_haarnoja2018", "SACConfig"),
    "train_hil_serl": ("wannado.drl.algos.actor_critic_off.hil_serl_luo2025", "train_hil_serl"),
    "HILSERLConfig": ("wannado.drl.algos.actor_critic_off.hil_serl_luo2025", "HILSERLConfig"),
    # 5. Safe & risk-constrained RL
    "train_ppo_lagrangian": ("wannado.drl.algos.safe_constrained.ppo_lagrangian_ray2019", "train_ppo_lagrangian"),
    "PPOLagrangianConfig": ("wannado.drl.algos.safe_constrained.ppo_lagrangian_ray2019", "PPOLagrangianConfig"),
    "train_sac_lagrangian": ("wannado.drl.algos.safe_constrained.sac_lagrangian_stooke2020", "train_sac_lagrangian"),
    "SACLagrangianConfig": ("wannado.drl.algos.safe_constrained.sac_lagrangian_stooke2020", "SACLagrangianConfig"),
    "train_cvar_ppo": ("wannado.drl.algos.safe_constrained.cvar_ppo_chow2018", "train_cvar_ppo"),
    "CVaRPPOConfig": ("wannado.drl.algos.safe_constrained.cvar_ppo_chow2018", "CVaRPPOConfig"),
    # 6. Offline RL
    "train_cql": ("wannado.drl.algos.offline_rl.cql_kumar2020", "train_cql"),
    "CQLConfig": ("wannado.drl.algos.offline_rl.cql_kumar2020", "CQLConfig"),
    "train_iql_offline": ("wannado.drl.algos.offline_rl.iql_kostrikov2022", "train_iql_offline"),
    "IQLConfig": ("wannado.drl.algos.offline_rl.iql_kostrikov2022", "IQLConfig"),
    # 6. Multi-agent RL (MARL)
    "train_vdn": ("wannado.drl.algos.multi_agent.vdn_sunehag2018", "train_vdn"),
    "VDNConfig": ("wannado.drl.algos.multi_agent.vdn_sunehag2018", "VDNConfig"),
    "train_qmix": ("wannado.drl.algos.multi_agent.qmix_rashid2018", "train_qmix"),
    "QMIXConfig": ("wannado.drl.algos.multi_agent.qmix_rashid2018", "QMIXConfig"),
    "train_mappo": ("wannado.drl.algos.multi_agent.mappo_yu2022", "train_mappo"),
    "MAPPOConfig": ("wannado.drl.algos.multi_agent.mappo_yu2022", "MAPPOConfig"),
    "train_maddpg": ("wannado.drl.algos.multi_agent.maddpg_lowe2017", "train_maddpg"),
    "MADDPGConfig": ("wannado.drl.algos.multi_agent.maddpg_lowe2017", "MADDPGConfig"),
    # 7. Frontier reasoning & discovered RL
    "train_grpo": ("wannado.drl.algos.llm_and_discovered.grpo_shao2024", "train_grpo"),
    "GRPOConfig": ("wannado.drl.algos.llm_and_discovered.grpo_shao2024", "GRPOConfig"),
    "train_dapo": ("wannado.drl.algos.llm_and_discovered.dapo_yu2025", "train_dapo"),
    "DAPOConfig": ("wannado.drl.algos.llm_and_discovered.dapo_yu2025", "DAPOConfig"),
    "train_gspo": ("wannado.drl.algos.llm_and_discovered.gspo_zheng2025", "train_gspo"),
    "GSPOConfig": ("wannado.drl.algos.llm_and_discovered.gspo_zheng2025", "GSPOConfig"),
    "train_arpo": ("wannado.drl.algos.llm_and_discovered.arpo_dong2025", "train_arpo"),
    "ARPOConfig": ("wannado.drl.algos.llm_and_discovered.arpo_dong2025", "ARPOConfig"),
    "train_discorl": ("wannado.drl.algos.llm_and_discovered.discorl_oh2025", "train_discorl"),
    "DiscoRLConfig": ("wannado.drl.algos.llm_and_discovered.discorl_oh2025", "DiscoRLConfig"),
}

ALGO_ID_TO_RUNNER: dict[str, tuple[str, str]] = {
    k.removeprefix("train_"): v for k, v in ALGO_EXPORTS.items() if k.startswith("train_")
}

__all__ = [
    "A2CConfig",
    "ARPOConfig",
    "C51Config",
    "CQLConfig",
    "CVaRPPOConfig",
    "DAPOConfig",
    "DDPGConfig",
    "DQNConfig",
    "DiscoRLConfig",
    "DoubleDQNConfig",
    "DuelingDQNConfig",
    "GRPOConfig",
    "GSPOConfig",
    "HILSERLConfig",
    "IQLConfig",
    "MADDPGConfig",
    "MAPPOConfig",
    "MonteCarloConfig",
    "PPOConfig",
    "PPOGRUConfig",
    "PPOLagrangianConfig",
    "QLearningConfig",
    "QMIXConfig",
    "ReinforceConfig",
    "SACConfig",
    "SACLagrangianConfig",
    "SarsaConfig",
    "TD3Config",
    "TRPOConfig",
    "VDNConfig",
    "train_a2c",
    "train_arpo",
    "train_c51",
    "train_cql",
    "train_cvar_ppo",
    "train_dapo",
    "train_ddpg",
    "train_discorl",
    "train_double_dqn",
    "train_dqn",
    "train_dueling_dqn",
    "train_grpo",
    "train_gspo",
    "train_hil_serl",
    "train_iql_offline",
    "train_maddpg",
    "train_mappo",
    "train_monte_carlo",
    "train_ppo",
    "train_ppo_gru",
    "train_ppo_lagrangian",
    "train_q_learning",
    "train_qmix",
    "train_reinforce",
    "train_sac",
    "train_sac_lagrangian",
    "train_sarsa",
    "train_td3",
    "train_trpo",
    "train_vdn",
]


def lazy_resolve(
    name: str,
    exports: dict[str, tuple[str, str]],
    namespace: dict[str, Any],
    package_name: str,
) -> Any:
    if name not in exports:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
    module_name, attr_name = exports[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    namespace[name] = value
    return value


def __getattr__(name: str) -> Any:
    return lazy_resolve(name, ALGO_EXPORTS, globals(), __name__)


def __dir__() -> list[str]:
    return sorted(__all__)
