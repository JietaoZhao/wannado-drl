"""Non-intrusive workspace integration (`drl_lab/`) and single-file algorithm ejection."""

from wannado.drl.workspace.lab_manager import EnvInterface, eject_algo, scaffold_lab

__all__ = [
    "EnvInterface",
    "eject_algo",
    "scaffold_lab",
]
