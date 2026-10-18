"""Solvers for a LowRankMF problem. Each one is used as Solver(problem).fit(...)."""

from pmf.solvers.als import ALSSolver
from pmf.solvers.closed_form import ClosedFormSolver
from pmf.solvers.gradient import GradientDescentSolver
from pmf.solvers.soft_impute import SoftImputeSolver, soft_impute

ALL_SOLVERS = [
    ClosedFormSolver,
    ALSSolver,
    GradientDescentSolver,
    SoftImputeSolver,
]

__all__ = [cls.__name__ for cls in ALL_SOLVERS] + ["ALL_SOLVERS", "soft_impute"]
