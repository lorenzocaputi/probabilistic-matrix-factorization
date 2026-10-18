"""Solvers for a LowRankMF problem. Each one is used as Solver(problem).fit(...)."""

from pmf.solvers.als import ALSSolver
from pmf.solvers.closed_form import ClosedFormSolver

ALL_SOLVERS = [
    ClosedFormSolver,
    ALSSolver,
]

__all__ = [cls.__name__ for cls in ALL_SOLVERS] + ["ALL_SOLVERS"]
