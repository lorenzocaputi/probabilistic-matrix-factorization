"""Solvers for a LowRankMF problem. Each one is used as Solver(problem).fit(...)."""

from pmf.solvers.als import ALSSolver

__all__ = ["ALSSolver"]
