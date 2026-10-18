import numpy as np

from pmf import LowRankMF
from pmf.solvers import SVTSolver


def test_svt_recovers_a_noiseless_low_rank_matrix():
    problem = LowRankMF(n=60, p=50, true_rank=3, R=3, lambda_reg=1.0,
                        singular_values=[30, 20, 10], noise_level=0.0, seed=0,
                        missing_frac=0.4)
    solver = SVTSolver(problem)
    history = solver.fit(max_iter=1000, tol=1e-6)
    assert history["loss"][-1] < 1e-6
    assert problem.relative_error(solver.estimate()) < 1e-4
    assert history["rank"][-1] == 3


def test_svt_rank_cap_is_respected():
    problem = LowRankMF(n=40, p=30, true_rank=6, R=2, lambda_reg=1.0,
                        noise_level=0.1, seed=1, missing_frac=0.3)
    solver = SVTSolver(problem, tau=5.0, rank_max=2)
    history = solver.fit(max_iter=50, tol=None)
    assert max(history["rank"]) <= 2
