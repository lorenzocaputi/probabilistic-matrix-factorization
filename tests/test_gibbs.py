import numpy as np

from pmf import LowRankMF
from pmf.solvers import GibbsSolver


def test_posterior_mean_recovers_the_signal():
    problem = LowRankMF(n=40, p=30, true_rank=3, R=3, lambda_reg=1.0,
                        singular_values=[20, 15, 10], noise_level=0.1, seed=0,
                        missing_frac=0.3)
    solver = GibbsSolver(problem, alpha=100.0, burn_in=50)
    solver.fit(max_iter=150)
    assert solver._n_samples == 100
    error = problem.relative_error(solver.estimate())
    assert error < 0.1
    assert error < problem.relative_error(problem.Y)  # better than the raw data


def test_running_mean_averages_the_post_burn_in_samples():
    problem = LowRankMF(n=10, p=8, true_rank=2, R=2, lambda_reg=1.0,
                        noise_level=0.1, seed=1)
    solver = GibbsSolver(problem, burn_in=3)
    samples = []
    solver.fit(max_iter=8, callback=lambda s, t: samples.append(s.U @ s.V.T))
    assert np.allclose(solver.estimate(), np.mean(samples[3:], axis=0))
