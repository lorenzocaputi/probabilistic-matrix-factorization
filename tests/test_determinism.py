import numpy as np
import pytest

from pmf import LowRankMF
from pmf.solvers import ALL_SOLVERS


def run(cls):
    problem = LowRankMF(n=20, p=15, true_rank=3, R=3, lambda_reg=1.0,
                        noise_level=0.1, seed=7, missing_frac=0.25)
    solver = cls(problem)
    solver.fit(max_iter=20)
    return solver.estimate(), np.array(solver.history["loss"])


@pytest.mark.parametrize("cls", ALL_SOLVERS, ids=lambda c: c.__name__)
def test_fixed_seed_gives_identical_results(cls):
    Z1, loss1 = run(cls)
    Z2, loss2 = run(cls)
    assert np.array_equal(Z1, Z2)
    assert np.array_equal(loss1, loss2)
