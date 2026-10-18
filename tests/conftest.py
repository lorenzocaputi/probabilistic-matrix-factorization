import numpy as np
import pytest

from pmf import LowRankMF


@pytest.fixture
def complete_problem():
    """Small complete-data problem with a clear spectral gap."""
    return LowRankMF(n=30, p=20, true_rank=5, R=3, lambda_reg=1.0,
                     singular_values=[10, 8, 6, 3, 2], noise_level=0.1, seed=0)


@pytest.fixture
def masked_problem():
    """Same kind of problem with 30% of the entries hidden."""
    return LowRankMF(n=30, p=20, true_rank=5, R=3, lambda_reg=1.0,
                     singular_values=[10, 8, 6, 3, 2], noise_level=0.1, seed=1,
                     missing_frac=0.3)


def random_factors(problem, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    U = scale * rng.standard_normal((problem.n, problem.R))
    V = scale * rng.standard_normal((problem.p, problem.R))
    return U, V
