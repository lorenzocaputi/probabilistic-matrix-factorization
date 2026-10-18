import numpy as np
import pytest

from pmf import LowRankMF
from conftest import random_factors


def finite_difference_grad(problem, U, V, eps=1e-6):
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    for M, G in ((U, gU), (V, gV)):
        for idx in np.ndindex(M.shape):
            old = M[idx]
            M[idx] = old + eps
            f_plus = problem.objective(U, V)
            M[idx] = old - eps
            f_minus = problem.objective(U, V)
            M[idx] = old
            G[idx] = (f_plus - f_minus) / (2 * eps)
    return gU, gV


@pytest.mark.parametrize("fixture", ["complete_problem", "masked_problem"])
def test_grad_matches_finite_differences(fixture, request):
    problem = request.getfixturevalue(fixture)
    U, V = random_factors(problem, seed=3)
    gU, gV = problem.grad(U, V)
    fU, fV = finite_difference_grad(problem, U, V)
    assert np.allclose(gU, fU, rtol=1e-6, atol=1e-6)
    assert np.allclose(gV, fV, rtol=1e-6, atol=1e-6)


def test_grad_with_diagonal_prior():
    problem = LowRankMF(n=12, p=9, true_rank=4, R=3, lambda_reg=[0.5, 1.0, 2.0],
                        noise_level=0.1, seed=2, missing_frac=0.2)
    U, V = random_factors(problem, seed=4)
    gU, gV = problem.grad(U, V)
    fU, fV = finite_difference_grad(problem, U, V)
    assert np.allclose(gU, fU, atol=1e-6)
    assert np.allclose(gV, fV, atol=1e-6)


@pytest.mark.parametrize("fixture", ["complete_problem", "masked_problem"])
def test_hvp_matches_finite_difference_of_grad(fixture, request):
    problem = request.getfixturevalue(fixture)
    U, V = random_factors(problem, seed=5)
    dU, dV = random_factors(problem, seed=6)
    eps = 1e-6
    gU_p, gV_p = problem.grad(U + eps * dU, V + eps * dV)
    gU_m, gV_m = problem.grad(U - eps * dU, V - eps * dV)
    hU, hV = problem.hvp(U, V, dU, dV)
    assert np.allclose(hU, (gU_p - gU_m) / (2 * eps), atol=1e-5)
    assert np.allclose(hV, (gV_p - gV_m) / (2 * eps), atol=1e-5)


def test_global_minimizer_is_stationary(complete_problem):
    U, V, loss = complete_problem.global_minimizer_Y()
    assert complete_problem.grad_norm(U, V) < 1e-10
    assert np.isclose(loss, complete_problem.objective(U, V))


def test_global_minimizer_beats_every_other_stationary_point(complete_problem):
    _, _, best = complete_problem.global_minimizer_Y()
    for index_set in ([0, 1, 3], [0, 2, 4], [1, 2, 3]):
        U, V = complete_problem.stationary_point(index_set)
        assert complete_problem.grad_norm(U, V) < 1e-10
        assert complete_problem.objective(U, V) > best


def test_diagonal_prior_pairs_largest_sigma_with_smallest_lambda():
    problem = LowRankMF(n=20, p=15, true_rank=3, R=3, lambda_reg=[3.0, 0.5, 1.5],
                        singular_values=[9, 6, 4], seed=0)
    U, V, _ = problem.global_minimizer_Y()
    col_norms = np.linalg.norm(V, axis=0) ** 2
    # slot 1 (lambda 0.5) carries sigma 9, slot 2 (1.5) sigma 6, slot 0 (3.0) sigma 4
    assert np.allclose(col_norms, [4 - 3.0, 9 - 0.5, 6 - 1.5])
    assert problem.grad_norm(U, V) < 1e-10


def test_mask_ignores_hidden_entries(masked_problem):
    U, V = random_factors(masked_problem, seed=7)
    before = masked_problem.objective(U, V)
    masked_problem.Y[~masked_problem.mask] += 100.0
    assert np.isclose(masked_problem.objective(U, V), before)


def test_same_seed_same_problem():
    a = LowRankMF(n=10, p=8, true_rank=3, R=2, lambda_reg=1.0, noise_level=0.1,
                  seed=11, missing_frac=0.3)
    b = LowRankMF(n=10, p=8, true_rank=3, R=2, lambda_reg=1.0, noise_level=0.1,
                  seed=11, missing_frac=0.3)
    assert np.array_equal(a.Y, b.Y)
    assert np.array_equal(a.mask, b.mask)
    assert np.array_equal(a.sample_initial_V(), b.sample_initial_V())
