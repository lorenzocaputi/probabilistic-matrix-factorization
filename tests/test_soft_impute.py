import numpy as np

from pmf import LowRankMF
from pmf.linalg import soft_threshold_svd
from pmf.solvers import SoftImputeSolver, soft_impute


def nuclear_norm_problem(seed=3):
    # R = min(n, p): no rank restriction, so soft-impute is the algorithm of
    # Mazumder, Hastie and Tibshirani for lambda * ||Z||_*.
    return LowRankMF(n=25, p=18, true_rank=4, R=18, lambda_reg=1.5,
                     singular_values=[12, 9, 6, 4], noise_level=0.2, seed=seed,
                     missing_frac=0.35)


def test_fixed_point_satisfies_the_proximal_condition():
    problem = nuclear_norm_problem()
    solver = SoftImputeSolver(problem)
    solver.fit(max_iter=5000, tol=1e-13)
    Z = solver.estimate()
    # Z = prox_{lambda ||.||_*}(P_Omega(Y) + P_Omega^perp(Z))
    F, s, Gt = soft_threshold_svd(solver.filled_matrix(), problem.lambda_reg)
    assert np.max(np.abs((F * s) @ Gt - Z)) < 1e-8


def test_masked_objective_never_increases(masked_problem):
    history = SoftImputeSolver(masked_problem).fit(max_iter=300, tol=None)
    loss = np.array(history["loss"])
    assert np.all(np.diff(loss) <= 1e-10 * np.abs(loss[:-1]))


def test_matrix_level_function_keeps_observed_entries():
    problem = nuclear_norm_problem(seed=4)
    X = soft_impute(problem.Y, problem.mask, lam=1.5, max_iter=500, tol=1e-10)
    assert np.array_equal(X[problem.mask], problem.Y[problem.mask])
    assert problem.relative_error(X) < problem.relative_error(np.where(problem.mask, problem.Y, 0.0))
