import numpy as np

from pmf.solvers import ALSSolver, ClosedFormSolver, SoftImputeSolver


def test_closed_form_is_stationary_and_global(complete_problem):
    solver = ClosedFormSolver(complete_problem)
    solver.fit()
    report = solver.analyze_solution()
    assert report["grad_norm"] < 1e-10
    assert report["relative_loss_gap_Y"] == 0.0


def test_als_soft_impute_and_closed_form_agree_on_complete_data(complete_problem):
    closed = ClosedFormSolver(complete_problem)
    closed.fit()
    als = ALSSolver(complete_problem, init_scale=0.1)
    als.fit(max_iter=2000, tol=1e-15)
    soft = SoftImputeSolver(complete_problem)
    soft.fit()

    Z = closed.estimate()
    assert np.max(np.abs(als.estimate() - Z)) < 1e-6
    assert np.max(np.abs(soft.estimate() - Z)) < 1e-6


def test_spectral_estimate_with_mask_uses_only_observed_entries(masked_problem):
    solver = ClosedFormSolver(masked_problem)
    solver.fit()
    Z = solver.estimate()
    masked_problem.Y[~masked_problem.mask] = 1e6
    solver.fit()
    assert np.allclose(solver.estimate(), Z)
