import numpy as np
import pytest

from pmf.solvers import ClosedFormSolver, GradientDescentSolver


def test_gradient_descent_reaches_closed_form(complete_problem):
    solver = GradientDescentSolver(complete_problem, init_scale=0.1)
    solver.fit(max_iter=5000, grad_tol=1e-9)
    report = solver.analyze_solution()
    assert report["grad_norm"] < 1e-8
    assert abs(report["relative_loss_gap_Y"]) < 1e-10


@pytest.mark.parametrize("fixture", ["complete_problem", "masked_problem"])
def test_gradient_descent_loss_never_increases(fixture, request):
    problem = request.getfixturevalue(fixture)
    history = GradientDescentSolver(problem, init_scale=0.5).fit(max_iter=300)
    loss = np.array(history["loss"])
    assert np.all(np.diff(loss) <= 1e-12 * np.abs(loss[:-1]))


def test_gradient_descent_matches_als_with_mask(masked_problem):
    from pmf.solvers import ALSSolver
    gd = GradientDescentSolver(masked_problem, init_scale=0.1)
    gd.fit(max_iter=20000, grad_tol=1e-9)
    als = ALSSolver(masked_problem, init_scale=0.1)
    als.fit(max_iter=5000, tol=1e-15)
    assert np.isclose(masked_problem.objective(gd.U, gd.V),
                      masked_problem.objective(als.U, als.V), rtol=1e-9)
