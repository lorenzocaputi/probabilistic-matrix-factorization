import numpy as np
import pytest

from pmf.solvers import VBSolver


@pytest.mark.parametrize("fixture", ["complete_problem", "masked_problem"])
@pytest.mark.parametrize("learn", [True, False])
def test_elbo_never_decreases(fixture, learn, request):
    problem = request.getfixturevalue(fixture)
    solver = VBSolver(problem, init_scale=0.5, learn_noise=learn, learn_prior=learn)
    history = solver.fit(max_iter=200, tol=None)
    elbo = np.array(history["elbo"])
    assert np.all(np.diff(elbo) >= -1e-9 * np.abs(elbo[:-1]))


def test_vb_learns_the_noise_level(complete_problem):
    solver = VBSolver(complete_problem, init_scale=0.5)
    solver.fit(max_iter=2000, tol=1e-12)
    # data noise sd is 0.1; the fitted variance absorbs the rank-2 residual
    # signal (singular values 3 and 2 outside R = 3) as well
    assert 0.005 < solver.s2 < 0.1
    assert complete_problem.relative_error(solver.estimate()) < 0.3
