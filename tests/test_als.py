import numpy as np
import pytest

from pmf.solvers import ALSSolver


@pytest.mark.parametrize("fixture", ["complete_problem", "masked_problem"])
def test_als_loss_never_increases(fixture, request):
    problem = request.getfixturevalue(fixture)
    history = ALSSolver(problem, init_scale=1.0).fit(max_iter=200)
    loss = np.array(history["loss"])
    assert np.all(np.diff(loss) <= 1e-10 * np.abs(loss[:-1]))


def test_als_from_random_start_reaches_closed_form(complete_problem):
    solver = ALSSolver(complete_problem, init_scale=0.1)
    solver.fit(max_iter=2000, tol=1e-15)
    report = solver.analyze_solution()
    assert abs(report["relative_loss_gap_Y"]) < 1e-10
    assert report["grad_norm"] < 1e-6


def test_als_follows_the_scalar_recursion_of_the_notes(complete_problem):
    # Propositions 1 and 4: started on scaled singular vectors, ALS only rescales
    # them, each scale following gamma <- sigma * gamma / (lambda + gamma^2).
    # (Few iterations: the subspace is a saddle, so round-off grows geometrically.)
    problem = complete_problem
    index_set = [0, 2, 4]
    V0 = problem.init_V_from_svd_directions(index_set, scaling="random")
    F, s, Gt = problem.get_svd()
    gamma = np.linalg.norm(V0, axis=0)
    solver = ALSSolver(problem, V0=V0)
    for _ in range(6):
        solver.fit(max_iter=1)
        gamma_u = s[index_set] * gamma / (problem.lam + gamma**2)
        gamma = s[index_set] * gamma_u / (problem.lam + gamma_u**2)
        assert np.allclose(solver.U, F[:, index_set] * gamma_u, atol=1e-10)
        assert np.allclose(solver.V, Gt[index_set].T * gamma, atol=1e-10)
