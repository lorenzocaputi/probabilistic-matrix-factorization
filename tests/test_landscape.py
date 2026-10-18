import numpy as np

from pmf import LowRankMF
from pmf.landscape import (ProjectedGradientSolver, hessian_inertia, make_S, pairings,
                           profiled_loss, riemannian_grad, swap_curvature)


def setup(seed=0, p=12, R=3):
    rng = np.random.RandomState(seed)
    eigvals = np.array([9.0, 7.0, 5.0, 3.0] + [1.0] * (p - 4))
    S, G = make_S(p, eigvals, rng)
    d = np.array([2.0, 0.2, 1.0])
    return S, G, d, eigvals, rng


def test_both_pairings_are_stationary_and_good_is_lower():
    S, G, d, _, _ = setup()
    V_good, V_bad = pairings(G, [0, 1, 2], d)
    for V in (V_good, V_bad):
        assert np.linalg.norm(riemannian_grad(S, V, d)) < 1e-10
    assert profiled_loss(S, V_good, d) < profiled_loss(S, V_bad, d)


def test_swap_curvature_matches_finite_differences():
    S, G, d, eigvals, _ = setup()
    w = 1.0 / (1.0 + d)
    V = G[:, :3].copy()  # eigenvalue i in slot i
    a, b = 0, 1

    def rotated(theta):
        Vt = V.copy()
        Vt[:, a] = np.cos(theta) * G[:, 0] + np.sin(theta) * G[:, 1]
        Vt[:, b] = -np.sin(theta) * G[:, 0] + np.cos(theta) * G[:, 1]
        return Vt

    eps = 1e-4
    fd = (profiled_loss(S, rotated(eps), d) - 2 * profiled_loss(S, rotated(0.0), d)
          + profiled_loss(S, rotated(-eps), d)) / eps**2
    assert np.isclose(fd, swap_curvature((eigvals[0], eigvals[1]), (w[a], w[b])), rtol=1e-4)


def test_batched_runs_match_one_at_a_time():
    S, G, d, _, _ = setup()
    V_good, V_bad = pairings(G, [0, 1, 2], d)
    batch = ProjectedGradientSolver(S, d, V_reference=V_good)
    h = batch.fit(V_bad, trials=3, steps=50, rng=np.random.RandomState(1))
    rng = np.random.RandomState(1)
    for t in range(3):
        single = ProjectedGradientSolver(S, d, V_reference=V_good)
        hs = single.fit(V_bad, trials=1, steps=50, rng=rng)
        assert np.allclose(h["loss"][t], hs["loss"][0], atol=1e-12)


def test_hessian_inertia_separates_minimum_and_saddle():
    problem = LowRankMF(n=12, p=10, true_rank=4, R=2, lambda_reg=1.0,
                        singular_values=[8, 6, 4, 3], seed=0)
    neg, _, pos, _ = hessian_inertia(problem, *problem.stationary_point([0, 1]))
    assert neg == 0 and pos > 0
    neg, _, pos, _ = hessian_inertia(problem, *problem.stationary_point([0, 2]))
    assert neg == 1 and pos > 0
