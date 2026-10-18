"""
The profiled MAP loss on the Stiefel manifold with a diagonal prior, and
tools to classify stationary points of the full MAP loss.

Profiling U out of the MAP objective with a diagonal prior precision
Delta = diag(d_1, ..., d_R) on the factors leaves (up to constants)

    L(V) = 0.5 tr(S) + 0.5 R - 0.5 tr(V^T S V W),   V^T V = I_R,

with S = Y^T Y and W = (I + Delta)^{-1} = diag(w_k), w_k = 1 / (1 + d_k).
Stationary points put an eigenvector of S in every column. A *bad pairing*
gives a larger eigenvalue to a column with a smaller weight; along the
rotation that swaps two such columns the second derivative is negative,
so a bad pairing is a strict saddle.

The functions mirror the notebook experiments/diagonal_prior_bad_pairing_saddle_experiment.ipynb.
Every random draw goes through the `rng` argument, which accepts the
legacy np.random module or a np.random.RandomState, so a fixed seed
reproduces the notebook exactly.
"""

import numpy as np


def make_S(p, eigvals, rng=np.random):
    """Return S with the given spectrum and an orthogonal eigenbasis G (columns)."""
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return Q @ np.diag(eigvals) @ Q.T, Q  # S, G


def profiled_loss(S, V, d):
    """Profiled loss for diagonal Delta with entries d (shape (R,)). V must have orthonormal columns."""
    w = 1.0 / (1.0 + d)
    W = np.diag(w)
    return 0.5 * np.trace(S) + 0.5 * V.shape[1] - 0.5 * np.trace(V.T @ S @ V @ W)


def riemannian_grad(S, V, d):
    """Riemannian gradient of the profiled loss on the Stiefel manifold."""
    w = 1.0 / (1.0 + d)
    W = np.diag(w)
    G = - S @ V @ W                      # Euclidean gradient of -0.5 tr(V^T S V W) up to constant
    sym = 0.5 * (V.T @ G + (V.T @ G).T)  # symmetrize
    return G - V @ sym                   # projection to tangent space


def reorth(V):
    """QR re-orthonormalization (keeps the span)."""
    Q, _ = np.linalg.qr(V)
    return Q


def procrustes(U, V):
    """Distance between column spaces (0 => same subspace)."""
    M = U.T @ V
    U1, _, V1 = np.linalg.svd(M, full_matrices=False)
    return np.linalg.norm(np.eye(M.shape[0]) - U1 @ V1)


def pairings(G, eigen_indices, d):
    """
    Good and bad pairings of the eigenvectors G[:, eigen_indices] with the
    column slots (prior precisions d).

    Good: larger eigenvalue (earlier index) in the slot with larger weight w.
    Bad: the reverse matching, so every pair of columns is an inversion.

    Returns
    -------
    V_good, V_bad : np.ndarray (p, R)
    """
    w = 1.0 / (1.0 + d)
    R = len(d)
    V_good = np.zeros((G.shape[0], R))
    for slot, col in enumerate(np.argsort(-w)):
        V_good[:, col] = G[:, eigen_indices[slot]]
    V_bad = np.zeros((G.shape[0], R))
    for slot, col in enumerate(np.argsort(w)):
        V_bad[:, col] = G[:, eigen_indices[slot]]
    return reorth(V_good), reorth(V_bad)


def swap_curvature(eigvals_pair, weights_pair):
    """
    Second derivative of the profiled loss at a stationary point along the
    rotation that mixes two columns holding eigenvalues (a, b) with weights
    (w_a, w_b):  L''(0) = (a - b)(w_a - w_b).
    It is negative exactly when the two columns are badly paired.
    """
    (a, b), (wa, wb) = eigvals_pair, weights_pair
    return (a - b) * (wa - wb)


class ProjectedGradientSolver:
    """
    Projected (Riemannian) gradient descent for the profiled loss:
    a gradient step in the tangent space followed by QR re-orthonormalization.

    Like the other solvers it is used as Solver(...).fit(...) and records a
    history; here the problem is the pair (S, d) and the iterate is V.
    fit() runs a batch of independent trials at once (one per perturbation).
    """

    def __init__(self, S, d, V_reference=None):
        self.S = S
        self.d = np.asarray(d, dtype=float)
        self.w = 1.0 / (1.0 + self.d)
        self.V_reference = V_reference
        self.V = None
        self.history = {}


    def _SV(self, V):
        # S @ V[t] for every trial t, as one matrix product
        T, p, R = V.shape
        flat = V.transpose(1, 0, 2).reshape(p, T * R)
        return (self.S @ flat).reshape(p, T, R).transpose(1, 0, 2)


    def _losses(self, V, SV):
        # 0.5 tr(S) + 0.5 R - 0.5 sum_k w_k v_k^T S v_k, for a stack of V
        quad = np.sum(V * SV, axis=1)
        return 0.5 * np.trace(self.S) + 0.5 * V.shape[2] - 0.5 * quad @ self.w


    def _distances(self, V):
        if self.V_reference is None:
            return np.full(V.shape[0], np.nan)
        return np.array([procrustes(Vt, self.V_reference) for Vt in V])


    def fit(self, start_V, trials=1, steps=500, eta=0.01, noise_scale=3e-3,
            rng=np.random, record=None):
        """
        Perturb start_V by Gaussian noise of size noise_scale (an independent
        draw per trial, in trial order), then take `steps` projected gradient
        steps of size eta.

        Parameters
        ----------
        record : array of step indices, optional
            Steps at which the distance to V_reference is computed
            (default: first and last). The loss is kept at every step.

        Returns
        -------
        history : dict with "loss" (trials, steps + 1), "distance"
        (trials, len(record)) and "record" (the step indices).
        """
        record = np.array([0, steps] if record is None else sorted(set(record) | {0, steps}))
        noise = noise_scale * rng.standard_normal((trials,) + start_V.shape)
        V = np.linalg.qr(start_V[None] + noise)[0]

        losses = np.empty((trials, steps + 1))
        dists = np.empty((trials, len(record)))
        r = 0
        for t in range(steps + 1):
            SV = self._SV(V)
            losses[:, t] = self._losses(V, SV)
            if r < len(record) and record[r] == t:
                dists[:, r] = self._distances(V)
                r += 1
            if t == steps:
                break
            G = -SV * self.w                            # Euclidean gradient, stacked
            VtG = np.transpose(V, (0, 2, 1)) @ G
            grad = G - V @ (0.5 * (VtG + np.transpose(VtG, (0, 2, 1))))
            V = np.linalg.qr(V - eta * grad)[0]

        self.V = V
        self.history = {"loss": losses, "distance": dists, "record": record}
        return self.history


    def analyze_solution(self, loss_tol=1e-6, dist_tol=1e-4, loss_optimum=None, step=None):
        """
        Per-trial report. `escaped` is the notebook's rule at the given step
        (default: the last one): the loss dropped by more than loss_tol and the
        distance to the reference (good pairing) dropped by more than dist_tol.
        With loss_optimum, `steps_to_optimum` is the first step at which the
        loss is within 1e-6 (relative) of it (-1 if never).
        """
        loss = self.history["loss"]
        dist = self.history["distance"]
        record = self.history["record"]
        k = len(record) - 1 if step is None else int(np.where(record == step)[0][0])
        s = record[k]
        report = {
            "loss_start": loss[:, 0],
            "loss_final": loss[:, s],
            "distance_start": dist[:, 0],
            "distance_final": dist[:, k],
            "escaped": (loss[:, s] < loss[:, 0] - loss_tol) & (dist[:, k] < dist[:, 0] - dist_tol),
        }
        if loss_optimum is not None:
            hit = loss <= loss_optimum + 1e-6 * abs(loss_optimum)
            report["steps_to_optimum"] = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        return report


def hessian_inertia(problem, U, V, tol=1e-8):
    """
    Inertia (#negative, #zero, #positive eigenvalues) of the Hessian of a
    LowRankMF objective at (U, V); |eigenvalue| <= tol counts as zero.
    """
    eigs = np.linalg.eigvalsh(problem.hessian(U, V))
    neg = int(np.sum(eigs < -tol))
    zero = int(np.sum(np.abs(eigs) <= tol))
    pos = int(np.sum(eigs > tol))
    return neg, zero, pos, eigs
