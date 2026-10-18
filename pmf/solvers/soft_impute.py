import numpy as np

from pmf.problem import LowRankMF
from pmf.solvers.base import BaseSolver


class SoftImputeSolver(BaseSolver):
    """
    Soft-impute (Mazumder, Hastie and Tibshirani 2010) in factored form.

    Each iteration fills the missing entries with the current estimate and
    applies the closed-form MAP step to the filled matrix:

        Y_fill = mask * Y + (1 - mask) * U V^T
        (U, V) = soft-thresholded SVD of Y_fill (rank R, prior Lambda).

    The filled-matrix objective majorizes the masked MAP objective and
    touches it at the current point, so the masked objective never
    increases. With a scalar lambda and R at least the rank of the solution,
    the fixed point solves the nuclear-norm problem

        min_Z 0.5 * ||mask * (Y - Z)||_F^2 + lambda * ||Z||_*.

    On complete data the first step already returns the closed form.
    """

    name = "soft-impute"

    def __init__(self, problem: LowRankMF, fill="zero", U0=None, V0=None):
        """
        Parameters
        ----------
        problem : LowRankMF
        fill : {"zero", "mean"}, optional
            Initial value of the missing entries when no warm start is given
            ("mean" uses the mean of the observed entries).
        U0, V0 : np.ndarray, optional
            Warm start; the first fill then uses U0 V0^T.
        """
        super().__init__(problem)
        if U0 is not None and V0 is not None:
            self.U = np.array(U0, dtype=float)
            self.V = np.array(V0, dtype=float)
            self._Z = self.U @ self.V.T
        else:
            self.U = None
            self.V = None
            if fill == "mean" and not problem.complete:
                value = problem.Y[problem.mask].mean()
            elif fill in ("zero", "mean"):
                value = 0.0
            else:
                raise ValueError('fill must be "zero" or "mean".')
            self._Z = np.full((problem.n, problem.p), value)


    def filled_matrix(self):
        """
        Observed entries of Y with the current estimate in the missing ones.
        """
        if self.problem.complete:
            return self.problem.Y.copy()
        return np.where(self.problem.mask, self.problem.Y, self._Z)


    def estimate(self):
        return self._Z


    def fit(self, max_iter=100, tol=1e-6, verbose=False, callback=None):
        """
        Run soft-impute.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        tol : float or None, optional
            Stop when ||Z_new - Z_old||_F / ||Z_old||_F < tol (Z = U V^T).
        verbose : bool
        callback : callable, optional
            Called as callback(solver, t) after every iteration.

        Returns
        -------
        history : dict with lists "loss", "time" and "rel_change".
        """
        problem = self.problem
        self._reset_history()
        self.history["rel_change"] = []

        for t in range(max_iter):
            Y_fill = self.filled_matrix()
            self.U, self.V, _ = problem._global_minimizer_matrix(Y_fill)
            Z_new = self.U @ self.V.T

            rel = np.linalg.norm(Z_new - self._Z) / max(np.linalg.norm(self._Z), 1e-12)
            self._Z = Z_new

            loss = problem.objective(self.U, self.V)
            self._record(loss)
            self.history["rel_change"].append(rel)

            if verbose:
                print(f"Iteration {t+1}/{max_iter}, loss = {loss:.6f}, rel_change = {rel:.3e}")
            if callback is not None:
                callback(self, t)

            if problem.complete or (tol is not None and rel < tol):
                break

        return self.history


def soft_impute(M_obs, Omega, lam, max_iter=200, tol=1e-4, rank_max=None, verbose=False):
    """
    Matrix-level soft-impute, for callers that work with plain arrays.

    Parameters
    ----------
    M_obs : np.ndarray (n, p)
        Matrix with arbitrary values in the missing entries.
    Omega : np.ndarray of bool (n, p)
        True where M_obs is observed.
    lam : float
        Singular value shrinkage (larger means lower rank and smoother).
    max_iter, tol : stopping rule (see SoftImputeSolver.fit).
    rank_max : int, optional
        Keep at most this many singular values (None keeps all).
    verbose : bool

    Returns
    -------
    X : np.ndarray (n, p)
        Observed entries of M_obs, with the low-rank estimate elsewhere.
    """
    R = min(M_obs.shape) if rank_max is None else int(rank_max)
    problem = LowRankMF.from_matrix(M_obs, R=R, lambda_reg=lam, mask=Omega)
    solver = SoftImputeSolver(problem, fill="mean")
    solver.fit(max_iter=max_iter, tol=tol, verbose=verbose)
    return solver.filled_matrix()
