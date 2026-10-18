import numpy as np

from pmf.linalg import balanced_factors, soft_threshold_svd
from pmf.problem import LowRankMF
from pmf.solvers.base import BaseSolver


class SVTSolver(BaseSolver):
    """
    Singular value thresholding (Cai, Candès and Shen 2010).

    Iterates on a dual matrix D:

        X_k = shrink_tau(D_{k-1})                      (soft-threshold the SVD)
        D_k = D_{k-1} + delta * mask * (Y - X_k)

    which solves  min tau ||X||_* + 0.5 ||X||_F^2  subject to  X = Y on the
    observed entries. This is not the MAP objective: SVT interpolates the
    observed entries, so with noisy data it is compared on recovery error,
    not on the MAP loss.

    rank_max (our addition) keeps only the leading singular values, which
    lets SVT be compared with the rank-R solvers on MovieLens.
    """

    name = "SVT"

    def __init__(self, problem: LowRankMF, tau=None, delta=None, rank_max=None):
        """
        Parameters
        ----------
        problem : LowRankMF
        tau : float, optional
            Threshold. Default 5 * sqrt(n p), as in the paper.
        delta : float, optional
            Step size. Default 1.2 / (observed fraction), as in the paper.
        rank_max : int, optional
            Cap on the rank of X (None = no cap).
        """
        super().__init__(problem)
        n, p = problem.n, problem.p
        frac = 1.0 if problem.complete else problem.mask.mean()
        self.tau = 5.0 * np.sqrt(n * p) if tau is None else float(tau)
        self.delta = 1.2 / frac if delta is None else float(delta)
        self.rank_max = rank_max
        self._F = np.zeros((n, 0))
        self._s = np.zeros(0)
        self._Gt = np.zeros((0, p))


    def estimate(self):
        return (self._F * self._s) @ self._Gt


    def factors(self):
        root = np.sqrt(self._s)
        return self._F * root, self._Gt.T * root


    def fit(self, max_iter=500, tol=1e-4, verbose=False, callback=None):
        """
        Run SVT.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        tol : float or None, optional
            Stop when ||mask * (X - Y)||_F / ||mask * Y||_F < tol (the paper's rule).
        verbose : bool
        callback : callable, optional
            Called as callback(solver, t) after every iteration.

        Returns
        -------
        history : dict with lists "loss" (the relative residual on the
        observed entries), "time" and "rank".
        """
        problem = self.problem
        W = problem.weights
        PY = W * problem.Y
        norm_PY = np.linalg.norm(PY)
        self._reset_history()
        self.history["rank"] = []

        # Kick start (section 5.1.1 of the paper): skip the first iterations
        # in which X would be zero.
        k0 = int(np.ceil(self.tau / (self.delta * np.linalg.norm(PY, ord=2))))
        D = k0 * self.delta * PY

        for t in range(max_iter):
            self._F, self._s, self._Gt = soft_threshold_svd(D, self.tau, rank=self.rank_max)
            X = self.estimate()
            residual = W * (problem.Y - X)
            rel = np.linalg.norm(residual) / max(norm_PY, 1e-12)
            self._record(rel)
            self.history["rank"].append(len(self._s))

            if verbose:
                print(f"Iteration {t+1}/{max_iter}, residual = {rel:.3e}, rank = {len(self._s)}")
            if callback is not None:
                callback(self, t)
            if tol is not None and rel < tol:
                break

            D = D + self.delta * residual

        # factors padded or truncated to R columns, for analyze_solution()
        self.U, self.V = balanced_factors(self._F, self._s, self._Gt, problem.R)
        return self.history
