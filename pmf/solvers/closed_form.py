import numpy as np

from pmf.linalg import top_svd
from pmf.problem import LowRankMF
from pmf.solvers.base import BaseSolver


class ClosedFormSolver(BaseSolver):
    """
    MAP estimate in closed form (part 1 of the notes).

    For complete data the global minimizer is the soft-thresholded SVD of Y:

        U* = F_R diag(sqrt((sigma_i - lambda_i)_+)),
        V* = G_R diag(sqrt((sigma_i - lambda_i)_+)),

    so fit() takes one step and needs no initialization.

    With missing entries there is no closed form. fit() then returns the
    one-step spectral estimate: the same formula applied to the zero-filled
    matrix rescaled by 1 / (observed fraction), which is unbiased for Y.
    """

    name = "closed form"

    def __init__(self, problem: LowRankMF, rescale=True):
        """
        Parameters
        ----------
        problem : LowRankMF
        rescale : bool, optional
            With missing entries, divide the zero-filled matrix by the
            observed fraction before thresholding.
        """
        super().__init__(problem)
        self.rescale = rescale


    def fit(self, verbose=False, callback=None, **unused):
        """
        Compute the estimate. Extra keyword arguments (max_iter, tol) are
        accepted and ignored so that all solvers share one calling convention.

        Returns
        -------
        history : dict with one entry in "loss" and "time".
        """
        problem = self.problem
        self._reset_history()

        if problem.complete:
            self.U, self.V, _ = problem.global_minimizer_Y()
        else:
            M = np.where(problem.mask, problem.Y, 0.0)
            if self.rescale:
                M = M / max(problem.mask.mean(), 1e-12)
            self.U, self.V, _ = problem._global_minimizer_matrix(M)

        loss = problem.objective(self.U, self.V)
        self._record(loss)
        if verbose:
            print(f"closed form: loss = {loss:.6f}")
        if callback is not None:
            callback(self, 0)
        return self.history
