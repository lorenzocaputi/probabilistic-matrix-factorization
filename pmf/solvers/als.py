import numpy as np

from pmf.problem import LowRankMF
from pmf.solvers.base import BaseSolver


class ALSSolver(BaseSolver):
    """
    Alternating Least Squares (ALS) solver for a LowRankMF problem.

    Each half-step is an exact ridge regression, so the MAP objective never
    increases. With complete data the column span of V follows orthogonal
    (subspace) iteration on S = Y^T Y; part 2 of the notes analyses these
    dynamics.
    """

    name = "ALS"

    def __init__(self, problem: LowRankMF, init_scale=0.01, V0=None):
        """
        Parameters
        ----------
        problem : LowRankMF
            The low-rank matrix factorization problem to solve.
        init_scale : float, optional
            Scale for the random initialization of V when V0 is not provided.
        V0 : np.ndarray, optional
            Custom initialization for V. If provided, must have shape (p, R).
        """
        super().__init__(problem)
        if V0 is not None:
            self.V = np.array(V0, dtype=float)
        else:
            self.V = self.problem.sample_initial_V(scale=init_scale)
        self.U = None


    def _update_complete(self, Y, lam):
        R = self.problem.R
        A_U = self.V.T @ self.V + np.diag(lam)
        B_U = Y @ self.V
        self.U = np.linalg.solve(A_U, B_U.T).T

        A_V = self.U.T @ self.U + np.diag(lam)
        B_V = Y.T @ self.U
        self.V = np.linalg.solve(A_V, B_V.T).T


    @staticmethod
    def _ridge_rows(W, WY, F, lam):
        """
        Row-wise ridge regressions with observation weights W:
            row i solves (sum_j W_ij f_j f_j^T + Lambda) x_i = sum_j W_ij y_ij f_j.
        """
        m, R = F.shape
        FF = (F[:, :, None] * F[:, None, :]).reshape(m, R * R)
        A = (W @ FF).reshape(W.shape[0], R, R) + np.diag(lam)
        B = WY @ F
        return np.linalg.solve(A, B[:, :, None])[:, :, 0]


    def _update_masked(self, W, WY, lam):
        self.U = self._ridge_rows(W, WY, self.V, lam)
        self.V = self._ridge_rows(W.T, WY.T, self.U, lam)


    def fit(self, max_iter=50, tol=None, verbose=False, callback=None):
        """
        Run ALS for a fixed number of iterations, with optional early stopping.

        Updates (complete data):
            U = Y V (V^T V + Lambda)^{-1}
            V = Y^T U (U^T U + Lambda)^{-1}
        With a mask, each row of U (and of V) solves its own ridge problem on
        its observed entries.

        Parameters
        ----------
        max_iter : int
            Maximum number of ALS iterations.
        tol : float or None, optional
            Relative tolerance on the loss for early stopping.
            If None, run exactly max_iter iterations.
        verbose : bool
            If True, print loss at each iteration.
        callback : callable, optional
            Called as callback(solver, t) after every iteration.

        Returns
        -------
        history : dict with lists "loss" and "time".
        """
        problem = self.problem
        Y = problem.Y
        lam = problem.lam
        if not problem.complete:
            W = problem.weights
            WY = W * Y

        self._reset_history()
        prev_loss = None

        for t in range(max_iter):
            if problem.complete:
                self._update_complete(Y, lam)
            else:
                self._update_masked(W, WY, lam)

            current_loss = problem.objective(self.U, self.V)
            self._record(current_loss)

            if verbose:
                print(f"Iteration {t+1}/{max_iter}, loss = {current_loss:.6f}")
            if callback is not None:
                callback(self, t)

            if tol is not None and prev_loss is not None:
                if self._relative_change(prev_loss, current_loss) < tol:
                    break

            prev_loss = current_loss

        return self.history
