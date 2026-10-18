import numpy as np

from pmf.problem import LowRankMF
from pmf.solvers.base import BaseSolver


class GradientDescentSolver(BaseSolver):
    """
    Gradient descent on (U, V) jointly, with step-size control.

    Each iteration tries a step slightly longer than the last accepted one
    and backtracks (Armijo rule) until

        L(U - eta gU, V - eta gV) <= L(U, V) - c * eta * ||(gU, gV)||^2,

    so the MAP objective decreases monotonically without a hand-tuned step.
    Along the gradient direction the change in L is a quartic polynomial in
    eta; its coefficients are computed once per iteration, which makes the
    test exact near the optimum (where a direct difference of two loss
    values is lost to rounding) and makes backtracking free.
    """

    name = "gradient descent"

    def __init__(self, problem: LowRankMF, init_scale=0.1, U0=None, V0=None,
                 step_size=None, armijo=1e-4, shrink=0.5, grow=1.5):
        """
        Parameters
        ----------
        problem : LowRankMF
        init_scale : float, optional
            Scale of the random initialization when U0, V0 are not given.
            (The origin is a stationary point, so a tiny scale slows the start.)
        U0, V0 : np.ndarray, optional
            Custom initialization, shapes (n, R) and (p, R).
        step_size : float, optional
            Initial step. Default 1 / (sigma_1(Y) + max lambda).
        armijo : float
            Sufficient-decrease constant c.
        shrink, grow : float
            Backtracking factor and per-iteration growth of the trial step.
        """
        super().__init__(problem)
        if U0 is not None and V0 is not None:
            self.U = np.array(U0, dtype=float)
            self.V = np.array(V0, dtype=float)
        else:
            self.U, self.V = problem.sample_initial_UV(scale=init_scale)
        if step_size is None:
            sigma_1 = np.linalg.norm(problem.Y, ord=2)
            step_size = 1.0 / (sigma_1 + np.max(problem.lam))
        self.step_size = float(step_size)
        self.armijo = armijo
        self.shrink = shrink
        self.grow = grow


    def _loss_change_polynomial(self, gU, gV):
        """
        Exact change L(U - eta gU, V - eta gV) - L(U, V) as a function of eta.

        With D = mask * (U V^T - Y), A = -mask * (gU V^T + U gV^T) and
        B = mask * (gU gV^T), the new residual is D + eta A + eta^2 B, so the
        change is a quartic in eta.
        """
        problem = self.problem
        U, V, lam = self.U, self.V, problem.lam
        D = U @ V.T - problem.Y
        A = -(gU @ V.T + U @ gV.T)
        B = gU @ gV.T
        if not problem.complete:
            D, A, B = (np.where(problem.mask, M, 0.0) for M in (D, A, B))
        c1 = np.sum(D * A) - np.sum(lam * (np.sum(U * gU, axis=0) + np.sum(V * gV, axis=0)))
        c2 = 0.5 * np.sum(A * A) + np.sum(D * B) \
             + 0.5 * np.sum(lam * (np.sum(gU**2, axis=0) + np.sum(gV**2, axis=0)))
        c3 = np.sum(A * B)
        c4 = 0.5 * np.sum(B * B)
        return lambda eta: eta * (c1 + eta * (c2 + eta * (c3 + eta * c4)))


    def fit(self, max_iter=1000, tol=None, grad_tol=1e-10, verbose=False, callback=None):
        """
        Run gradient descent.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        tol : float or None, optional
            Relative tolerance on the loss for early stopping.
        grad_tol : float, optional
            Stop when the gradient norm falls below this value.
        verbose : bool
        callback : callable, optional
            Called as callback(solver, t) after every iteration.

        Returns
        -------
        history : dict with lists "loss", "time" and "step".
        """
        problem = self.problem
        self._reset_history()
        self.history["step"] = []

        loss = problem.objective(self.U, self.V)
        eta = self.step_size

        for t in range(max_iter):
            gU, gV = problem.grad(self.U, self.V)
            g2 = np.sum(gU**2) + np.sum(gV**2)
            if np.sqrt(g2) < grad_tol:
                break

            change = self._loss_change_polynomial(gU, gV)
            eta *= self.grow
            while change(eta) > -self.armijo * eta * g2 and eta > 1e-16:
                eta *= self.shrink

            self.U = self.U - eta * gU
            self.V = self.V - eta * gV
            prev_loss, loss = loss, problem.objective(self.U, self.V)
            self._record(loss)
            self.history["step"].append(eta)

            if verbose:
                print(f"Iteration {t+1}/{max_iter}, loss = {loss:.6f}, step = {eta:.3e}")
            if callback is not None:
                callback(self, t)

            if tol is not None and self._relative_change(prev_loss, loss) < tol:
                break

        self.step_size = eta
        return self.history
