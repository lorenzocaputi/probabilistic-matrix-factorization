import time

import numpy as np

from pmf.problem import LowRankMF


class BaseSolver:
    """
    Bookkeeping shared by all solvers.

    A solver does NOT generate data. It only:
        - keeps the current iterate (U, V),
        - runs its updates in fit(), which returns a history dict,
        - reports on the final iterate in analyze_solution().

    Subclasses implement fit(). The history always has the keys "loss"
    (the quantity the solver tracks: MAP objective, ELBO, residual...) and
    "time" (seconds since the start of fit), one entry per iteration.
    """

    name = "base"

    def __init__(self, problem: LowRankMF):
        self.problem = problem
        self.U = None
        self.V = None
        self.history = {"loss": [], "time": []}
        self.loss_history = self.history["loss"]
        self.n_iter = 0


    def estimate(self):
        """
        Current estimate of the signal matrix (U V^T unless overridden).
        """
        return self.U @ self.V.T


    def factors(self):
        """
        Factors (A, B) with estimate() == A @ B.T, cheap to store per iteration.
        """
        return self.U, self.V


    def _reset_history(self):
        self.history = {"loss": [], "time": []}
        self.loss_history = self.history["loss"]
        self.n_iter = 0
        self._t0 = time.perf_counter()


    def _record(self, loss):
        self.history["loss"].append(float(loss))
        self.history["time"].append(time.perf_counter() - self._t0)
        self.n_iter += 1


    @staticmethod
    def _relative_change(prev, curr):
        return abs(prev - curr) / max(abs(prev), 1e-12)


    def analyze_solution(self):
        """
        Analyze the final solution (U, V) reached by the solver.

        Returns
        -------
        report : dict
            Dictionary with:
                - n_iter: number of iterations run
                - loss_final: MAP objective at (U, V) on the observed entries
                - grad_norm: Frobenius norm of the gradient at (U, V)
                - global_loss_Y: objective at the global minimizer for Y
                  (complete data only, NaN otherwise)
                - loss_gap_Y: loss_final - global_loss_Y
                - relative_loss_gap_Y: loss_gap_Y / |global_loss_Y|
                - loss_true_solution: objective at (U, V) w.r.t. Y_true
                - global_loss_Ytrue: objective at global minimizer for Y_true
                - loss_gap_Ytrue: loss_true_solution - global_loss_Ytrue
                - relative_loss_gap_Ytrue: loss_gap_Ytrue / |global_loss_Ytrue|
                - relative_signal_error: ||Y_true - Z||_F / ||Y_true||_F, where
                  Z = estimate() (the posterior mean for the Bayesian solvers)
        """
        if self.U is None or self.V is None:
            raise ValueError("The solver has not been run yet; U and V are None.")

        problem = self.problem
        U = self.U
        V = self.V
        nan = float("nan")

        # Objective and gradient on the observed entries of Y
        loss_final = problem.objective(U, V)
        grad_norm = problem.grad_norm(U, V)

        if problem.complete:
            _, _, global_loss_Y = problem.global_minimizer_Y()
            loss_gap_Y = loss_final - global_loss_Y
            relative_loss_gap_Y = loss_gap_Y / max(abs(global_loss_Y), 1e-12)
        else:
            global_loss_Y = loss_gap_Y = relative_loss_gap_Y = nan

        if problem.Y_true is not None:
            # Objective on true Y_true for the current solution
            loss_true_solution = problem.loss(problem.Y_true, U, V)
            _, _, global_loss_Ytrue = problem.global_minimizer_Ytrue()
            loss_gap_Ytrue = loss_true_solution - global_loss_Ytrue
            relative_loss_gap_Ytrue = loss_gap_Ytrue / max(abs(global_loss_Ytrue), 1e-12)
        else:
            loss_true_solution = global_loss_Ytrue = nan
            loss_gap_Ytrue = relative_loss_gap_Ytrue = nan

        relative_signal_error = problem.relative_error(self.estimate())

        report = {
            "n_iter": self.n_iter,
            "loss_final": loss_final,
            "grad_norm": grad_norm,  # are we close to a stationary point of the noisy Y?
            "global_loss_Y": global_loss_Y,
            "loss_gap_Y": loss_gap_Y,
            "relative_loss_gap_Y": relative_loss_gap_Y,  # are we at the global minimizer of the noisy Y?
            "loss_true_solution": loss_true_solution,
            "global_loss_Ytrue": global_loss_Ytrue,
            "loss_gap_Ytrue": loss_gap_Ytrue,
            "relative_loss_gap_Ytrue": relative_loss_gap_Ytrue,  # are we at the global minimizer of the true Y?
            "relative_signal_error": relative_signal_error,
        }
        return report
