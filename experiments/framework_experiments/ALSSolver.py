import numpy as np
from LowRankMF import LowRankMF

class ALSSolver:
    """
    Alternating Least Squares (ALS) solver for a LowRankMF problem.

    This class does NOT generate data. It only:
        - keeps the current iterate U, V,
        - runs ALS updates,
        - records the loss trajectory.
    """

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
        self.problem = problem
        if V0 is not None:
            self.V = V0
        else:
            self.V = self.problem.sample_initial_V(scale=init_scale)
        self.U = None
        self.loss_history = []


    def fit(self, max_iter=50, tol=None, verbose=False):
        """
        Run ALS for a fixed number of iterations, with optional early stopping.

        Updates:
            U = Y V (V^T V + lambda I_R)^{-1}
            V = Y^T U (U^T U + lambda I_R)^{-1}

        Parameters
        ----------
        max_iter : int
            Maximum number of ALS iterations.
        tol : float or None, optional
            Relative tolerance on the loss for early stopping.
            If None, run exactly max_iter iterations.
        verbose : bool
            If True, print loss at each iteration.
        """
        Y = self.problem.Y
        lam = self.problem.lambda_reg
        R = self.problem.R

        self.loss_history = []
        prev_loss = None

        for t in range(max_iter):
            A_U = self.V.T @ self.V + lam * np.eye(R)
            B_U = Y @ self.V
            self.U = np.linalg.solve(A_U, B_U.T).T

            A_V = self.U.T @ self.U + lam * np.eye(R)
            B_V = Y.T @ self.U
            self.V = np.linalg.solve(A_V, B_V.T).T

            current_loss = self.problem.loss(self.problem.Y, self.U, self.V)
            self.loss_history.append(current_loss)

            if verbose:
                print(f"Iteration {t+1}/{max_iter}, loss = {current_loss:.6f}")

            if tol is not None and prev_loss is not None:
                rel_change = abs(prev_loss - current_loss) / max(prev_loss, 1e-12)
                if rel_change < tol:
                    break

            prev_loss = current_loss


    def analyze_solution(self):
        """
        Analyze the final solution (U, V) reached by ALS.

        Returns
        -------
        report : dict
            Dictionary with:
                - loss_final: objective at (U, V) w.r.t. Y
                - grad_norm: Frobenius norm of the gradient at (U, V) (w.r.t. Y)
                - global_loss_Y: objective at global minimizer for Y
                - loss_gap_Y: loss_final - global_loss_Y
                - relative_loss_gap_Y: loss_gap_Y / |global_loss_Y|
                - loss_true_solution: objective at (U, V) w.r.t. Y_true
                - global_loss_Ytrue: objective at global minimizer for Y_true
                - loss_gap_Ytrue: loss_true_solution - global_loss_Ytrue
                - relative_loss_gap_Ytrue: loss_gap_Ytrue / |global_loss_Ytrue|
                - relative_signal_error: ||Y_true - U V^T||_F / ||Y_true||_F
        """
        if self.U is None or self.V is None:
            raise ValueError("ALS has not been run yet; U and V are None.")

        problem = self.problem
        U = self.U
        V = self.V

        # Objective and gradient on noisy Y
        loss_final = problem.loss(problem.Y, U, V)
        grad_norm = problem.grad_norm(U, V)

        U_star_Y, V_star_Y, global_loss_Y = problem.global_minimizer_Y()
        loss_gap_Y = loss_final - global_loss_Y
        relative_loss_gap_Y = loss_gap_Y / max(abs(global_loss_Y), 1e-12)

        # Objective on true Y_true for the current solution
        loss_true_solution = problem.loss(problem.Y_true, U, V)

        # Global minimizer on Y_true
        U_star_true, V_star_true, global_loss_Ytrue = problem.global_minimizer_Ytrue()
        loss_gap_Ytrue = loss_true_solution - global_loss_Ytrue
        relative_loss_gap_Ytrue = loss_gap_Ytrue / max(abs(global_loss_Ytrue), 1e-12)

        relative_signal_error = problem.relative_signal_error(U, V)

        report = {
            "loss_final": loss_final, 
            "grad_norm": grad_norm, # are we close to a stationary point of the noisy Y?
            "global_loss_Y": global_loss_Y,
            "loss_gap_Y": loss_gap_Y,
            "relative_loss_gap_Y": relative_loss_gap_Y, # are we at the global minimizer of the noisy Y?
            "loss_true_solution": loss_true_solution,
            "global_loss_Ytrue": global_loss_Ytrue,
            "loss_gap_Ytrue": loss_gap_Ytrue,
            "relative_loss_gap_Ytrue": relative_loss_gap_Ytrue, # are we at the global minimizer of the true Y?
            "relative_signal_error": relative_signal_error,
        }
        return report





