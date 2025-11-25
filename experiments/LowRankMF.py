import numpy as np

class LowRankMF:
    """
    Low-rank matrix factorization problem with synthetic data generation.

    This class represents a *problem instance*:

        - It generates a synthetic low-rank matrix Y_true of size (n x p),
          with given true_rank and singular values.
        - It adds Gaussian noise to obtain Y.
        - It provides the objective (loss) for given U, V.

    The optimization (ALS) is handled by a separate ALSSolver.
    """

    def __init__(self,
                 n,
                 p,
                 true_rank,
                 R,
                 lambda_reg,
                 singular_values=None,
                 noise_level=0.0,
                 seed=None):
        """
        Parameters
        ----------
        n : int
            Number of rows of Y.
        p : int
            Number of columns of Y.
        true_rank : int
            Rank of the *true* underlying low-rank matrix Y_true.
        R : int
            Latent dimension used in the factorization model (rank of U, V).
        lambda_reg : float
            Regularization parameter lambda.
        singular_values : array-like of length true_rank, optional
            Singular values for Y_true. If None, use a simple decreasing sequence.
        noise_level : float, optional
            Standard deviation of Gaussian noise added to Y_true.
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.n = int(n)
        self.p = int(p)
        self.true_rank = int(true_rank)
        self.R = int(R)
        self.lambda_reg = float(lambda_reg)
        self.noise_level = float(noise_level)
        self.seed = seed

        if singular_values is None:
            # default: [true_rank, ..., 2, 1]
            self.singular_values = np.arange(self.true_rank, 0, -1, dtype=float)
        else:
            sv = np.asarray(singular_values, dtype=float)
            if sv.shape[0] != self.true_rank:
                raise ValueError("singular_values must have length true_rank.")
            self.singular_values = sv

        self._generate_data()

    def _generate_data(self):
        """
        Sample U_true, V_true with orthonormal columns and construct

            Y_true = U_true diag(singular_values) V_true^T,
            Y = Y_true + noise_level * N(0,1)^{n x p}.
        """
        # U_true: (n x true_rank), orthonormal columns
        U_rand = np.random.randn(self.n, self.true_rank)
        self.U_true, _ = np.linalg.qr(U_rand)

        # V_true: (p x true_rank), orthonormal columns
        V_rand = np.random.randn(self.p, self.true_rank)
        self.V_true, _ = np.linalg.qr(V_rand)

        S = np.diag(self.singular_values)
        self.Y_true = self.U_true @ S @ self.V_true.T

        if self.noise_level > 0.0:
            noise = self.noise_level * np.random.randn(self.n, self.p)
            self.Y = self.Y_true + noise
        else:
            self.Y = self.Y_true.copy()

    def sample_initial_V(self, scale=0.01):
        """
        Sample a random starting point V0 with small Gaussian entries.
        """
        V0 = scale * np.random.randn(self.p, self.R)
        return V0


    def loss(self, M, U, V):
        """
        Compute the regularized objective for a given matrix M:

            0.5 * ||M - U V^T||_F^2 + 0.5 * lambda_reg * (||U||_F^2 + ||V||_F^2)
        """
        Y_hat = U @ V.T
        residual = M - Y_hat

        data_term = 0.5 * np.linalg.norm(residual, ord="fro")**2
        reg_term = 0.5 * self.lambda_reg * (
            np.linalg.norm(U, ord="fro")**2 + np.linalg.norm(V, ord="fro")**2
        )

        return data_term + reg_term


    def reconstruct(self, U, V):
        """
        Return the reconstructed matrix Y_hat = U V^T
        for given factors U, V.
        """
        return U @ V.T
    
    def relative_signal_error(self, U, V):
        """
        Relative Frobenius error with respect to Y_true:
            ||Y_true - U V^T||_F / ||Y_true||_F
        """
        Y_hat = U @ V.T
        num = np.linalg.norm(self.Y_true - Y_hat, ord="fro")
        den = np.linalg.norm(self.Y_true, ord="fro")
        return num / max(den, 1e-12)
    
    def grad(self, U, V):
        """
        Gradient of the regularized objective with respect to U and V.
        """
        Y_hat = U @ V.T
        residual = Y_hat - self.Y

        grad_U = residual @ V + self.lambda_reg * U
        grad_V = residual.T @ U + self.lambda_reg * V

        return grad_U, grad_V

    def grad_norm(self, U, V):
        """
        Frobenius norm of the full gradient (U and V parts together).
        """
        grad_U, grad_V = self.grad(U, V)
        norm_U = np.linalg.norm(grad_U, ord="fro")
        norm_V = np.linalg.norm(grad_V, ord="fro")
        return np.sqrt(norm_U**2 + norm_V**2)
    
    def _global_minimizer_matrix(self, M):
        """
        Global minimizer of the regularized objective
            0.5 * ||M - U V^T||_F^2 + 0.5 * lambda_reg * (||U||_F^2 + ||V||_F^2)
        for a given matrix M, via soft-thresholded SVD.
        """
        U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)

        k = min(self.R, s.shape[0])
        U_R = U_svd[:, :k]
        V_R = Vt_svd[:k, :].T
        s_R = s[:k]

        gamma = np.sqrt(np.maximum(s_R - self.lambda_reg, 0.0))

        # Build factors U_star, V_star
        U_star = U_R * gamma[np.newaxis, :]
        V_star = V_R * gamma[np.newaxis, :]

        if k < self.R:
            n_pad = self.R - k
            U_pad = np.zeros((self.n, n_pad))
            V_pad = np.zeros((self.p, n_pad))
            U_star = np.concatenate([U_star, U_pad], axis=1)
            V_star = np.concatenate([V_star, V_pad], axis=1)

        # Use the existing objective function (DRY)
        global_loss = self.loss(M, U_star, V_star)
        return U_star, V_star, global_loss
    
    def global_minimizer_Y(self):
        """
        Global minimizer of the regularized objective using the noisy matrix Y.
        """
        return self._global_minimizer_matrix(self.Y)

    def global_minimizer_Ytrue(self):
        """
        Global minimizer of the regularized objective using the true matrix Y_true.
        """
        return self._global_minimizer_matrix(self.Y_true)



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





