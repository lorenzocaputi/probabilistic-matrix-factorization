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



