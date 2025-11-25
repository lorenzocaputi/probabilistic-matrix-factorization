import numpy as np

class LowRankMF:
    """
    Low-rank matrix factorization with L2 regularization, solved by ALS.

    Given Y (n x p), we look for U (n x R), V (p x R) solving

        min_{U,V} 0.5 * ||Y - U V^T||_F^2 + 0.5 * lambda_reg * (||U||_F^2 + ||V||_F^2)

    using alternating least squares (ALS).
    """

    def __init__(self, Y, R, lambda_reg):
        """
        Parameters
        ----------
        Y : np.ndarray, shape (n, p)
            Data matrix to factorize.
        R : int
            Latent dimension (rank of the factorization).
        lambda_reg : float
            Regularization parameter lambda.
        """
        Y = np.asarray(Y)
        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array.")

        self.Y = Y
        self.n, self.p = Y.shape
        self.R = int(R)
        self.lambda_reg = float(lambda_reg)

        # Factors will be initialized only if needed inside fit_als
        self.U = None  # shape (n, R)
        self.V = None  # shape (p, R)

        self.loss_history = []

    def _init_factors(self):
        """Randomly initialize U and V with small Gaussian entries."""
        self.U = 0.01 * np.random.randn(self.n, self.R)
        self.V = 0.01 * np.random.randn(self.p, self.R)

    def loss(self, U=None, V=None):
        """
        Compute the objective value:

            L(U, V) = 0.5 * ||Y - U V^T||_F^2 + 0.5 * lambda_reg * (||U||_F^2 + ||V||_F^2)

        If U or V are None, use the current self.U, self.V.
        """
        if U is None:
            U = self.U
        if V is None:
            V = self.V

        if U is None or V is None:
            raise ValueError("U and V must be initialized before computing loss.")

        Y_hat = U @ V.T
        residual = self.Y - Y_hat

        data_term = 0.5 * np.linalg.norm(residual, ord="fro")**2
        reg_term = 0.5 * self.lambda_reg * (
            np.linalg.norm(U, ord="fro")**2 + np.linalg.norm(V, ord="fro")**2
        )

        return data_term + reg_term

    def fit_als(self, max_iter=50, verbose=False):
        """
        Run ALS for a fixed number of iterations.

        Updates:
            U = Y V (V^T V + lambda I_R)^{-1}
            V = Y^T U (U^T U + lambda I_R)^{-1}

        Parameters
        ----------
        max_iter : int
            Number of ALS iterations.
        verbose : bool
            If True, print loss at each iteration.
        """
        if self.U is None or self.V is None:
            self._init_factors()

        self.loss_history = []

        lam = self.lambda_reg
        R = self.R

        for t in range(max_iter):
            # --- Update U given V ---
            A_U = self.V.T @ self.V + lam * np.eye(R)   
            B_U = self.Y @ self.V                       
            self.U = np.linalg.solve(A_U, B_U.T).T      

            # --- Update V given U ---
            A_V = self.U.T @ self.U + lam * np.eye(R)   
            B_V = self.Y.T @ self.U                     
            self.V = np.linalg.solve(A_V, B_V.T).T      

            # --- Compute and store loss ---
            current_loss = self.loss()
            self.loss_history.append(current_loss)

            if verbose:
                print(f"Iteration {t+1}/{max_iter}, loss = {current_loss:.6f}")

    def reconstruct(self):
        """
        Return the reconstructed matrix Y_hat = U V^T
        using the current factors.
        """
        if self.U is None or self.V is None:
            raise ValueError("Model has not been fitted or initialized yet.")
        return self.U @ self.V.T
    


