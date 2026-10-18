import numpy as np
from scipy.stats import wishart

from pmf.problem import LowRankMF
from pmf.solvers.base import BaseSolver


class GibbsSolver(BaseSolver):
    """
    Bayesian PMF by Gibbs sampling (Salakhutdinov and Mnih 2008).

    Model:  y_ij ~ N(u_i^T v_j, 1/alpha) on the observed entries,
            u_i ~ N(mu_U, Lambda_U^{-1}),  v_j ~ N(mu_V, Lambda_V^{-1}),
            (mu_U, Lambda_U), (mu_V, Lambda_V) ~ Gaussian-Wishart(mu0, beta0, W0, nu0).

    One sweep samples the hyperparameters of U, every row of U, the
    hyperparameters of V, then every row of V. The rows of U are independent
    given V (and vice versa), so each block is sampled in one batched step.

    The estimate is the running average of U V^T over the samples drawn
    after burn_in (the current sample before that). U and V hold the last sample.
    """

    name = "Gibbs"

    def __init__(self, problem: LowRankMF, alpha=2.0, init_scale=0.1, U0=None, V0=None,
                 beta0=2.0, nu0=None, burn_in=50):
        """
        Parameters
        ----------
        problem : LowRankMF
        alpha : float, optional
            Observation precision (the paper uses 2 for ratings).
        init_scale : float, optional
            Scale of the random initial sample.
        U0, V0 : np.ndarray, optional
            Initial sample (the paper starts from the MAP solution).
        beta0, nu0 : hyperprior parameters; mu0 = 0 and W0 = I as in the paper,
            nu0 defaults to R.
        burn_in : int
            Number of sweeps discarded before averaging.
        """
        super().__init__(problem)
        R = problem.R
        self.rng = problem.rng
        if U0 is not None and V0 is not None:
            self.U = np.array(U0, dtype=float)
            self.V = np.array(V0, dtype=float)
        else:
            self.U, self.V = problem.sample_initial_UV(scale=init_scale)
        self.alpha = float(alpha)
        self.beta0 = float(beta0)
        self.nu0 = float(R if nu0 is None else nu0)
        self.mu0 = np.zeros(R)
        self.W0_inv = np.eye(R)
        self.burn_in = int(burn_in)
        self._Z_sum = np.zeros((problem.n, problem.p))
        self._n_samples = 0


    def estimate(self):
        if self._n_samples == 0:
            return self.U @ self.V.T
        return self._Z_sum / self._n_samples


    def _sample_hyper(self, F):
        """Sample (mu, Lambda) from the Gaussian-Wishart posterior given the rows of F."""
        m, R = F.shape
        mean = F.mean(axis=0)
        S = (F - mean).T @ (F - mean) / m
        beta_n = self.beta0 + m
        nu_n = self.nu0 + m
        mu_n = (self.beta0 * self.mu0 + m * mean) / beta_n
        diff = (self.mu0 - mean)[:, None]
        Wn_inv = self.W0_inv + m * S + (self.beta0 * m / beta_n) * (diff @ diff.T)
        Wn = np.linalg.inv(Wn_inv)
        Wn = 0.5 * (Wn + Wn.T)
        Lam = np.atleast_2d(wishart(df=nu_n, scale=Wn).rvs(random_state=self.rng))
        cov = np.linalg.inv(beta_n * Lam)
        mu = self.rng.multivariate_normal(mu_n, 0.5 * (cov + cov.T))
        return mu, Lam


    def _sample_rows(self, W, WY, F_other, mu, Lam):
        """
        Sample every row x_i from N(m_i, P_i^{-1}) with
            P_i = Lambda + alpha sum_j W_ij f_j f_j^T,
            m_i = P_i^{-1} (alpha sum_j W_ij y_ij f_j + Lambda mu).
        """
        m_other, R = F_other.shape
        FF = (F_other[:, :, None] * F_other[:, None, :]).reshape(m_other, R * R)
        P = self.alpha * (W @ FF).reshape(W.shape[0], R, R) + Lam
        b = self.alpha * (WY @ F_other) + Lam @ mu
        L = np.linalg.cholesky(P)
        mean = np.linalg.solve(P, b[:, :, None])[:, :, 0]
        z = self.rng.standard_normal(b.shape)
        # x = L^{-T} z has covariance P^{-1}
        noise = np.linalg.solve(L.transpose(0, 2, 1), z[:, :, None])[:, :, 0]
        return mean + noise


    def fit(self, max_iter=200, tol=None, verbose=False, callback=None):
        """
        Run max_iter Gibbs sweeps (tol is accepted for a common calling
        convention and ignored: a sampler has no convergence tolerance).

        Returns
        -------
        history : dict with lists "loss" (RMSE of the running estimate on the
        observed entries) and "time".
        """
        problem = self.problem
        W = problem.weights
        WY = W * problem.Y
        n_obs = W.sum()
        self._reset_history()

        for t in range(max_iter):
            mu_U, Lam_U = self._sample_hyper(self.U)
            self.U = self._sample_rows(W, WY, self.V, mu_U, Lam_U)
            mu_V, Lam_V = self._sample_hyper(self.V)
            self.V = self._sample_rows(W.T, WY.T, self.U, mu_V, Lam_V)

            if t >= self.burn_in:
                self._Z_sum += self.U @ self.V.T
                self._n_samples += 1

            Z = self.estimate()
            train_rmse = np.sqrt(np.sum(W * (problem.Y - Z) ** 2) / n_obs)
            self._record(train_rmse)

            if verbose:
                print(f"Sweep {t+1}/{max_iter}, train RMSE = {train_rmse:.4f}")
            if callback is not None:
                callback(self, t)

        return self.history
