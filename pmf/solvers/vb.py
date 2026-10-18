import numpy as np

from pmf.problem import LowRankMF
from pmf.solvers.base import BaseSolver


class VBSolver(BaseSolver):
    """
    Variational Bayes for PMF (Lim and Teh 2007), with the evidence lower
    bound (ELBO) tracked.

    Model:  y_ij ~ N(u_i^T v_j, s2) on the observed entries,
            u_i ~ N(0, diag(cu)),  v_j ~ N(0, diag(cv)).
    Mean field: q(U, V) = prod_i N(u_i | mu_i, Sigma_i) prod_j N(v_j | nu_j, Psi_j).

    One iteration updates q(U) given q(V), q(V) given q(U), then (optionally)
    the noise variance s2 and the prior variances cu, cv by maximizing the
    ELBO. Every step maximizes the ELBO over its block, so the ELBO never
    decreases. Learning cu, cv per column switches off unneeded columns
    (automatic relevance determination).

    The estimate of the signal is the posterior mean of U V^T under q,
    mu nu^T. U and V hold the posterior means.
    """

    name = "VB"

    def __init__(self, problem: LowRankMF, init_scale=0.1, U0=None, V0=None,
                 noise_var=1.0, prior_var=None, learn_noise=True, learn_prior=True):
        """
        Parameters
        ----------
        problem : LowRankMF
        init_scale : float, optional
            Scale of the random initial means.
        U0, V0 : np.ndarray, optional
            Initial posterior means.
        noise_var : float, optional
            Initial (or fixed) noise variance s2. 1.0 matches the MAP model.
        prior_var : float or array of length R, optional
            Initial (or fixed) prior variances. Default 1 / lambda, the prior
            whose MAP estimate is the problem's objective.
        learn_noise, learn_prior : bool
            Update s2 and (cu, cv) by maximizing the ELBO.
        """
        super().__init__(problem)
        R = problem.R
        if U0 is not None and V0 is not None:
            self.U = np.array(U0, dtype=float)
            self.V = np.array(V0, dtype=float)
        else:
            self.U, self.V = problem.sample_initial_UV(scale=init_scale)
        self.Sigma = np.tile(np.eye(R) * init_scale**2, (problem.n, 1, 1))
        self.Psi = np.tile(np.eye(R) * init_scale**2, (problem.p, 1, 1))
        if prior_var is None:
            prior_var = 1.0 / np.maximum(problem.lam, 1e-12)
        self.cu = np.broadcast_to(np.asarray(prior_var, dtype=float), (R,)).copy()
        self.cv = self.cu.copy()
        self.s2 = float(noise_var)
        self.learn_noise = learn_noise
        self.learn_prior = learn_prior
        self.history["elbo"] = []


    @staticmethod
    def _second_moments(M, C):
        """E[x x^T] = C + m m^T for every row, flattened to (rows, R*R)."""
        m, R = M.shape
        return (C + M[:, :, None] * M[:, None, :]).reshape(m, R * R)


    def _update_rows(self, W, WY, M_other, C_other, prior_var):
        """
        Optimal Gaussian factor for every row given the other side:
            Sigma_i = (diag(1/c) + sum_j W_ij E[v_j v_j^T] / s2)^{-1}
            mu_i    = Sigma_i sum_j W_ij y_ij nu_j / s2
        """
        R = M_other.shape[1]
        E2 = self._second_moments(M_other, C_other)
        precision = (W @ E2).reshape(W.shape[0], R, R) / self.s2 + np.diag(1.0 / prior_var)
        cov = np.linalg.inv(precision)
        cov = 0.5 * (cov + cov.transpose(0, 2, 1))
        mean = np.einsum("irk,ik->ir", cov, (WY @ M_other) / self.s2)
        return mean, cov


    def _expected_sq_error(self, W, Y):
        """Sum over observed entries of E_q[(y_ij - u_i^T v_j)^2]."""
        EU = self._second_moments(self.U, self.Sigma)
        EV = self._second_moments(self.V, self.Psi)
        cross = W * Y * (self.U @ self.V.T)
        quad = W * (EU @ EV.T)
        return np.sum(W * Y**2) - 2.0 * np.sum(cross) + np.sum(quad)


    def elbo(self):
        """
        Evidence lower bound
            E_q[log p(Y | U, V)] + E_q[log p(U)] + E_q[log p(V)] + H[q(U)] + H[q(V)].
        """
        problem = self.problem
        W = problem.weights
        n_obs = W.sum()
        R = problem.R

        lik = -0.5 * n_obs * np.log(2 * np.pi * self.s2) \
              - 0.5 * self._expected_sq_error(W, problem.Y) / self.s2

        def prior_and_entropy(M, C, c):
            m = M.shape[0]
            diag = np.diagonal(C, axis1=1, axis2=2)
            prior = -0.5 * m * np.sum(np.log(2 * np.pi * c)) \
                    - 0.5 * np.sum((M**2 + diag) / c)
            _, logdet = np.linalg.slogdet(C)
            entropy = 0.5 * np.sum(logdet) + 0.5 * m * R * (1.0 + np.log(2 * np.pi))
            return prior + entropy

        return lik + prior_and_entropy(self.U, self.Sigma, self.cu) \
                   + prior_and_entropy(self.V, self.Psi, self.cv)


    def fit(self, max_iter=200, tol=1e-8, verbose=False, callback=None):
        """
        Run coordinate ascent on the ELBO.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations.
        tol : float or None, optional
            Stop when the relative change of the ELBO falls below tol.
        verbose : bool
        callback : callable, optional
            Called as callback(solver, t) after every iteration.

        Returns
        -------
        history : dict with lists "loss" (the negative ELBO), "elbo", "time"
        and "noise_var".
        """
        problem = self.problem
        W = problem.weights
        WY = W * problem.Y
        n_obs = W.sum()
        self._reset_history()
        self.history["elbo"] = []
        self.history["noise_var"] = []
        prev = None

        for t in range(max_iter):
            self.U, self.Sigma = self._update_rows(W, WY, self.V, self.Psi, self.cu)
            self.V, self.Psi = self._update_rows(W.T, WY.T, self.U, self.Sigma, self.cv)

            if self.learn_noise:
                self.s2 = max(self._expected_sq_error(W, problem.Y) / n_obs, 1e-12)
            if self.learn_prior:
                self.cu = np.maximum(np.mean(self.U**2 + np.diagonal(self.Sigma, axis1=1, axis2=2), axis=0), 1e-12)
                self.cv = np.maximum(np.mean(self.V**2 + np.diagonal(self.Psi, axis1=1, axis2=2), axis=0), 1e-12)

            elbo = self.elbo()
            self._record(-elbo)
            self.history["elbo"].append(elbo)
            self.history["noise_var"].append(self.s2)

            if verbose:
                print(f"Iteration {t+1}/{max_iter}, ELBO = {elbo:.6f}, noise var = {self.s2:.4g}")
            if callback is not None:
                callback(self, t)
            if tol is not None and prev is not None and self._relative_change(prev, elbo) < tol:
                break
            prev = elbo

        return self.history
