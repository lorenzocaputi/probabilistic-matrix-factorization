import numpy as np

from pmf.linalg import top_svd


class LowRankMF:
    """
    Low-rank matrix factorization problem (MAP estimation in PMF).

    This class represents a *problem instance*:

        - It holds a data matrix Y of size (n x p), either generated here
          (synthetic low-rank signal plus Gaussian noise) or supplied with
          `LowRankMF.from_matrix`.
        - It optionally holds an observation mask (True = observed) for
          matrix completion.
        - It provides the MAP objective, its gradient and Hessian, and the
          closed-form global minimizer for complete data.

    The objective is

        L(U, V) = 0.5 * || mask * (Y - U V^T) ||_F^2
                  + 0.5 * ( tr(U Lambda U^T) + tr(V Lambda V^T) ),

    with Lambda = lambda_reg * I_R when lambda_reg is a scalar and
    Lambda = diag(lambda_reg) when it is a vector of length R.

    The optimization is handled by separate solver objects (pmf.solvers).
    """


    def __init__(self,
                 n,
                 p,
                 true_rank,
                 R,
                 lambda_reg,
                 singular_values=None,
                 noise_level=0.0,
                 seed=None,
                 missing_frac=0.0):
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
        lambda_reg : float or array-like of length R
            Prior precision. A scalar gives the isotropic prior of part 1 of
            the notes; a vector gives the diagonal prior Lambda of part 2.
        singular_values : array-like of length true_rank, optional
            Singular values for Y_true. If None, use a simple decreasing sequence.
        noise_level : float, optional
            Standard deviation of Gaussian noise added to Y_true.
        seed : int, optional
            Random seed. All randomness of the instance (data, mask and the
            initializations sampled from it) comes from one generator.
        missing_frac : float, optional
            Fraction of entries hidden uniformly at random (0 = complete data).
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.n = int(n)
        self.p = int(p)
        self.true_rank = int(true_rank)
        self.R = int(R)
        self._set_lambda(lambda_reg)
        self.noise_level = float(noise_level)
        self.missing_frac = float(missing_frac)

        if singular_values is None:
            # default: [true_rank, ..., 2, 1]
            self.singular_values = np.arange(self.true_rank, 0, -1, dtype=float)
        else:
            sv = np.asarray(singular_values, dtype=float)
            if sv.shape[0] != self.true_rank:
                raise ValueError("singular_values must have length true_rank.")
            self.singular_values = sv

        self._generate_data()

        if self.missing_frac > 0.0:
            self.mask = self.rng.random((self.n, self.p)) >= self.missing_frac
        else:
            self.mask = None


    @classmethod
    def from_matrix(cls, Y, R, lambda_reg, mask=None, Y_true=None, seed=None):
        """
        Build a problem from a given data matrix instead of synthetic data.

        Parameters
        ----------
        Y : np.ndarray, shape (n, p)
            Data matrix. Entries outside the mask are ignored by the objective.
        R : int
            Latent dimension.
        lambda_reg : float or array-like of length R
            Prior precision (see the class docstring).
        mask : np.ndarray of bool, shape (n, p), optional
            True where Y is observed. None means complete data.
        Y_true : np.ndarray, optional
            Clean signal, if known (used only for diagnostics).
        seed : int, optional
            Seed for the generator used by initializations.
        """
        problem = cls.__new__(cls)
        problem.rng = np.random.default_rng(seed)
        problem.seed = seed
        Y = np.asarray(Y, dtype=float)
        problem.n, problem.p = Y.shape
        problem.R = int(R)
        problem._set_lambda(lambda_reg)
        problem.noise_level = float("nan")
        problem.true_rank = None
        problem.singular_values = None
        problem.U_true = None
        problem.V_true = None
        problem.Y_true = None if Y_true is None else np.asarray(Y_true, dtype=float)

        if mask is None:
            problem.mask = None
            problem.missing_frac = 0.0
            problem.Y = Y.copy()
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != Y.shape:
                raise ValueError("mask must have the same shape as Y.")
            problem.mask = mask
            problem.missing_frac = 1.0 - mask.mean()
            # Missing entries are stored as 0 so that they never leak into
            # computations that forget to apply the mask.
            problem.Y = np.where(mask, Y, 0.0)
        return problem


    def _set_lambda(self, lambda_reg):
        lam = np.asarray(lambda_reg, dtype=float)
        if lam.ndim == 0:
            self.lambda_reg = float(lam)
            self.lam = np.full(self.R, float(lam))
        else:
            if lam.shape != (self.R,):
                raise ValueError("lambda_reg must be a scalar or have length R.")
            self.lambda_reg = lam.copy()
            self.lam = lam.copy()
        if np.any(self.lam < 0):
            raise ValueError("lambda_reg must be nonnegative.")


    def _generate_data(self):
        """
        Sample U_true, V_true with orthonormal columns and construct

            Y_true = U_true diag(singular_values) V_true^T,
            Y = Y_true + noise_level * N(0,1)^{n x p}.
        """
        # U_true: (n x true_rank), orthonormal columns
        U_rand = self.rng.standard_normal((self.n, self.true_rank))
        self.U_true, _ = np.linalg.qr(U_rand)

        # V_true: (p x true_rank), orthonormal columns
        V_rand = self.rng.standard_normal((self.p, self.true_rank))
        self.V_true, _ = np.linalg.qr(V_rand)

        S = np.diag(self.singular_values)
        self.Y_true = self.U_true @ S @ self.V_true.T

        if self.noise_level > 0.0:
            noise = self.noise_level * self.rng.standard_normal((self.n, self.p))
            self.Y = self.Y_true + noise
        else:
            self.Y = self.Y_true.copy()


    @property
    def complete(self):
        """True when every entry of Y is observed."""
        return self.mask is None


    @property
    def weights(self):
        """Observation mask as floats (all ones for complete data)."""
        if self.mask is None:
            return np.ones((self.n, self.p))
        return self.mask.astype(float)


    def sample_initial_V(self, scale=0.01):
        """
        Sample a random starting point V0 with small Gaussian entries.
        """
        V0 = scale * self.rng.standard_normal((self.p, self.R))
        return V0


    def sample_initial_UV(self, scale=0.01):
        """
        Sample a random starting point (U0, V0) with small Gaussian entries
        (for solvers that update U and V jointly).
        """
        U0 = scale * self.rng.standard_normal((self.n, self.R))
        V0 = scale * self.rng.standard_normal((self.p, self.R))
        return U0, V0


    def loss(self, M, U, V, mask=None):
        """
        Compute the regularized objective for a given matrix M:

            0.5 * ||mask * (M - U V^T)||_F^2
            + 0.5 * (tr(U Lambda U^T) + tr(V Lambda V^T))

        mask=None means every entry counts (use `objective` for the loss that
        the solvers minimize, which applies the problem's own mask).
        """
        Y_hat = U @ V.T
        residual = M - Y_hat
        if mask is not None:
            residual = np.where(mask, residual, 0.0)

        data_term = 0.5 * np.linalg.norm(residual, ord="fro")**2
        reg_term = 0.5 * (
            np.sum(self.lam * np.sum(U**2, axis=0))
            + np.sum(self.lam * np.sum(V**2, axis=0))
        )

        return data_term + reg_term


    def objective(self, U, V):
        """
        The MAP objective on the observed entries of Y.
        """
        return self.loss(self.Y, U, V, mask=self.mask)


    def relative_error(self, Z, reference="Y_true"):
        """
        Relative Frobenius error ||ref - Z||_F / ||ref||_F with ref = Y_true
        (default) or Y.
        """
        ref = self.Y_true if reference == "Y_true" else self.Y
        if ref is None:
            return float("nan")
        num = np.linalg.norm(ref - Z, ord="fro")
        den = np.linalg.norm(ref, ord="fro")
        return num / max(den, 1e-12)


    def relative_signal_error(self, U, V):
        """
        Relative Frobenius error with respect to Y_true:
            ||Y_true - U V^T||_F / ||Y_true||_F
        """
        return self.relative_error(U @ V.T, reference="Y_true")


    def grad(self, U, V):
        """
        Gradient of the regularized objective with respect to U and V.
        """
        residual = U @ V.T - self.Y
        if self.mask is not None:
            residual = np.where(self.mask, residual, 0.0)

        grad_U = residual @ V + U * self.lam
        grad_V = residual.T @ U + V * self.lam

        return grad_U, grad_V

    def grad_norm(self, U, V):
        """
        Frobenius norm of the full gradient (U and V parts together).
        """
        grad_U, grad_V = self.grad(U, V)
        norm_U = np.linalg.norm(grad_U, ord="fro")
        norm_V = np.linalg.norm(grad_V, ord="fro")
        return np.sqrt(norm_U**2 + norm_V**2)


    def hvp(self, U, V, dU, dV):
        """
        Hessian-vector product of the objective at (U, V) in direction (dU, dV).

        With E = mask * (U V^T - Y) and dE = mask * (dU V^T + U dV^T):

            H_U[dU, dV] = dE V + E dV + dU Lambda
            H_V[dU, dV] = dE^T U + E^T dU + dV Lambda
        """
        E = U @ V.T - self.Y
        dE = dU @ V.T + U @ dV.T
        if self.mask is not None:
            E = np.where(self.mask, E, 0.0)
            dE = np.where(self.mask, dE, 0.0)
        H_U = dE @ V + E @ dV + dU * self.lam
        H_V = dE.T @ U + E.T @ dU + dV * self.lam
        return H_U, H_V


    def hessian(self, U, V):
        """
        Dense Hessian of the objective with respect to vec(U), vec(V)
        (row-major), built column by column from `hvp`. Only for small problems.
        """
        nU = self.n * self.R
        dim = nU + self.p * self.R
        H = np.zeros((dim, dim))
        for k in range(dim):
            e = np.zeros(dim)
            e[k] = 1.0
            dU = e[:nU].reshape(self.n, self.R)
            dV = e[nU:].reshape(self.p, self.R)
            H_U, H_V = self.hvp(U, V, dU, dV)
            H[:, k] = np.concatenate([H_U.ravel(), H_V.ravel()])
        return 0.5 * (H + H.T)


    def get_svd(self, matrix="Y"):
        """
        Return (U_svd, s, Vt_svd) for the chosen matrix.

        Parameters
        ----------
        matrix : {"Y", "Y_true"}
            - "Y":      use the noisy observed matrix.
            - "Y_true": use the clean signal matrix.

        Returns
        -------
        U_svd : np.ndarray, shape (n, r)
        s : np.ndarray, shape (r,)
        Vt_svd : np.ndarray, shape (r, p)
            Such that M ≈ U_svd @ np.diag(s) @ Vt_svd.
        """
        if matrix == "Y":
            M = self.Y
            attr = "_svd_Y"
        elif matrix == "Y_true":
            M = self.Y_true
            attr = "_svd_Y_true"
        else:
            raise ValueError('matrix must be either "Y" or "Y_true".')

        cached = getattr(self, attr, None)
        if cached is None:
            U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
            setattr(self, attr, (U_svd, s, Vt_svd))
        else:
            U_svd, s, Vt_svd = cached

        return U_svd, s, Vt_svd


    def init_V_from_svd_directions(
        self,
        index_set,
        matrix="Y",
        scaling="random",
        scale_range=(0.1, 10.0),
    ):
        """
        Build an initialization V0 in a chosen singular subspace of the given matrix.

        V0 has the form:
            V0 = V_subset @ diag(scales),

        where V_subset contains the right singular vectors corresponding to
        the indices in index_set.

        Parameters
        ----------
        index_set : array-like of ints
            Indices of singular directions (0-based) to use. Must have length R.
        matrix : {"Y", "Y_true"}, optional
            Which matrix to take the SVD of (typically "Y").
        scaling : {"random", "soft_threshold"}, optional
            - "random":
                scales[j] ~ Uniform(scale_range[0], scale_range[1]).
            - "soft_threshold":
                scales[j] = sqrt(max(s[i_j] - lambda_j, 0)),
                where s[i_j] is the singular value at index i_j and lambda_j
                the prior precision of column slot j.
        scale_range : tuple of (float, float), optional
            Range for random scaling when scaling == "random".

        Returns
        -------
        V0 : np.ndarray, shape (p, R)
            Initialization for V with columns in the span of the selected
            singular right vectors.
        """
        U_svd, s, Vt_svd = self.get_svd(matrix=matrix)

        index_set = np.asarray(index_set, dtype=int)
        if index_set.shape[0] != self.R:
            raise ValueError("index_set must have length equal to R.")
        if index_set.min() < 0 or index_set.max() >= s.shape[0]:
            raise ValueError("index_set contains invalid singular value indices.")

        V_subset = Vt_svd[index_set, :].T  # shape (p, R)
        s_subset = s[index_set]

        if scaling == "random":
            low, high = scale_range
            scales = self.rng.uniform(low, high, size=self.R)
        elif scaling == "soft_threshold":
            scales = np.sqrt(np.maximum(s_subset - self.lam, 0.0))
        else:
            raise ValueError('scaling must be "random" or "soft_threshold".')

        V0 = V_subset @ np.diag(scales)
        return V0


    def stationary_point(self, index_set, matrix="Y"):
        """
        The stationary point (U, V) supported on the singular directions in
        index_set, with soft-thresholded scales: column j is
        f_{i_j} sqrt((s_{i_j} - lambda_j)_+) for U and g_{i_j} (same scale) for V.
        Every stationary point of the complete-data objective has this form
        up to an orthogonal rotation (part 1 of the notes).
        """
        U_svd, s, Vt_svd = self.get_svd(matrix=matrix)
        index_set = np.asarray(index_set, dtype=int)
        if index_set.shape[0] != self.R:
            raise ValueError("index_set must have length equal to R.")
        gamma = np.sqrt(np.maximum(s[index_set] - self.lam, 0.0))
        U = U_svd[:, index_set] * gamma
        V = Vt_svd[index_set, :].T * gamma
        return U, V


    def init_V_orthogonal_to_global_minimizer(
        self,
        matrix="Y",
        init_scale=0.1,
        epsilon=0.0,
        active_tol=1e-12,
    ):
        """
        Build an initialization V0 whose columns are (approximately) orthogonal
        to the right-singular subspace used by the global minimizer.

        Parameters
        ----------
        matrix : {"Y", "Y_true"}
            Which matrix defines the global minimizer subspace.
        init_scale : float
            Overall scale of the initialization (like ALSSolver(init_scale=...)).
        epsilon : float
            Optional leakage into the minimizer subspace:
                V0 = V_perp + epsilon * V_star
            Use epsilon=0.0 for exact orthogonal start.
        active_tol : float
            Columns of V_star with norm <= active_tol are treated as inactive
            (e.g., when soft-thresholding zeroes them out).

        Returns
        -------
        V0 : np.ndarray, shape (p, R)
        """
        # 1) Get a global minimizer (U_star, V_star) for the chosen matrix
        if matrix == "Y":
            _, V_star, _ = self.global_minimizer_Y()
        elif matrix == "Y_true":
            _, V_star, _ = self.global_minimizer_Ytrue()
        else:
            raise ValueError('matrix must be either "Y" or "Y_true".')

        # 2) Identify active columns (avoid using zero-padded directions)
        col_norms = np.linalg.norm(V_star, axis=0)
        active = col_norms > active_tol

        # 3) Sample a random V and project it to the orthogonal complement
        V_rand = self.rng.standard_normal((self.p, self.R))

        if np.any(active):
            # Orthonormal basis for span(V_star_active)
            Q, _ = np.linalg.qr(V_star[:, active])  # shape (p, k)
            V_perp = V_rand - Q @ (Q.T @ V_rand)
        else:
            # If minimizer uses no active directions, "orthogonal complement" is all of R^p
            V_perp = V_rand

        # 4) Normalize columns (avoid tiny norms after projection)
        norms = np.linalg.norm(V_perp, axis=0)
        norms = np.maximum(norms, 1e-12)
        V_perp = V_perp / norms

        # 5) Scale to match the usual init magnitude
        V0 = init_scale * V_perp

        # 6) Optional leakage into the minimizer subspace
        if epsilon != 0.0:
            V0 = V0 + epsilon * V_star  # V_star is already shape (p, R), padded if needed

        return V0


    def _global_minimizer_matrix(self, M):
        """
        Global minimizer of the complete-data objective
            0.5 * ||M - U V^T||_F^2 + 0.5 * (tr(U Lambda U^T) + tr(V Lambda V^T))
        for a given matrix M, via soft-thresholded SVD.

        With a diagonal Lambda the top-R singular values are paired with the
        precisions in ascending order (largest sigma with smallest lambda),
        which is the optimal pairing by the rearrangement inequality.
        """
        k = min(self.R, min(M.shape))
        U_svd, s, Vt_svd = top_svd(M, k)

        U_R = U_svd[:, :k]
        V_R = Vt_svd[:k, :].T
        s_R = s[:k]

        # slot order: the slot with the smallest lambda gets the largest sigma
        slots = np.argsort(self.lam, kind="stable")[:k]
        gamma = np.sqrt(np.maximum(s_R - self.lam[slots], 0.0))

        U_star = np.zeros((M.shape[0], self.R))
        V_star = np.zeros((M.shape[1], self.R))
        U_star[:, slots] = U_R * gamma[np.newaxis, :]
        V_star[:, slots] = V_R * gamma[np.newaxis, :]

        global_loss = self.loss(M, U_star, V_star)
        return U_star, V_star, global_loss

    def global_minimizer_Y(self):
        """
        Global minimizer of the regularized objective using the matrix Y.
        Only defined for complete data (no closed form exists with a mask).
        """
        if self.mask is not None:
            raise ValueError("No closed-form minimizer with missing entries.")
        return self._global_minimizer_matrix(self.Y)

    def global_minimizer_Ytrue(self):
        """
        Global minimizer of the regularized objective using the true matrix Y_true.
        """
        if self.Y_true is None:
            raise ValueError("Y_true is unknown for this problem.")
        return self._global_minimizer_matrix(self.Y_true)
