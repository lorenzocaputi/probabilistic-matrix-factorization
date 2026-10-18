"""Small linear-algebra helpers shared by the problem and the solvers."""

import numpy as np
from scipy.sparse.linalg import ArpackError, svds


def top_svd(A, k):
    """
    Leading k singular triplets of A, sorted in decreasing order.

    Uses a dense SVD unless k is small compared to the matrix, in which case
    ARPACK (scipy.sparse.linalg.svds) is used with a fixed starting vector so
    that results are deterministic (falling back to the dense SVD if ARPACK fails).

    Returns
    -------
    U : (n, k), s : (k,), Vt : (k, p)
    """
    n, p = A.shape
    q = min(n, p)
    k = int(min(k, q))
    if k <= 0:
        return np.zeros((n, 0)), np.zeros(0), np.zeros((0, p))
    if q < 200 or k > q // 5:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k, :]
    v0 = np.random.default_rng(0).standard_normal(q)  # fixed start: deterministic
    try:
        U, s, Vt = svds(A, k=k, v0=v0, tol=0.0)
    except ArpackError:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k, :]
    order = np.argsort(-s)
    return U[:, order], s[order], Vt[order, :]


def soft_threshold_svd(A, lam, rank=None):
    """
    Singular value soft-thresholding S_lam(A) = F diag((sigma - lam)_+) G^T,
    optionally restricted to the leading `rank` singular values.

    Returns
    -------
    F : (n, r), s_shrunk : (r,), Gt : (r, p)
        Only the components with a positive shrunk singular value are kept.
    """
    k = min(A.shape) if rank is None else rank
    F, s, Gt = top_svd(A, k)
    s_shrunk = np.maximum(s - lam, 0.0)
    keep = s_shrunk > 0
    return F[:, keep], s_shrunk[keep], Gt[keep, :]


def balanced_factors(F, s, Gt, R):
    """
    Balanced factors U = F diag(sqrt(s)), V = G diag(sqrt(s)), padded with zero
    columns (or truncated) to exactly R columns.
    """
    r = min(len(s), R)
    root = np.sqrt(s[:r])
    U = np.zeros((F.shape[0], R))
    V = np.zeros((Gt.shape[1], R))
    U[:, :r] = F[:, :r] * root
    V[:, :r] = Gt[:r, :].T * root
    return U, V


def subspace_distance(A, B):
    """
    Distance between the column spaces of A and B: the sine of the largest
    principal angle (0 = same subspace, 1 = some direction is orthogonal).
    Zero columns are ignored.
    """
    def basis(M):
        M = M[:, np.linalg.norm(M, axis=0) > 1e-12]
        if M.shape[1] == 0:
            return M
        Q, _ = np.linalg.qr(M)
        return Q

    QA, QB = basis(A), basis(B)
    if QA.shape[1] == 0 or QB.shape[1] == 0:
        return 1.0
    if QA.shape[1] > QB.shape[1]:
        QA, QB = QB, QA
    cosines = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - np.min(cosines) ** 2)))
