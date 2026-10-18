"""
The MAP loss landscape around its stationary points.

Problem: 50 x 40, singular values 10, 9, ..., 1 (exact, no noise), R = 4,
lambda = 3.5, so seven singular values exceed lambda. By part 1 of the notes
every stationary point is, up to rotation, a choice of k <= R of those seven
directions with scales sqrt(sigma - lambda) (the other columns zero), with loss

    L(I) = 0.5 ||Y||_F^2 - 0.5 * sum_{i in I} (sigma_i - lambda)^2.

For all 99 such points we compute the loss and the inertia of the Hessian
(numbers of negative, zero and positive eigenvalues). Then 1-D and 2-D slices
of the loss around the global minimum and around the saddle that swaps
sigma_4 = 7 for sigma_5 = 6, along Hessian eigenvectors.

Writes results/stationary_points.csv, results/landscape_slices.csv and
results/landscape_grid.csv.
"""

from _common import parse_args, write_csv

from itertools import combinations

import numpy as np
import pandas as pd

from pmf import LowRankMF
from pmf.landscape import hessian_inertia

LAM = 3.5
R = 4


def make_problem(n=50, p=40):
    return LowRankMF(n=n, p=p, true_rank=10, R=R, lambda_reg=LAM,
                     singular_values=np.arange(10, 0, -1), noise_level=0.0, seed=43)


def stationary_point(problem, subset):
    """Columns on the singular directions in subset, zero columns after them."""
    F, s, Gt = problem.get_svd()
    U = np.zeros((problem.n, R))
    V = np.zeros((problem.p, R))
    for j, i in enumerate(subset):
        g = np.sqrt(max(s[i] - LAM, 0.0))
        U[:, j] = F[:, i] * g
        V[:, j] = Gt[i] * g
    return U, V


def predicted_negative_directions(subset, active):
    """
    Count of descent directions predicted by the 2 x 2 analysis of the notes:
    one for every pair (i in I, j active and not in I) with sigma_j > sigma_i,
    and one for every zero column paired with every active j not in I.
    (Indices are ordered, so sigma_j > sigma_i means j < i.)
    """
    outside = [j for j in active if j not in subset]
    swaps = sum(1 for i in subset for j in outside if j < i)
    zero_cols = R - len(subset)
    return swaps + zero_cols * len(outside)


def eigen_direction(problem, U, V, which):
    """Unit Hessian eigenvector (as a (dU, dV) pair) for the smallest or largest eigenvalue."""
    H = problem.hessian(U, V)
    vals, vecs = np.linalg.eigh(H)
    k = 0 if which == "min" else -1
    if which == "min_positive":
        k = int(np.argmax(vals > 1e-8))
    v = vecs[:, k]
    nU = problem.n * R
    return vals[k], v[:nU].reshape(problem.n, R), v[nU:].reshape(problem.p, R)


def main():
    args = parse_args(__doc__.splitlines()[1])
    problem = make_problem(*( (20, 16) if args.quick else (50, 40)))
    s = problem.get_svd()[1]
    active = [i for i in range(len(s)) if s[i] > LAM]
    print(f"active singular values: {s[active].round(3)}")

    rows = []
    half_norm = 0.5 * np.linalg.norm(problem.Y) ** 2
    for k in range(R + 1):
        for subset in combinations(active, k):
            U, V = stationary_point(problem, subset)
            neg, zero, pos, _ = hessian_inertia(problem, U, V, tol=1e-8)
            rows.append({
                "subset": " ".join(str(i + 1) for i in subset),  # 1-based singular value index
                "k": k,
                "loss": problem.objective(U, V),
                "loss_formula": half_norm - 0.5 * sum((s[i] - LAM) ** 2 for i in subset),
                "grad_norm": problem.grad_norm(U, V),
                "n_negative": neg, "n_zero": zero, "n_positive": pos,
                "predicted_negative": predicted_negative_directions(subset, active),
            })
    table = pd.DataFrame(rows).sort_values("loss").reset_index(drop=True)
    write_csv(table, "stationary_points.csv", args.quick)
    minima = table[table.n_negative == 0]
    print(f"{len(table)} stationary points, {len(minima)} with no negative curvature: "
          f"{minima.subset.tolist()}")

    # 1-D and 2-D slices around the minimum and one saddle
    points = {"minimum": (0, 1, 2, 3), "saddle": (0, 1, 2, 4)}
    t = np.linspace(-2.0, 2.0, 201)
    slices, grids = [], []
    m = 41 if args.quick else 81
    a_grid = np.linspace(-2.5, 2.5, m)   # along the smallest (negative at the saddle) eigenvector
    b_grid = np.linspace(-0.5, 0.5, m)   # along the largest eigenvector (much steeper)
    for name, subset in points.items():
        U, V = stationary_point(problem, subset)
        L0 = problem.objective(U, V)
        lo_val, lo_U, lo_V = eigen_direction(problem, U, V, "min" if name == "saddle" else "min_positive")
        hi_val, hi_U, hi_V = eigen_direction(problem, U, V, "max")
        rng = np.random.default_rng(0)
        r_U, r_V = rng.standard_normal(U.shape), rng.standard_normal(V.shape)
        scale = 1.0 / np.sqrt(np.sum(r_U**2) + np.sum(r_V**2))
        directions = {
            "smallest eigenvalue" if name == "minimum" else "negative eigenvalue": (lo_U, lo_V, lo_val),
            "largest eigenvalue": (hi_U, hi_V, hi_val),
            "random": (r_U * scale, r_V * scale, np.nan),
        }
        for label, (dU, dV, val) in directions.items():
            for ti in t:
                slices.append({"point": name, "direction": label, "eigenvalue": val, "t": ti,
                               "loss_change": problem.objective(U + ti * dU, V + ti * dV) - L0})
        for a in a_grid:
            for b in b_grid:
                grids.append({"point": name, "a": a, "b": b,
                              "loss_change": problem.objective(U + a * lo_U + b * hi_U,
                                                               V + a * lo_V + b * hi_V) - L0})
    write_csv(pd.DataFrame(slices), "landscape_slices.csv", args.quick)
    write_csv(pd.DataFrame(grids), "landscape_grid.csv", args.quick)


if __name__ == "__main__":
    main()
