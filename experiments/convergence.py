"""
Convergence of the seven solvers.

Part A (curves): one matrix-completion problem (300 x 200, rank 5,
singular values 100, 80, 60, 40, 20, noise sd 0.1, half of the entries
missing). Every solver runs 500 iterations from the same kind of start and
we record the relative error ||Z_t - Y_true||_F / ||Y_true||_F per iteration.
The MAP solvers use lambda = noise * (sqrt(n) + sqrt(p)) * sqrt(observed
fraction), the size of the observed noise's top singular value.

Part B (rates): iterations to tolerance and the empirical linear rate of
each solver towards its own limit, on a grid of lambda and of the spectral
gap sigma_R - sigma_{R+1} (100 x 80, R = 4, singular values
10, 9, 8, 7, s5, 3, 2, 1 with s5 varied), for complete data and with 30%
of the entries missing. For ALS on complete data the rate predicted by
the notes is also recorded:

    rho = max( (sigma_{R+1} / sigma_R)^2 , max_{i <= R} (1 - 2 lambda / sigma_i)^2 ).

The first term is orthogonal iteration (span(V) <- span(Y^T Y V) under ALS);
the second is the local rate |g'(gamma*)| = |1 - 2 lambda / sigma| of the
rank-1 recursion (Proposition 2c), squared because one ALS iteration is
two half-steps.

Writes results/convergence_curves.csv and results/convergence_rates.csv.
"""

from _common import parse_args, run_parallel, write_csv

import numpy as np
import pandas as pd

from pmf import LowRankMF
from pmf.solvers import (ALSSolver, ClosedFormSolver, GibbsSolver, GradientDescentSolver,
                         SoftImputeSolver, SVTSolver, VBSolver)


# ---------------------------------------------------------------- Part A

CURVE_SETUP = dict(n=300, p=200, true_rank=5, R=5, singular_values=[100, 80, 60, 40, 20],
                   noise_level=0.1, missing_frac=0.5, seed=0)


def curve_problem():
    s = CURVE_SETUP
    lam = s["noise_level"] * (np.sqrt(s["n"]) + np.sqrt(s["p"])) * np.sqrt(1 - s["missing_frac"])
    return LowRankMF(lambda_reg=lam, **s), lam


def curve_solvers(problem, noise):
    return {
        "closed form": (ClosedFormSolver(problem), {}),
        "ALS": (ALSSolver(problem, init_scale=0.1), {"tol": None}),
        "gradient descent": (GradientDescentSolver(problem, init_scale=0.1), {"tol": None}),
        "soft-impute": (SoftImputeSolver(problem), {"tol": None}),
        "SVT": (SVTSolver(problem), {"tol": None}),
        "VB": (VBSolver(problem), {"tol": None}),
        "Gibbs": (GibbsSolver(problem, alpha=1.0 / noise**2, burn_in=50), {}),
    }


def run_curve(job):
    name, max_iter = job
    problem, lam = curve_problem()
    solver, kwargs = curve_solvers(problem, CURVE_SETUP["noise_level"])[name]
    rows = []

    def record(s, t):
        rows.append({"solver": name, "iteration": t + 1, "time_s": s.history["time"][-1],
                     "relative_error": problem.relative_error(s.estimate())})

    solver.fit(max_iter=max_iter, callback=record, **kwargs)
    for row in rows:
        row["lambda"] = lam
    return rows


# ---------------------------------------------------------------- Part B

def rate_problem(lam, s5, missing):
    return LowRankMF(n=100, p=80, true_rank=8, R=4, lambda_reg=lam,
                     singular_values=[10, 9, 8, 7, s5, 3, 2, 1], noise_level=0.05,
                     seed=1, missing_frac=missing)


def make_rate_solver(name, problem):
    if name == "ALS":
        return ALSSolver(problem, init_scale=0.1)
    if name == "gradient descent":
        return GradientDescentSolver(problem, init_scale=0.1)
    if name == "soft-impute":
        return SoftImputeSolver(problem)
    if name == "SVT":
        return SVTSolver(problem)
    if name == "VB":
        # fixed hyperparameters: the variational counterpart of the same MAP model
        return VBSolver(problem, learn_noise=False, learn_prior=False)
    raise ValueError(name)


def empirical_rate(errors, lo=1e-10, hi=1e-4):
    """Geometric decay factor per iteration, fitted on the tail where lo < error < hi."""
    e = np.asarray(errors)
    idx = np.where((e > lo) & (e < hi))[0]
    if len(idx) < 5:
        return np.nan
    slope = np.polyfit(idx, np.log(e[idx]), 1)[0]
    return float(np.exp(slope))


def run_rate(job):
    name, lam, s5, missing, max_iter = job
    problem = rate_problem(lam, s5, missing)
    solver = make_rate_solver(name, problem)
    factors = []
    solver.fit(max_iter=max_iter, tol=None,
               callback=lambda s, t: factors.append(tuple(f.copy() for f in s.factors())))
    Z_limit = solver.estimate()
    norm = max(np.linalg.norm(Z_limit), 1e-12)
    errors = [np.linalg.norm(A @ B.T - Z_limit) / norm for A, B in factors[:-1]]
    below = np.where(np.array(errors) < 1e-6)[0]

    s = problem.get_svd()[1]
    R = problem.R
    gap = s[R - 1] - s[R]
    pred = np.nan
    if name == "ALS" and missing == 0.0:
        pred = max((s[R] / s[R - 1]) ** 2, np.max((1 - 2 * lam / s[:R]) ** 2))
    return {
        "solver": name, "data": "complete" if missing == 0.0 else f"{int(missing * 100)}% missing",
        "lambda": lam if name != "SVT" else np.nan, "sigma_R": s[R - 1], "sigma_R1": s[R],
        "gap": gap, "iterations_to_1e-6": int(below[0]) + 1 if len(below) else np.nan,
        "empirical_rate": empirical_rate(errors), "predicted_rate": pred,
        "final_relative_change": errors[-1] if errors else np.nan,
    }


def main():
    args = parse_args(__doc__.splitlines()[1])
    curve_iter = 30 if args.quick else 500
    names = ["closed form", "ALS", "gradient descent", "soft-impute", "SVT", "VB", "Gibbs"]
    rows = run_parallel(run_curve, [(name, curve_iter) for name in names], args.workers)
    write_csv(pd.DataFrame([r for rs in rows for r in rs]), "convergence_curves.csv", args.quick)

    lambdas = [1.0, 3.0] if args.quick else [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    s5_values = [6.0, 4.0] if args.quick else [6.5, 6.0, 5.0, 4.0, 3.5]
    max_iter = 200 if args.quick else 3000
    jobs = []
    for missing in (0.0, 0.3):
        for s5 in s5_values:
            for name in ["ALS", "gradient descent", "soft-impute", "VB"]:
                if name == "soft-impute" and missing == 0.0:
                    continue  # one step on complete data
                for lam in lambdas:
                    jobs.append((name, lam, s5, missing, max_iter))
            if missing > 0:
                jobs.append(("SVT", np.nan, s5, missing, max_iter))
    rows = run_parallel(run_rate, jobs, args.workers)
    write_csv(pd.DataFrame(rows), "convergence_rates.csv", args.quick)


if __name__ == "__main__":
    main()
