"""
MovieLens-100K: test RMSE against rank for every solver.

Data: 100,000 ratings (1 to 5) of 1,682 movies by 943 users, downloaded from
files.grouplens.org into data/ (never committed) and checked against its MD5.

Protocol, for each of 5 seeds:
  - a random 80/20 train/test split of the ratings; 10% of the training
    ratings are held out for validation and the models are fitted on the rest;
  - ratings are centred by the training mean, predictions are clipped to [1, 5];
  - ranks R in {2, 5, 10, 20};
  - ALS: lambda chosen on the validation set from a grid. Gradient descent and
    soft-impute minimize the same masked MAP objective, so they reuse ALS's lambda;
  - closed form: the soft-thresholded SVD of the zero-filled matrix, with
    lambda and the rescaling chosen on the validation set;
  - VB learns its noise and prior variances; Gibbs uses alpha = 2, burn-in 50
    and 250 kept samples, as in Salakhutdinov and Mnih (2008);
  - SVT has no rank parameter: tau = 5 sqrt(n p) as in Cai, Candes and Shen,
    step delta = 1.9 (the paper's 1.2 / p diverges on these non-uniformly
    sampled data; the convergence proof needs delta < 2), stopped at the
    iteration with the best validation RMSE. Its rank is whatever it reached there.

Writes results/movielens_rmse.csv (one row per seed, solver and rank) and
results/movielens_tuning.csv (validation RMSE over the lambda grids).
"""

from _common import DATA, parse_args, run_parallel, write_csv

import hashlib
import io
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from pmf import LowRankMF
from pmf.solvers import (ALSSolver, ClosedFormSolver, GibbsSolver, GradientDescentSolver,
                         SoftImputeSolver, SVTSolver, VBSolver)

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MD5 = "0e33842e24a9c977be4e0107933c0723"
ZIP = DATA / "ml-100k.zip"

RANKS = [2, 5, 10, 20]
ALS_LAMBDAS = [1.0, 2.0, 4.0, 7.0, 10.0, 14.0, 20.0, 28.0]
CLOSED_FORM_GRID = [(False, 0.125), (False, 0.5), (False, 2.0), (False, 8.0),
                    (True, 200.0), (True, 800.0), (True, 3200.0)]


def download():
    """Download MovieLens-100K once and verify its MD5."""
    if not ZIP.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, ZIP)
    digest = hashlib.md5(ZIP.read_bytes()).hexdigest()
    if digest != MD5:
        raise RuntimeError(f"{ZIP} has MD5 {digest}, expected {MD5}; delete it and rerun.")


def load_ratings():
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read("ml-100k/u.data")
    return pd.read_csv(io.BytesIO(raw), sep="\t", names=["user", "item", "rating", "timestamp"])


class Split:
    """One seeded train / validation / test split, as dense centred matrices."""

    def __init__(self, ratings, seed):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ratings))
        n_test = len(ratings) // 5
        test = ratings.iloc[perm[:n_test]]
        train = ratings.iloc[perm[n_test:]]
        n_val = len(train) // 10
        self.val, self.fit_part, self.test = train.iloc[:n_val], train.iloc[n_val:], test
        self.n, self.p = ratings.user.max(), ratings.item.max()
        self.mean = self.fit_part.rating.mean()
        self.Y = np.zeros((self.n, self.p))
        self.mask = np.zeros((self.n, self.p), dtype=bool)
        rows, cols = self.fit_part.user.values - 1, self.fit_part.item.values - 1
        self.Y[rows, cols] = self.fit_part.rating.values - self.mean
        self.mask[rows, cols] = True

    def problem(self, R, lam, seed):
        return LowRankMF.from_matrix(self.Y, R=R, lambda_reg=lam, mask=self.mask, seed=seed)

    def rmse(self, Z, part):
        df = self.val if part == "val" else self.test
        pred = np.clip(Z[df.user.values - 1, df.item.values - 1] + self.mean, 1.0, 5.0)
        return float(np.sqrt(np.mean((pred - df.rating.values) ** 2)))


def fit_and_score(split, solver, fit_kwargs):
    t0 = time.perf_counter()
    solver.fit(**fit_kwargs)
    Z = solver.estimate()
    return {"val_rmse": split.rmse(Z, "val"), "test_rmse": split.rmse(Z, "test"),
            "iterations": solver.n_iter, "time_s": time.perf_counter() - t0}


def run_rank(job):
    seed, R, quick = job
    split = Split(load_ratings(), seed)
    it = (lambda k: max(5, k // 10)) if quick else (lambda k: k)
    rows, tuning = [], []

    def add(name, lam, result, objective=np.nan):
        rows.append({"seed": seed, "solver": name, "rank": R, "lambda": lam,
                     "objective": objective, **result})

    # closed form: lambda and rescaling on validation
    best = None
    for rescale, lam in CLOSED_FORM_GRID:
        solver = ClosedFormSolver(split.problem(R, lam, seed), rescale=rescale)
        res = fit_and_score(split, solver, {})
        tuning.append({"seed": seed, "rank": R, "solver": "closed form", "lambda": lam,
                       "rescale": rescale, "val_rmse": res["val_rmse"]})
        if best is None or res["val_rmse"] < best[1]["val_rmse"]:
            best = (lam, res)
    add("closed form", *best)

    # ALS: lambda on validation, reused by gradient descent and soft-impute
    best = None
    for lam in (ALS_LAMBDAS[1::2] if quick else ALS_LAMBDAS):
        problem = split.problem(R, lam, seed)
        solver = ALSSolver(problem, init_scale=0.1)
        res = fit_and_score(split, solver, {"max_iter": it(100), "tol": 1e-6})
        tuning.append({"seed": seed, "rank": R, "solver": "ALS", "lambda": lam,
                       "rescale": np.nan, "val_rmse": res["val_rmse"]})
        if best is None or res["val_rmse"] < best[1]["val_rmse"]:
            best = (lam, res, problem.objective(solver.U, solver.V))
    lam = best[0]
    add("ALS", *best)

    problem = split.problem(R, lam, seed)
    solver = GradientDescentSolver(problem, init_scale=0.1)
    res = fit_and_score(split, solver, {"max_iter": it(500), "tol": 1e-8})
    add("gradient descent", lam, res, problem.objective(solver.U, solver.V))

    problem = split.problem(R, lam, seed)
    solver = SoftImputeSolver(problem)
    res = fit_and_score(split, solver, {"max_iter": it(200), "tol": 1e-5})
    add("soft-impute", lam, res, problem.objective(solver.U, solver.V))

    solver = VBSolver(split.problem(R, lam, seed), init_scale=0.1)
    add("VB", np.nan, fit_and_score(split, solver, {"max_iter": it(100), "tol": 1e-7}))

    solver = GibbsSolver(split.problem(R, lam, seed), alpha=2.0, burn_in=it(50))
    add("Gibbs", np.nan, fit_and_score(split, solver, {"max_iter": it(300)}))
    return rows, tuning


def run_svt(job):
    seed, quick = job
    split = Split(load_ratings(), seed)
    problem = split.problem(1, 1.0, seed)  # R and lambda are not used by SVT
    solver = SVTSolver(problem, delta=1.9, rank_max=60)
    trace = []
    solver.fit(max_iter=15 if quick else 150, tol=None,
               callback=lambda s, t: trace.append((split.rmse(s.estimate(), "val"),
                                                   split.rmse(s.estimate(), "test"),
                                                   len(s._s), s.history["time"][-1])))
    trace = np.array(trace)
    k = int(np.argmin(trace[:, 0]))
    return [{"seed": seed, "solver": "SVT", "rank": int(trace[k, 2]), "lambda": np.nan,
             "objective": np.nan, "val_rmse": trace[k, 0], "test_rmse": trace[k, 1],
             "iterations": k + 1, "time_s": trace[k, 3],
             "max_rank_reached": int(trace[:, 2].max())}], []


def run_job(job):
    return run_svt(job[1:]) if job[0] == "svt" else run_rank(job[1:])


def main():
    args = parse_args(__doc__.splitlines()[1],
                      download_only={"action": "store_true", "help": "fetch the data and stop"})
    download()
    if args.download_only:
        return
    seeds = [0] if args.quick else [0, 1, 2, 3, 4]
    ranks = [2, 5] if args.quick else RANKS
    jobs = [("svt", s, args.quick) for s in seeds]
    jobs += [("rank", s, R, args.quick) for R in sorted(ranks, reverse=True) for s in seeds]
    results = run_parallel(run_job, jobs, args.workers)

    rows = [r for rs, _ in results for r in rs]
    tuning = [t for _, ts in results for t in ts]
    ratings = load_ratings()
    for seed in seeds:  # baseline: predict the training mean everywhere
        split = Split(ratings, seed)
        zero = np.zeros((split.n, split.p))
        rows.append({"seed": seed, "solver": "training mean", "rank": 0,
                     "val_rmse": split.rmse(zero, "val"), "test_rmse": split.rmse(zero, "test")})
    table = pd.DataFrame(rows).sort_values(["solver", "rank", "seed"])
    write_csv(table, "movielens_rmse.csv", args.quick)
    write_csv(pd.DataFrame(tuning), "movielens_tuning.csv", args.quick)

    summary = table.groupby(["solver", "rank"]).test_rmse.agg(["mean", "std"]).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
