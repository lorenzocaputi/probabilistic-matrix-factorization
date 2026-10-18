"""
Subspace dynamics with complete and incomplete data (part 4 of the notes).

Problem: 100 x 80, singular values 10, 9, ..., 1, noise sd 0.1, R = 3,
lambda = 2. Four starting points, as in the notes:
    A  the stationary point on the top three singular directions (global minimum
       for complete data),
    B  the top three directions with random scales,
    C  the stationary point on directions 4, 5, 6 (a saddle),
    D  directions 4, 5, 6 with random scales.
From each, ALS, soft-impute and gradient descent run for a fixed number of
iterations (no early stopping) on the complete matrix and with a fraction of
the entries hidden at random (5 masks per fraction).

For every run we record the final masked objective, its gap to the best value
found on that problem, the distance between the final right factor and the
best solution's right factor (sine of the largest principal angle), and the
first iteration at which that distance fell below 0.01.

Writes results/missing_data_runs.csv and results/missing_data_trajectories.csv
(per-iteration distances for ALS from C and D, first mask).
"""

from _common import parse_args, run_parallel, write_csv

import numpy as np
import pandas as pd

from pmf import LowRankMF
from pmf.linalg import subspace_distance
from pmf.solvers import ALSSolver, GradientDescentSolver, SoftImputeSolver

TOP, ALT = [0, 1, 2], [3, 4, 5]


def make_problem(missing, seed):
    return LowRankMF(n=100, p=80, true_rank=10, R=3, lambda_reg=2.0,
                     singular_values=np.arange(10, 0, -1), noise_level=0.1,
                     seed=seed, missing_frac=missing)


def starting_points(problem):
    rng = np.random.default_rng(1000 + (problem.seed or 0))
    F, s, Gt = problem.get_svd()
    points = {}
    for label, idx in (("A", TOP), ("C", ALT)):
        points[label] = problem.stationary_point(idx)
    for label, idx in (("B", TOP), ("D", ALT)):
        a, b = rng.uniform(0.1, 3.0, size=(2, len(idx)))
        points[label] = (F[:, idx] * a, Gt[idx].T * b)
    return points


def make_solver(name, problem, U0, V0):
    if name == "ALS":
        return ALSSolver(problem, V0=V0)
    if name == "soft-impute":
        return SoftImputeSolver(problem, U0=U0, V0=V0)
    if name == "gradient descent":
        return GradientDescentSolver(problem, U0=U0, V0=V0)
    raise ValueError(name)


def run_problem(job):
    missing, seed, max_iter, keep_trajectories = job
    problem = make_problem(missing, seed)
    points = starting_points(problem)
    runs, trajectories = [], []
    for name in ["ALS", "soft-impute", "gradient descent"]:
        for start, (U0, V0) in points.items():
            solver = make_solver(name, problem, U0, V0)
            Vs = [V0.copy()]
            solver.fit(max_iter=max_iter, tol=None,
                       callback=lambda s, t: Vs.append(s.V.copy()))
            runs.append(dict(solver=name, start=start, final_V=Vs[-1], Vs=Vs,
                             loss=problem.objective(solver.U, solver.V),
                             iterations=len(Vs) - 1))

    best = min(runs, key=lambda r: r["loss"])
    rows = []
    for r in runs:
        dists = np.array([subspace_distance(V, best["final_V"]) for V in r["Vs"]])
        hit = np.where(dists < 0.01)[0]
        rows.append({
            "missing_frac": missing, "seed": seed, "solver": r["solver"], "start": r["start"],
            "iterations": r["iterations"], "final_loss": r["loss"],
            "gap_to_best": r["loss"] - best["loss"],
            "relative_gap": (r["loss"] - best["loss"]) / abs(best["loss"]),
            "distance_start": dists[0], "distance_final": dists[-1],
            "iterations_to_best_subspace": int(hit[0]) if len(hit) else -1,
        })
        if keep_trajectories and r["solver"] == "ALS" and r["start"] in ("C", "D"):
            for t, dist in enumerate(dists):
                trajectories.append({"missing_frac": missing, "start": r["start"],
                                     "iteration": t, "distance": dist})
    return rows, trajectories


def main():
    args = parse_args(__doc__.splitlines()[1])
    fractions = [0.0, 0.2] if args.quick else [0.0, 0.01, 0.05, 0.1, 0.2, 0.4]
    seeds = [0] if args.quick else [0, 1, 2, 3, 4]
    max_iter = 200 if args.quick else 2000
    jobs = [(m, s, max_iter, s == 0) for m in fractions for s in seeds]
    results = run_parallel(run_problem, jobs, args.workers)

    runs = pd.DataFrame([row for rows, _ in results for row in rows])
    traj = pd.DataFrame([row for _, tr in results for row in tr])
    write_csv(runs, "missing_data_runs.csv", args.quick)
    write_csv(traj, "missing_data_trajectories.csv", args.quick)

    summary = (runs.assign(at_best=runs.relative_gap < 1e-9)
               .groupby(["missing_frac", "solver", "start"]).at_best.mean().unstack())
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
