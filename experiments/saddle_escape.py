"""
Escape from a bad-pairing saddle of the diagonal-prior profiled MAP loss.

Base case (the notebook diagonal_prior_bad_pairing_saddle_experiment.ipynb):
p = 30, R = 5, eigenvalues 12, 11, 10, 8, 7, 6, 4.5, 3.5, 2.8, 2, then 1,
prior precisions d = (2.5, 0.1, 1.8, 0.6, 3.2), seed 42. 200 runs start from
Gaussian perturbations (sd 3e-3) of the bad pairing and take projected
gradient steps of size 0.01. The notebook stops at 500 steps and counts a
run as escaped when both the loss and the distance to the good pairing
dropped; here the runs continue to 20,000 steps and we also record when
each one reaches the global minimum (loss within 1e-6, relative).

Grid: the same experiment with prior precisions d = c * linspace(0.1, 3.2, R)
for prior scales c in {0.25, 0.5, 1, 2, 4} and ranks R in {2, 3, 5, 8}
(100 runs per cell, same S).

Writes results/saddle_escape_runs.csv (one row per base-case run),
results/saddle_escape_trajectories.csv (base-case losses on a log-spaced
step grid) and results/saddle_escape_grid.csv.
"""

from _common import parse_args, run_parallel, write_csv

import numpy as np
import pandas as pd

from pmf.landscape import ProjectedGradientSolver, make_S, pairings, profiled_loss, swap_curvature

P = 30
EIGVALS = np.array([12, 11, 10, 8, 7, 6, 4.5, 3.5, 2.8, 2.0] + [1.0] * (P - 10))
BASE_D = np.array([2.5, 0.1, 1.8, 0.6, 3.2])


def swap_curvatures(d, good):
    """
    Curvatures (a - b)(w_a - w_b) of all column swaps at the good or bad pairing
    of the top-R eigenvalues. At the bad pairing the most negative one sets
    the initial escape rate; at the good pairing the smallest one sets the
    final rate of convergence.
    """
    R = len(d)
    lam = EIGVALS[:R]
    w = np.sort(1.0 / (1.0 + d))
    w = w[::-1] if good else w  # good: largest eigenvalue with largest weight
    return [swap_curvature((lam[a], lam[b]), (w[a], w[b]))
            for a in range(R) for b in range(a + 1, R)]


def run_cell(job):
    scale, R, trials, steps, seed = job
    rng = np.random.RandomState(seed)
    S, G = make_S(P, EIGVALS, rng)
    d = BASE_D if scale is None else scale * np.linspace(0.1, 3.2, R)
    V_good, V_bad = pairings(G, list(range(R)), d)
    L_good = profiled_loss(S, V_good, d)
    L_bad = profiled_loss(S, V_bad, d)

    solver = ProjectedGradientSolver(S, d, V_reference=V_good)
    history = solver.fit(V_bad, trials=trials, steps=steps, rng=rng, record=[500])
    at_500 = solver.analyze_solution(step=500)
    final = solver.analyze_solution(loss_optimum=L_good)
    return dict(scale=scale, R=R, d=d, L_good=L_good, L_bad=L_bad,
                history=history if scale is None else None,  # only the base case is plotted
                escaped_500=at_500["escaped"], loss_500=at_500["loss_final"],
                steps_to_optimum=final["steps_to_optimum"], loss_final=final["loss_final"],
                curvature_bad=min(swap_curvatures(d, good=False)),
                curvature_good=min(swap_curvatures(d, good=True)))


def main():
    args = parse_args(__doc__.splitlines()[1])
    steps = 1000 if args.quick else 20000
    base_trials = 20 if args.quick else 200
    grid_trials = 10 if args.quick else 100
    scales = [0.5, 2.0] if args.quick else [0.25, 0.5, 1.0, 2.0, 4.0]
    ranks = [2, 5] if args.quick else [2, 3, 5, 8]

    jobs = [(None, 5, base_trials, steps, 42)]
    jobs += [(c, R, grid_trials, steps, 42) for R in ranks for c in scales]
    cells = run_parallel(run_cell, jobs, args.workers)

    base = cells[0]
    print(f"bad pairing loss {base['L_bad']:.2f}, optimum {base['L_good']:.2f}")
    print(f"escaped within 500 steps: {base['escaped_500'].sum()}/{base_trials}")
    reached = base["steps_to_optimum"] >= 0
    print(f"reached the optimum within {steps} steps: {reached.sum()}/{base_trials}")

    runs = pd.DataFrame({
        "run": np.arange(base_trials),
        "escaped_by_500": base["escaped_500"],
        "loss_at_500": base["loss_500"],
        "steps_to_optimum": base["steps_to_optimum"],
        "loss_final": base["loss_final"],
    })
    runs["loss_bad_pairing"] = base["L_bad"]
    runs["loss_optimum"] = base["L_good"]
    write_csv(runs, "saddle_escape_runs.csv", args.quick)

    grid_steps = np.unique(np.concatenate([[0], np.geomspace(1, steps, 120).astype(int)]))
    loss = base["history"]["loss"][:, grid_steps]
    traj = pd.DataFrame({
        "run": np.repeat(np.arange(base_trials), len(grid_steps)),
        "step": np.tile(grid_steps, base_trials),
        "loss": loss.ravel(),
    })
    write_csv(traj, "saddle_escape_trajectories.csv", args.quick)

    rows = []
    for cell in cells:
        st = cell["steps_to_optimum"]
        ok = st[st >= 0]
        rows.append({
            "prior_scale": "notebook" if cell["scale"] is None else cell["scale"],
            "R": cell["R"], "runs": len(st),
            "d": " ".join(f"{x:.3g}" for x in cell["d"]),
            "loss_bad_pairing": cell["L_bad"], "loss_optimum": cell["L_good"],
            "most_negative_curvature_at_saddle": cell["curvature_bad"],
            "smallest_curvature_at_optimum": cell["curvature_good"],
            "escaped_by_500": int(cell["escaped_500"].sum()),
            "reached_optimum": int(len(ok)),
            "median_steps_to_optimum": float(np.median(ok)) if len(ok) else np.nan,
            "p90_steps_to_optimum": float(np.percentile(ok, 90)) if len(ok) else np.nan,
            "median_final_gap": float(np.median(cell["loss_final"] - cell["L_good"])),
            "max_steps": steps,
        })
    write_csv(pd.DataFrame(rows), "saddle_escape_grid.csv", args.quick)


if __name__ == "__main__":
    main()
