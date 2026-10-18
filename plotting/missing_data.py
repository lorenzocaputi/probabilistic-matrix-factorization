"""Subspace dynamics of ALS from a saddle, complete against incomplete data."""

import matplotlib.pyplot as plt
import numpy as np

from common import INK, PALETTE, read, save_both


def draw(mode):
    ink = INK[mode]
    traj = read("missing_data_trajectories.csv")
    runs = read("missing_data_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))

    ax = axes[0]
    d = traj[traj.start == "C"]
    for k, (frac, s) in enumerate(d.groupby("missing_frac")):
        label = "complete" if frac == 0 else f"{frac:.0%} missing"
        s = s[s.iteration <= 400]
        ax.plot(s.iteration + 1, s.distance.clip(lower=1e-16), color=PALETTE[mode][k], label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 5)
    ax.set_xlabel("ALS iteration + 1")
    ax.set_ylabel("sine of largest angle to best subspace")
    ax.set_title("(a) ALS started at the saddle C")
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)

    ax = axes[1]
    c = runs[runs.start == "C"]
    fr = sorted(c.missing_frac.unique())
    xpos = {f: (0.004 if f == 0 else f) for f in fr}
    for k, name in enumerate(["ALS", "gradient descent", "soft-impute"]):
        d = c[c.solver == name]
        stuck = d[d.iterations_to_best_subspace < 0]
        med = d[d.iterations_to_best_subspace >= 0].groupby("missing_frac").iterations_to_best_subspace.median()
        ax.plot([xpos[f] for f in med.index], med.values.clip(min=1), marker="o", ms=5,
                mec=ink["surface"], mew=1.2, color=PALETTE[mode][k], label=name)
        for f in stuck.missing_frac.unique():
            ax.annotate("never moves:\nzero gradient", (xpos[f], 1500), fontsize=7.5,
                        color=ink["secondary"], ha="left", va="center",
                        xytext=(6, 0), textcoords="offset points")
            ax.plot(xpos[f], 1500, marker="x", ms=6, color=PALETTE[mode][k], mew=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 4000)
    ax.set_xticks([xpos[f] for f in fr])
    ax.set_xticklabels(["0" if f == 0 else f"{f:g}" for f in fr])
    ax.minorticks_off()
    ax.set_xlabel("fraction of entries missing")
    ax.set_ylabel("iterations to the best subspace")
    ax.set_title("(b) From the saddle C, median of 5 masks")
    ax.legend(loc="upper right", fontsize=7.5)
    return fig


if __name__ == "__main__":
    save_both(draw, "missing_data")
