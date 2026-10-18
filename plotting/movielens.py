"""MovieLens-100K: test RMSE against rank for every solver (mean and spread over 5 seeds)."""

import matplotlib.pyplot as plt
import numpy as np

from common import INK, SOLVERS, read, save_both, solver_color


def draw(mode):
    ink = INK[mode]
    table = read("movielens_rmse.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    stats = table.groupby(["solver", "rank"]).test_rmse.agg(["mean", "std"]).reset_index()
    for name in SOLVERS:
        d = stats[stats.solver == name]
        if d.empty or name == "closed form":
            continue
        color = solver_color(name, mode)
        if name == "SVT":
            ranks = table[table.solver == "SVT"]["rank"]
            ax.errorbar(ranks.mean(), d["mean"].iloc[0], xerr=ranks.std(), yerr=d["std"].iloc[0],
                        fmt="o", color=color, ms=6, mec=ink["surface"], mew=1.5,
                        elinewidth=1.0, capsize=0, label="SVT (rank set by early stopping)")
            continue
        ax.errorbar(d["rank"], d["mean"], yerr=d["std"], color=color, lw=1.6, marker="o",
                    ms=5.5, mec=ink["surface"], mew=1.5, elinewidth=1.0, capsize=0, label=name)
    # the two weak baselines sit far above the rest; they are named, not drawn
    base = table[table.solver == "training mean"].test_rmse.mean()
    cf = stats[stats.solver == "closed form"]["mean"]
    ax.text(0.02, 0.97, f"off the scale: closed form on the zero-filled matrix "
            f"{cf.min():.3f} to {cf.max():.3f},\npredicting the training mean {base:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=ink["secondary"])
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20])
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlim(1.7, 24)
    lo = stats[~stats.solver.isin(["closed form", "training mean"])]
    ax.set_ylim((lo["mean"] - lo["std"]).min() - 0.01, (lo["mean"] + lo["std"]).max() + 0.025)
    ax.set_xlabel("rank R")
    ax.set_ylabel("test RMSE (stars)")
    ax.set_title("MovieLens-100K, 80/20 splits, mean and sd over 5 seeds")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.6)
    return fig


if __name__ == "__main__":
    save_both(draw, "movielens_rmse")
