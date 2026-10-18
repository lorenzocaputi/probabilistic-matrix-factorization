"""Linear convergence rates: ALS against the predicted rate, and iterations against lambda and gap."""

import matplotlib.pyplot as plt
import numpy as np

from common import INK, read, save_both, solver_color

ITERATIVE = ["ALS", "gradient descent", "soft-impute", "VB"]


def draw(mode):
    ink = INK[mode]
    rates = read("convergence_rates.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9))

    # (a) measured against predicted rate, ALS, complete data
    ax = axes[0]
    d = rates[(rates.solver == "ALS") & (rates.data == "complete")].dropna(subset=["empirical_rate"])
    ax.plot([0, 1], [0, 1], color=ink["axis"], lw=0.8)
    ax.plot(d.predicted_rate, d.empirical_rate, "o", color=solver_color("ALS", mode),
            ms=5, mec=ink["surface"], mew=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted rate")
    ax.set_ylabel("measured rate")
    ax.set_title(f"(a) ALS, complete data, {len(d)} settings")

    # (b) iterations to 1e-6 against lambda, 30% missing, smallest gap
    missing = rates[rates.data != "complete"]
    gap_small = missing.gap.min()
    ax = axes[1]
    for name in ITERATIVE:
        d = missing[(missing.solver == name) & np.isclose(missing.gap, gap_small)].sort_values("lambda")
        ax.plot(d["lambda"], d["iterations_to_1e-6"], marker="o", ms=4.5, mec=ink["surface"],
                mew=1.0, color=solver_color(name, mode), label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"iterations to $10^{-6}$")
    ax.set_title(f"(b) 30% missing, gap {gap_small:.2f}")

    # (c) iterations against the gap, 30% missing, lambda = 2
    ax = axes[2]
    for name in ITERATIVE:
        d = missing[(missing.solver == name) & np.isclose(missing["lambda"], 2.0)].sort_values("gap")
        ax.plot(d.gap, d["iterations_to_1e-6"], marker="o", ms=4.5, mec=ink["surface"],
                mew=1.0, color=solver_color(name, mode), label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"gap $\sigma_R - \sigma_{R+1}$")
    ax.set_title(r"(c) 30% missing, $\lambda = 2$")
    lo = min(a.get_ylim()[0] for a in axes[1:])
    hi = max(a.get_ylim()[1] for a in axes[1:])
    for a in axes[1:]:
        a.set_ylim(lo, hi)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    return fig


if __name__ == "__main__":
    save_both(draw, "convergence_rates")
