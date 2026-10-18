"""Saddle escape: steps to reach the optimum, base case and grid over prior scale and rank."""

import matplotlib.pyplot as plt
import numpy as np

from common import INK, PALETTE, read, save_both, solver_color


def draw(mode):
    ink = INK[mode]
    runs = read("saddle_escape_runs.csv")
    grid = read("saddle_escape_grid.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))

    # (a) distribution of steps to the optimum in the base case
    ax = axes[0]
    steps = runs.steps_to_optimum[runs.steps_to_optimum >= 0]
    bins = np.geomspace(max(steps.min(), 1) * 0.9, steps.max() * 1.1, 25)
    ax.hist(steps, bins=bins, color=solver_color("ALS", mode), rwidth=0.85)
    ax.axvline(500, color=ink["muted"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("steps until the loss is within 1e-6 of the optimum")
    ax.set_ylabel("runs")
    escaped = int(runs.escaped_by_500.sum())
    ax.set_title(f"(a) Notebook case: {len(steps)} of {len(runs)} reach the optimum")
    ax.text(560, ax.get_ylim()[1] * 0.97, f"step 500: {escaped} of {len(runs)}\nhad left the saddle",
            fontsize=8, color=ink["secondary"], va="top")

    # (b) median steps to the optimum against prior scale, one line per rank
    ax = axes[1]
    g = grid[grid.prior_scale != "notebook"].copy()
    g["prior_scale"] = g.prior_scale.astype(float)
    slow = []
    for k, (R, d) in enumerate(g.groupby("R")):
        d = d.sort_values("prior_scale")
        if d.median_steps_to_optimum.isna().all():
            slow.append(R)
            continue
        ax.plot(d.prior_scale, d.median_steps_to_optimum, marker="o", ms=5,
                mec=ink["surface"], mew=1.2, color=PALETTE[mode][k], label=f"R = {R}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(g.prior_scale.unique()))
    ax.set_xticklabels([f"{x:g}" for x in sorted(g.prior_scale.unique())])
    ax.minorticks_off()
    ax.set_ylim(0, 1.2 * g.median_steps_to_optimum.max())
    ax.set_xlabel(r"prior scale $c$,  $d = c\,\cdot$ linspace(0.1, 3.2, R)")
    ax.set_ylabel("median steps to the optimum")
    ax.set_title("(b) Grid over prior scale and rank")
    ax.legend(loc="upper center", ncol=3)
    if slow:
        steps_max = int(grid.max_steps.iloc[0])
        ax.text(0.03, 0.04, f"R = {', '.join(map(str, slow))}: no run within 1e-6 of the optimum\n"
                f"after {steps_max:,} steps", transform=ax.transAxes, fontsize=8,
                color=ink["secondary"], va="bottom")
    return fig


if __name__ == "__main__":
    save_both(draw, "saddle_escape")
