"""Hero figure: (a) inpainting, (b) convergence of the seven solvers, (c) saddle escape."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

from common import INK, RESULTS, SOLVERS, read, save_both, solver_color


def draw(mode):
    ink = INK[mode]
    fig = plt.figure(figsize=(8, 3.7))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1.2, 1.1])

    # (a) inpainting: masked input above, soft-impute output below
    for row, (fname, caption) in enumerate([
            ("inpainting_masked.png", "half of the pixels hidden"),
            ("inpainting_reconstructed.png", "completed by soft-impute")]):
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(Image.open(RESULTS / fname))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(caption, fontsize=8.5, color=ink["secondary"], labelpad=3)
        if row == 0:
            ax.set_title("(a) Image inpainting")

    # (b) convergence curves
    ax = fig.add_subplot(gs[:, 1])
    curves = read("convergence_curves.csv")
    for name in SOLVERS:
        d = curves[curves.solver == name]
        color = solver_color(name, mode)
        if name == "closed form":
            ax.axhline(d.relative_error.iloc[0], color=color, lw=1.4, label=name)
        else:
            ax.plot(d.iteration, d.relative_error, color=color, lw=1.6, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, curves.iteration.max())
    ax.set_ylim(0.03, 1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative error against the true matrix")
    ax.set_title("(b) Seven solvers, 50% missing")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, handlelength=1.2,
              columnspacing=0.9, labelspacing=0.3, fontsize=7.5)

    # (c) saddle escape
    ax = fig.add_subplot(gs[:, 2])
    traj = read("saddle_escape_trajectories.csv")
    runs = read("saddle_escape_runs.csv")
    color = solver_color("ALS", mode)
    for _, d in traj.groupby("run"):
        ax.plot(d.step.clip(lower=1), d.loss, color=color, lw=0.6, alpha=0.18)
    bad, best = runs.loss_bad_pairing.iloc[0], runs.loss_optimum.iloc[0]
    for y, label in [(bad, f"bad pairing {bad:.2f}"), (best, f"optimum {best:.2f}")]:
        ax.axhline(y, color=ink["secondary"], lw=0.8)
        ax.text(1.3, y + 0.06, label, fontsize=7.5, color=ink["secondary"], va="bottom")
    ax.axvline(500, color=ink["muted"], lw=0.8)
    escaped = int(runs.escaped_by_500.sum())
    ax.text(650, bad - 0.2, f"step 500:\n{escaped} of {len(runs)}\nhave left\nthe saddle",
            fontsize=7.5, color=ink["secondary"], va="top")
    ax.set_xscale("log")
    ax.set_xlim(1, traj.step.max())
    ax.set_ylim(best - 0.25, bad + 0.45)
    ax.set_xlabel("projected gradient step")
    ax.set_ylabel("profiled loss")
    ax.set_title(f"(c) {len(runs)} runs from a saddle")
    return fig


if __name__ == "__main__":
    save_both(draw, "hero")
