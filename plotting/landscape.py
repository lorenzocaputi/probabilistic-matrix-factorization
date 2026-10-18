"""Stationary points of the MAP loss: loss against Morse index, and slices around a saddle."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import BLUES, INK, PALETTE, read, save_both


def draw(mode):
    ink = INK[mode]
    points = read("stationary_points.csv")
    slices = read("landscape_slices.csv")
    grid = read("landscape_grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9))

    # (a) loss of every stationary point against its number of negative eigenvalues
    ax = axes[0]
    for k, (cols, d) in enumerate(points.groupby("k")):
        ax.plot(d.n_negative, d.loss, "o", ms=5, mec=ink["surface"], mew=1.0,
                color=PALETTE[mode][k % 7], label=f"k = {cols}")
    ax.set_xlabel("negative Hessian eigenvalues")
    ax.set_ylabel("MAP loss")
    ax.set_title(f"(a) All {len(points)} stationary points")
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.2, labelspacing=0.25,
              title="nonzero columns", title_fontsize=7.5)

    # (b) 1-D slices through the saddle
    ax = axes[1]
    d = slices[slices.point == "saddle"]
    for k, (label, s) in enumerate(d.groupby("direction", sort=False)):
        ax.plot(s.t, s.loss_change, color=PALETTE[mode][k], label=label)
    ax.axhline(0, color=ink["axis"], lw=0.8)
    ax.set_ylim(-2.4, 3.0)
    ax.set_xlabel("step along unit direction")
    ax.set_ylabel("loss change")
    ax.set_title("(b) Slices through a saddle")
    ax.legend(loc="lower center", fontsize=7.5, labelspacing=0.25)

    # (c) 2-D slice: plane of the most negative and most positive eigenvectors
    ax = axes[2]
    g = grid[grid.point == "saddle"]
    a = np.sort(g.a.unique())
    b = np.sort(g.b.unique())
    Z = g.pivot(index="b", columns="a", values="loss_change").loc[b, a].values
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#256abf", "#86b6ef", "#f0efec" if mode == "light" else "#383835", "#ec835a", "#d03b3b"])
    # each side of zero gets its own half of the colour map
    levels = np.concatenate([np.linspace(Z.min(), 0, 8)[:-1], np.linspace(0, Z.max(), 12)])
    cs = ax.contourf(a, b, Z, levels=levels, cmap=cmap, norm=TwoSlopeNorm(0, Z.min(), Z.max()))
    ax.contour(a, b, Z, levels=[0], colors=[ink["primary"]], linewidths=0.8)
    ax.plot(0, 0, "o", color=ink["primary"], ms=4)
    ax.set_xlabel("negative-curvature direction")
    ax.set_ylabel("positive-curvature direction")
    ax.set_title("(c) Loss change near the saddle")
    ax.grid(False)
    cb = fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_ticks([t for t in (-0.4, -0.2, 0.0, 1.0, 2.0, 3.0) if Z.min() <= t <= Z.max()])
    cb.ax.yaxis.set_major_formatter(lambda x, _: f"{x:g}".replace("-", "\u2212"))
    return fig


if __name__ == "__main__":
    save_both(draw, "landscape")
