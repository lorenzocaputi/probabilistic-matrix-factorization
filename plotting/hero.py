"""Hero figure: (a) the seven solvers on a simulated matrix, (b) their test error on MovieLens."""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import INK, SOLVERS, read, save_both, solver_color


def movielens_summary():
    """
    Mean and sd over seeds of the test RMSE at the rank each solver chose on
    validation (SVT has no rank to choose: its rank is set by early stopping).
    """
    table = read("movielens_rmse.csv")
    stats = (table.groupby(["solver", "rank"])
             .agg(val=("val_rmse", "mean"), mean=("test_rmse", "mean"), sd=("test_rmse", "std"))
             .reset_index())
    best = stats.loc[stats.groupby("solver").val.idxmin()].set_index("solver")
    svt = table[table.solver == "SVT"].test_rmse
    best.loc["SVT", ["mean", "sd"]] = svt.mean(), svt.std()
    baseline = best.loc["training mean", "mean"]
    best = best.drop(index="training mean").sort_values("mean", ascending=False)
    return best, baseline


def draw(mode):
    ink = INK[mode]
    fig = plt.figure(figsize=(8, 3.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0])

    # (a) convergence curves on a simulated matrix, labelled at their right ends
    ax = fig.add_subplot(gs[0, 0])
    curves = read("convergence_curves.csv")
    for name in SOLVERS:
        d = curves[curves.solver == name]
        color = solver_color(name, mode)
        if name == "closed form":
            ax.axhline(d.relative_error.iloc[0], color=color, lw=1.4)
        else:
            ax.plot(d.iteration, d.relative_error, color=color, lw=1.6)
    final = curves.groupby("solver").relative_error.last()
    for names, text in [(["closed form"], "closed form"),
                        (["SVT"], "SVT, now fitting\nthe noise"),
                        (["ALS", "gradient descent", "soft-impute"], "ALS, gradient\ndescent, soft-impute"),
                        (["VB", "Gibbs"], "VB, Gibbs")]:
        ax.annotate(text, xy=(1.03, final[names].mean()), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=7.5, linespacing=1.1, annotation_clip=False,
                    color=solver_color(names[0], mode) if len(names) == 1 else ink["secondary"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, curves.iteration.max())
    ax.set_ylim(0.03, 1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("error against the true matrix")
    ax.set_title("(a) Simulated matrix, half of it hidden", loc="left")

    # (b) MovieLens test error per solver, sorted, with the predict-the-mean baseline
    ax = fig.add_subplot(gs[0, 1])
    best, baseline = movielens_summary()
    for y, (name, row) in enumerate(best.iterrows()):
        color = solver_color(name, mode)
        ax.errorbar(row["mean"], y, xerr=row["sd"], fmt="o", color=color, ms=6.5,
                    mec=ink["surface"], mew=1.5, elinewidth=1.2, capsize=0)
        ax.text(row["mean"] + 0.012, y, f"{row['mean']:.3f}", va="center", fontsize=8,
                color=ink["secondary"])
    ax.axvline(baseline, color=ink["muted"], lw=1.0, ls="--")
    ax.text(baseline - 0.006, len(best) - 0.55, f"predict the\naverage rating\n{baseline:.3f}",
            ha="right", va="top", fontsize=7.5, color=ink["secondary"])
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(best.index)
    for label, name in zip(ax.get_yticklabels(), best.index):
        label.set_color(solver_color(name, mode))
    ax.set_ylim(-0.6, len(best) - 0.4)
    ax.set_xlim(0.87, 1.14)
    ax.set_xticks([0.9, 0.95, 1.0, 1.05, 1.1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test error, stars (lower is better)")
    ax.set_title("(b) MovieLens, held-out ratings", loc="left")
    return fig


if __name__ == "__main__":
    save_both(draw, "hero")
