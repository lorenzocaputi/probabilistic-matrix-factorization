"""Shared plotting setup: style files, solver colours, light and dark output."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# PMF_RESULTS=results/quick points the scripts at the output of --quick runs.
RESULTS = ROOT / os.environ.get("PMF_RESULTS", "results")
FIGURES = ROOT / "figures"
STYLE = FIGURES / "style.mplstyle"
STYLE_DARK = FIGURES / "style-dark.mplstyle"

# One fixed colour per solver, in the validated categorical order.
SOLVERS = ["ALS", "gradient descent", "soft-impute", "SVT", "VB", "Gibbs", "closed form"]
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
}
INK = {
    "light": dict(primary="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb"),
    "dark": dict(primary="#f0efec", secondary="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", surface="#1a1a19"),
}
# Sequential single-hue ramp (blue), light to dark.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def solver_color(name, mode):
    return PALETTE[mode][SOLVERS.index(name)]


def read(name):
    return pd.read_csv(RESULTS / name)


def save_both(draw, name):
    """
    Call draw(mode) -> figure for mode in ("light", "dark") and write
    figures/<name>.png and figures/<name>-dark.png (1600 px wide at 8 in, 200 dpi).
    """
    for mode in ("light", "dark"):
        styles = [STYLE] if mode == "light" else [STYLE, STYLE_DARK]
        with plt.style.context(styles):
            fig = draw(mode)
            suffix = "" if mode == "light" else "-dark"
            path = FIGURES / f"{name}{suffix}.png"
            fig.savefig(path)
            plt.close(fig)
            print(f"wrote {path.relative_to(ROOT)}")
