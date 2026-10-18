"""Image inpainting: the photo, and for each way of hiding pixels the input and soft-impute's completion."""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from common import INK, RESULTS, ROOT, save_both

PHOTO = ROOT / "applications" / "image_reconstruction" / "rocket.jpg"
CASES = [("random", "three quarters of the pixels removed"),
         ("text", "text written over it")]


def show(ax, path):
    ax.imshow(Image.open(path))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw(mode):
    ink = INK[mode]
    fig = plt.figure(figsize=(8, 3.9))
    gs = GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[:, 0])
    show(ax, PHOTO)
    ax.set_title("original", fontsize=9.5)

    for col, (case, title) in enumerate(CASES, start=1):
        for row, kind in enumerate(["masked", "completed"]):
            ax = fig.add_subplot(gs[row, col])
            show(ax, RESULTS / f"inpainting_{case}_{kind}.png")
            if row == 0:
                ax.set_title(title, fontsize=9.5)
            if col == 1:
                ax.set_ylabel("input" if row == 0 else "completed", fontsize=9,
                              color=ink["secondary"])
    return fig


if __name__ == "__main__":
    save_both(draw, "inpainting")
