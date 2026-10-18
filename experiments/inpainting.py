"""
Image inpainting with soft-impute (the application in applications/image_reconstruction).

The photo is resized to 512 pixels wide and half of its pixels are hidden at
random (seed 0). Each colour channel is completed by soft-impute with no rank
cap. lambda is chosen from a grid by the PSNR on 5% of the *observed* pixels,
held out from the fit; the reported PSNR is on the hidden pixels.

Writes results/inpainting.csv and, for the chosen lambda,
results/inpainting_masked.png and results/inpainting_reconstructed.png.
"""

from _common import ROOT, RESULTS, parse_args, write_csv

import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(ROOT / "applications" / "image_reconstruction"))
from matrix_completion_inpaint import make_random_mask, to_float01, to_uint8  # noqa: E402

from pmf.solvers import soft_impute  # noqa: E402

IMAGE = ROOT / "applications" / "image_reconstruction" / "landscape.jpg"
MISSING = 0.5


def load_image(width=512):
    img = Image.open(IMAGE).convert("RGB")
    w0, h0 = img.size
    img = img.resize((width, int(round(h0 * width / w0))), Image.BICUBIC)
    return to_float01(np.array(img))


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(1.0 / mse))


def complete(img, observed, lam, max_iter):
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = soft_impute(img[:, :, c], observed, lam=lam, max_iter=max_iter, tol=1e-5)
    return np.clip(out, 0.0, 1.0)


def main():
    args = parse_args(__doc__.splitlines()[1])
    img = load_image(128 if args.quick else 512)
    h, w = img.shape[:2]
    observed = make_random_mask(h, w, missing_frac=MISSING, seed=0)
    rng = np.random.default_rng(1)
    holdout = observed & (rng.random((h, w)) < 0.05)
    fit_mask = observed & ~holdout
    max_iter = 30 if args.quick else 300

    # baseline: every hidden pixel set to its channel's mean over the observed pixels
    mean_fill = np.broadcast_to(img[observed].mean(axis=0), img.shape)

    rows, best = [], None
    for lam in [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
        rec = complete(img, fit_mask, lam, max_iter)
        row = {"lambda": lam, "missing_frac": MISSING,
               "holdout_psnr": psnr(rec[holdout], img[holdout]),
               "hidden_psnr": psnr(rec[~observed], img[~observed]),
               "mean_fill_hidden_psnr": psnr(mean_fill[~observed], img[~observed])}
        rows.append(row)
        print(row)
        if best is None or row["holdout_psnr"] > best[0]["holdout_psnr"]:
            best = (row, rec)
    table = pd.DataFrame(rows)
    table["chosen"] = table["lambda"] == best[0]["lambda"]
    write_csv(table, "inpainting.csv", args.quick)

    if not args.quick:
        # final picture: refit on every observed pixel with the chosen lambda
        rec = complete(img, observed, best[0]["lambda"], max_iter)
        rec[observed] = img[observed]
        masked = img.copy()
        masked[~observed] = 1.0  # hidden pixels shown white
        RESULTS.mkdir(exist_ok=True)
        Image.fromarray(to_uint8(masked)).save(RESULTS / "inpainting_masked.png", optimize=True)
        Image.fromarray(to_uint8(rec)).save(RESULTS / "inpainting_reconstructed.png", optimize=True)
        print("wrote results/inpainting_masked.png and results/inpainting_reconstructed.png")


if __name__ == "__main__":
    main()
