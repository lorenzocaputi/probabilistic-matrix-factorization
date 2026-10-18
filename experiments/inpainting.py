"""
Image inpainting with soft-impute (the application in applications/image_reconstruction).

The photo (rocket.jpg, 640 x 427: the DSCOVR launch, SpaceX, public domain)
loses its pixels in two ways: three quarters of them hidden at random
(seed 0), and the pixels under five lines of text. Each colour channel is a
427 x 640 matrix completed by soft-impute with no rank cap. lambda is chosen from a
grid by the PSNR on 5% of the *observed* pixels, held out from the fit; the
reported PSNR is on the hidden pixels.

Writes results/inpainting.csv and, for the chosen lambda of each case,
results/inpainting_<case>_masked.png and results/inpainting_<case>_completed.png.
"""

from _common import ROOT, RESULTS, parse_args, run_parallel, write_csv

import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(ROOT / "applications" / "image_reconstruction"))
from matrix_completion_inpaint import make_random_mask, make_text_mask, to_float01, to_uint8  # noqa: E402

from pmf.solvers import soft_impute  # noqa: E402

IMAGE = ROOT / "applications" / "image_reconstruction" / "rocket.jpg"
CASES = {"random": 0.75, "text": None}  # fraction hidden at random, or text
LAMBDAS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]


def load_image(width):
    img = Image.open(IMAGE).convert("RGB")
    w0, h0 = img.size
    if width != w0:
        img = img.resize((width, int(round(h0 * width / w0))), Image.BICUBIC)
    return to_float01(np.array(img))


def observed_mask(case, h, w):
    frac = CASES[case]
    return make_text_mask(h, w) if frac is None else make_random_mask(h, w, missing_frac=frac, seed=0)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(1.0 / mse))


def complete(img, observed, lam, max_iter):
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = soft_impute(img[:, :, c], observed, lam=lam, max_iter=max_iter, tol=1e-5)
    return np.clip(out, 0.0, 1.0)


def run(job):
    case, lam, width, max_iter = job
    img = load_image(width)
    h, w = img.shape[:2]
    observed = observed_mask(case, h, w)
    rng = np.random.default_rng(1)
    holdout = observed & (rng.random((h, w)) < 0.05)
    rec = complete(img, observed & ~holdout, lam, max_iter)
    # baseline: every hidden pixel set to its channel's mean over the observed pixels
    mean_fill = np.broadcast_to(img[observed].mean(axis=0), img.shape)
    return {"case": case, "lambda": lam, "missing_frac": round(float(1 - observed.mean()), 4),
            "holdout_psnr": psnr(rec[holdout], img[holdout]),
            "hidden_psnr": psnr(rec[~observed], img[~observed]),
            "mean_fill_hidden_psnr": psnr(mean_fill[~observed], img[~observed])}


def main():
    args = parse_args(__doc__.splitlines()[1])
    width, max_iter = (128, 30) if args.quick else (640, 300)
    lambdas = LAMBDAS[1::2] if args.quick else LAMBDAS
    rows = run_parallel(run, [(case, lam, width, max_iter) for case in CASES for lam in lambdas],
                        args.workers)
    table = pd.DataFrame(rows)
    best = table.loc[table.groupby("case").holdout_psnr.idxmax()]
    table["chosen"] = table.index.isin(best.index)
    print(table.to_string(index=False))
    write_csv(table, "inpainting.csv", args.quick)

    if not args.quick:
        # final pictures: refit on every observed pixel with each case's chosen lambda
        img = load_image(width)
        h, w = img.shape[:2]
        RESULTS.mkdir(exist_ok=True)
        for _, row in best.iterrows():
            observed = observed_mask(row["case"], h, w)
            rec = complete(img, observed, row["lambda"], max_iter)
            rec[observed] = img[observed]
            masked = img.copy()
            masked[~observed] = 1.0  # hidden pixels shown white
            for kind, pic in [("masked", masked), ("completed", rec)]:
                path = RESULTS / f"inpainting_{row['case']}_{kind}.png"
                Image.fromarray(to_uint8(pic)).save(path, optimize=True)
                print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
