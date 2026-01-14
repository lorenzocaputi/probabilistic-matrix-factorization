#!/usr/bin/env python3
"""
Matrix completion for image inpainting using Soft-Impute (iterative SVD + shrinkage).

Usage examples:
  # Inpaint a rectangular missing region
  python matrix_completion_inpaint.py --input input.jpg --output out.png --mask rect \
    --rect 0.25 0.25 0.5 0.5 --lam 30 --max-iter 200

  # Inpaint random missing pixels (e.g., 60% missing)
  python matrix_completion_inpaint.py --input input.jpg --output out.png --mask random \
    --missing-frac 0.6 --lam 20 --max-iter 200

Notes:
- Soft-Impute assumes the image (or channel) is approximately low-rank.
- Works best for smooth/structured regions; struggles on high-frequency textures.
"""

import argparse
import numpy as np
from PIL import Image
import os


def soft_threshold_singular_values(A: np.ndarray, lam: float, rank_max: int | None = None) -> np.ndarray:
    """Return S_lambda(A): shrink singular values by lam (soft-threshold)."""
    # Full SVD (fine for moderate images). For large images, consider resizing or randomized SVD.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    if rank_max is not None:
        U = U[:, :rank_max]
        s = s[:rank_max]
        Vt = Vt[:rank_max, :]

    s_shrunk = np.maximum(s - lam, 0.0)
    if np.all(s_shrunk == 0):
        return np.zeros_like(A)

    return (U * s_shrunk) @ Vt


def soft_impute(M_obs: np.ndarray, Omega: np.ndarray, lam: float, max_iter: int = 200,
                tol: float = 1e-4, rank_max: int | None = None, verbose: bool = True) -> np.ndarray:
    """
    Soft-Impute for matrix completion.

    M_obs: observed matrix with arbitrary values in missing entries (they will be ignored by Omega)
    Omega: boolean mask where True indicates observed entries
    lam: shrinkage parameter (higher => lower rank, smoother)
    """
    # Initialize with observed values, zeros elsewhere
    X = np.zeros_like(M_obs, dtype=np.float64)
    X[Omega] = M_obs[Omega]

    # Optional: initialize missing entries with mean of observed
    if np.any(Omega):
        mean_val = float(np.mean(M_obs[Omega]))
        X[~Omega] = mean_val

    prev_X = X.copy()
    eps = 1e-12

    for it in range(1, max_iter + 1):
        # Fill observed entries with ground truth, keep current estimates elsewhere
        Z = X.copy()
        Z[Omega] = M_obs[Omega]

        # Low-rank shrinkage step
        X = soft_threshold_singular_values(Z, lam=lam, rank_max=rank_max)

        # Enforce observed entries exactly (common in practice)
        X[Omega] = M_obs[Omega]

        # Convergence check (relative change)
        num = np.linalg.norm(X - prev_X)
        den = np.linalg.norm(prev_X) + eps
        rel = num / den

        if verbose and (it == 1 or it % 10 == 0 or rel < tol):
            # compute reconstruction error on observed set (should be 0 due to enforcement)
            print(f"iter={it:4d}  rel_change={rel:.3e}")

        if rel < tol:
            break

        prev_X = X.copy()

    return X


def make_rect_mask(h: int, w: int, rect: tuple[float, float, float, float]) -> np.ndarray:
    """
    rect specified as (x, y, width, height) in relative [0,1] units.
    Returns Omega mask where True = observed.
    """
    x, y, rw, rh = rect
    x0 = int(round(x * w))
    y0 = int(round(y * h))
    x1 = int(round((x + rw) * w))
    y1 = int(round((y + rh) * h))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    Omega = np.ones((h, w), dtype=bool)
    Omega[y0:y1, x0:x1] = False
    return Omega


def make_random_mask(h: int, w: int, missing_frac: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Omega = rng.random((h, w)) >= missing_frac  # True observed
    # Ensure at least something observed
    if not np.any(Omega):
        Omega[rng.integers(0, h), rng.integers(0, w)] = True
    return Omega


def to_float01(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 255.0


def to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def inpaint_image_matrix_completion(img: np.ndarray, Omega: np.ndarray, lam: float,
                                    max_iter: int, tol: float, rank_max: int | None,
                                    verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    img: HxW (grayscale) or HxWxC (RGB) uint8
    Omega: HxW bool mask (True observed)
    Returns (masked_image_uint8, reconstructed_image_uint8)
    """
    img01 = to_float01(img)
    h, w = img01.shape[:2]

    # Create a visible "masked" version (missing entries set to 0)
    masked01 = img01.copy()
    if img01.ndim == 2:
        masked01[~Omega] = 0.0
    else:
        masked01[~Omega, :] = 0.0

    # Reconstruct each channel independently (simple baseline)
    if img01.ndim == 2:
        X = soft_impute(img01, Omega, lam=lam, max_iter=max_iter, tol=tol,
                        rank_max=rank_max, verbose=verbose)
        rec01 = X
    else:
        rec01 = np.zeros_like(img01)
        for c in range(img01.shape[2]):
            if verbose:
                print(f"\nChannel {c}:")
            Xc = soft_impute(img01[:, :, c], Omega, lam=lam, max_iter=max_iter, tol=tol,
                             rank_max=rank_max, verbose=verbose)
            rec01[:, :, c] = Xc

    return to_uint8(masked01), to_uint8(rec01)


def parse_args():
    p = argparse.ArgumentParser(description="Image inpainting via matrix completion (Soft-Impute).")
    p.add_argument("--input", required=True, help="Path to input image.")
    p.add_argument("--output", required=True, help="Path to save reconstructed image (png recommended).")
    p.add_argument("--masked-output", default=None, help="Optional: save the masked image too.")

    p.add_argument("--mask", choices=["rect", "random"], default="rect", help="Mask type.")
    p.add_argument("--rect", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                   default=[0.25, 0.25, 0.5, 0.5],
                   help="Rect mask in relative coords: x y w h (each in [0,1]).")
    p.add_argument("--missing-frac", type=float, default=0.6, help="Fraction missing for random mask (0..1).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for random mask.")

    p.add_argument("--lam", type=float, default=25.0, help="Shrinkage strength (larger => smoother/lower-rank).")
    p.add_argument("--max-iter", type=int, default=200, help="Max Soft-Impute iterations.")
    p.add_argument("--tol", type=float, default=1e-4, help="Relative-change tolerance.")
    p.add_argument("--rank-max", type=int, default=0,
                   help="Optional cap on rank used in SVD (0 means no cap). Speeds up but may reduce quality.")
    p.add_argument("--resize-max", type=int, default=512,
                   help="Resize so max(H,W) <= this (0 disables). Helps runtime.")
    p.add_argument("--verbose", action="store_true", help="Print iteration progress.")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = args.input
    out_path = args.output
    masked_path = args.masked_output

    img_pil = Image.open(in_path).convert("RGB")  # use RGB consistently
    w0, h0 = img_pil.size

    if args.resize_max and max(w0, h0) > args.resize_max:
        scale = args.resize_max / float(max(w0, h0))
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        img_pil = img_pil.resize((new_w, new_h), Image.BICUBIC)

    img = np.array(img_pil)  # HxWx3 uint8
    h, w = img.shape[:2]

    if args.mask == "rect":
        Omega = make_rect_mask(h, w, tuple(args.rect))
    else:
        Omega = make_random_mask(h, w, missing_frac=args.missing_frac, seed=args.seed)

    rank_max = None if args.rank_max == 0 else int(args.rank_max)

    masked, rec = inpaint_image_matrix_completion(
        img=img,
        Omega=Omega,
        lam=float(args.lam),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        rank_max=rank_max,
        verbose=bool(args.verbose),
    )

    # Save outputs
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(rec).save(out_path)

    if masked_path is not None:
        os.makedirs(os.path.dirname(masked_path) or ".", exist_ok=True)
        Image.fromarray(masked).save(masked_path)

    print(f"\nSaved reconstructed: {out_path}")
    if masked_path is not None:
        print(f"Saved masked:        {masked_path}")


if __name__ == "__main__":
    main()
