"""Shared setup for the experiment scripts: paths, worker pool, single-threaded BLAS."""

import os

# One BLAS thread per worker process; the scripts parallelize over runs instead.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def parse_args(description, **extra):
    """--workers (default 3) and --quick (a small version for smoke tests)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workers", type=int, default=3, help="worker processes (default 3)")
    parser.add_argument("--quick", action="store_true", help="tiny run for testing the pipeline")
    for name, kwargs in extra.items():
        parser.add_argument(f"--{name.replace('_', '-')}", **kwargs)
    return parser.parse_args()


def run_parallel(func, jobs, workers):
    """Map func over jobs with a process pool (in order); workers=1 runs inline."""
    if workers <= 1:
        return [func(job) for job in jobs]
    with Pool(workers) as pool:
        return pool.map(func, jobs, chunksize=1)


def write_csv(df, name, quick=False):
    """Write a results table; quick runs go to results/quick/ so they never overwrite real results."""
    folder = RESULTS / "quick" if quick else RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {path.relative_to(ROOT)} ({len(df)} rows)")
    return path
