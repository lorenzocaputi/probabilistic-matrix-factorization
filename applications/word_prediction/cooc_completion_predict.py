#!/usr/bin/env python3
"""
Text prediction using co-occurrence matrices + matrix completion (Soft-Impute).

What it does:
1) Reads a text corpus from --corpus (plain text).
2) Tokenizes and builds a vocabulary (top --vocab-size words).
3) Builds a word-context co-occurrence matrix C (center word x context word).
4) Hides a random fraction of entries in C (simulating missing co-occurrences).
5) Runs Soft-Impute to reconstruct the matrix.
6) Predicts [MASK] tokens in --queries using the reconstructed associations.

Example:
  python cooc_completion_predict.py \
    --corpus corpus.txt \
    --queries queries.txt \
    --window 4 \
    --vocab-size 8000 \
    --missing-frac 0.6 \
    --lam 5.0 \
    --rank-max 200 \
    --max-iter 150

queries.txt format: one sentence per line, containing exactly one [MASK]
  the cat sat on the [MASK]
  she poured [MASK] into the cup
"""

import argparse
import re
from collections import Counter
import numpy as np


MASK_TOKEN = "[mask]"


def tokenize(text: str):
    # Simple, robust tokenizer: lowercase words + apostrophes
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def build_vocab(tokens, vocab_size: int, min_count: int):
    counts = Counter(tokens)
    items = [(w, c) for w, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:vocab_size]
    vocab = [w for w, _ in items]
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i, counts


def build_cooc_matrix(tokens, w2i, window: int):
    """
    C[i, j] = # times vocab word i appears with context vocab word j within +/- window
    """
    V = len(w2i)
    C = np.zeros((V, V), dtype=np.float64)

    idxs = [w2i.get(t, -1) for t in tokens]
    n = len(idxs)
    for center_pos, center_id in enumerate(idxs):
        if center_id < 0:
            continue
        left = max(0, center_pos - window)
        right = min(n, center_pos + window + 1)
        for ctx_pos in range(left, right):
            if ctx_pos == center_pos:
                continue
            ctx_id = idxs[ctx_pos]
            if ctx_id < 0:
                continue
            C[center_id, ctx_id] += 1.0
    return C


def soft_threshold_singular_values(A: np.ndarray, lam: float, rank_max: int | None):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if rank_max is not None:
        U = U[:, :rank_max]
        s = s[:rank_max]
        Vt = Vt[:rank_max, :]
    s2 = np.maximum(s - lam, 0.0)
    if np.all(s2 == 0):
        return np.zeros_like(A)
    return (U * s2) @ Vt


def soft_impute(M_obs: np.ndarray, Omega: np.ndarray, lam: float,
                max_iter: int, tol: float, rank_max: int | None, verbose: bool):
    """
    Soft-Impute matrix completion:
      iterate X <- S_lambda( P_Omega(M_obs) + P_notOmega(X) )
    """
    X = np.zeros_like(M_obs, dtype=np.float64)
    # init missing entries with mean of observed
    if np.any(Omega):
        mean_val = float(np.mean(M_obs[Omega]))
        X[~Omega] = mean_val
        X[Omega] = M_obs[Omega]

    prev = X.copy()
    eps = 1e-12

    for it in range(1, max_iter + 1):
        Z = X.copy()
        Z[Omega] = M_obs[Omega]

        X = soft_threshold_singular_values(Z, lam=lam, rank_max=rank_max)

        # keep observed exactly
        X[Omega] = M_obs[Omega]

        rel = np.linalg.norm(X - prev) / (np.linalg.norm(prev) + eps)
        if verbose and (it == 1 or it % 10 == 0 or rel < tol):
            print(f"iter={it:4d} rel_change={rel:.3e}")
        if rel < tol:
            break
        prev = X.copy()

    return X


def make_observation_mask(shape, missing_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    Omega = rng.random(shape) >= missing_frac  # True = observed
    # Ensure diagonal observed (stability; also common to keep self-cooc irrelevant but stable)
    d = min(shape[0], shape[1])
    Omega[np.arange(d), np.arange(d)] = True
    return Omega


def score_candidates(masked_sentence_tokens, mask_index, vocab, w2i, assoc_matrix,
                     stopwords_set: set[str], top_k: int, exclude_context: bool = True):
    """
    Predict [MASK] using surrounding context words:
      score(w) = sum_{c in context} assoc[w, c]
    where assoc is the completed co-occurrence matrix.

    Notes:
    - This is a deliberately simple scoring rule.
    - Works best when the corpus is topical / consistent.
    """
    ctx_words = []
    for i, t in enumerate(masked_sentence_tokens):
        if i == mask_index:
            continue
        if t in w2i:
            ctx_words.append(t)

    if not ctx_words:
        return []

    ctx_ids = [w2i[t] for t in ctx_words]

    # Candidate set: all vocab words except stopwords (optional) and context words (optional)
    banned = set()
    if exclude_context:
        banned.update(ctx_words)
    banned.update(stopwords_set)

    scores = []
    for w in vocab:
        if w in banned:
            continue
        wi = w2i[w]
        s = float(np.sum(assoc_matrix[wi, ctx_ids]))
        scores.append((w, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def default_stopwords():
    # Tiny list; feel free to expand.
    return {
        "the","a","an","and","or","to","of","in","on","for","with","as","at","by","from",
        "is","are","was","were","be","been","it","this","that","these","those","i","you",
        "he","she","they","we","my","your","his","her","their","our","not"
    }


def parse_args():
    p = argparse.ArgumentParser(description="Co-occurrence matrix completion + [MASK] prediction")
    p.add_argument("--corpus", required=True, help="Path to corpus text file.")
    p.add_argument("--queries", required=True, help="Path to queries file (one sentence per line with [MASK]).")

    p.add_argument("--window", type=int, default=4, help="Context window radius.")
    p.add_argument("--vocab-size", type=int, default=8000, help="Max vocabulary size.")
    p.add_argument("--min-count", type=int, default=5, help="Min count to include a word in vocab.")

    p.add_argument("--missing-frac", type=float, default=0.6, help="Fraction of co-occurrence entries to hide.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for masking co-occurrence entries.")

    p.add_argument("--lam", type=float, default=5.0, help="Soft-Impute shrinkage parameter.")
    p.add_argument("--rank-max", type=int, default=200, help="Rank cap (0 disables).")
    p.add_argument("--max-iter", type=int, default=150, help="Max iterations.")
    p.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance.")
    p.add_argument("--top-k", type=int, default=10, help="Top-K predictions to print.")

    p.add_argument("--no-stopwords", action="store_true", help="Do not exclude stopwords from predictions.")
    p.add_argument("--verbose", action="store_true", help="Print Soft-Impute progress.")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.corpus, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    tokens = tokenize(text)

    vocab, w2i, counts = build_vocab(tokens, args.vocab_size, args.min_count)
    print(f"Corpus tokens: {len(tokens):,}")
    print(f"Vocab size:    {len(vocab):,} (min_count={args.min_count})")

    # Re-tokenize using vocab filter (optional but keeps matrix consistent)
    tokens_in_vocab = [t for t in tokens if t in w2i]

    C = build_cooc_matrix(tokens_in_vocab, w2i, window=args.window)
    print(f"Built co-occurrence matrix: {C.shape}  (nonzero ~ {np.count_nonzero(C):,})")

    Omega = make_observation_mask(C.shape, missing_frac=args.missing_frac, seed=args.seed)
    C_obs = C.copy()
    # Missing entries are ignored via Omega; values here don't matter
    C_obs[~Omega] = 0.0

    rank_max = None if args.rank_max == 0 else int(args.rank_max)
    print(f"Running Soft-Impute: missing_frac={args.missing_frac}, lam={args.lam}, rank_max={rank_max}")
    C_hat = soft_impute(C_obs, Omega, lam=float(args.lam),
                        max_iter=int(args.max_iter), tol=float(args.tol),
                        rank_max=rank_max, verbose=bool(args.verbose))

    # Prediction phase
    stopwords_set = set() if args.no_stopwords else default_stopwords()

    with open(args.queries, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    print("\n=== Predictions ===")
    for line in lines:
        if MASK_TOKEN not in line:
            print(f"\nSkipping (no {MASK_TOKEN}): {line}")
            continue

        # Keep [mask] as a token; tokenize words separately
        parts = line.replace(MASK_TOKEN, f" {MASK_TOKEN} ")
        toks = re.findall(r"\[mask\]|[a-z]+(?:'[a-z]+)?", parts.lower())
        if toks.count(MASK_TOKEN) != 1:
            print(f"\nSkipping (need exactly one {MASK_TOKEN}): {line}")
            continue

        mpos = toks.index(MASK_TOKEN)
        preds = score_candidates(toks, mpos, vocab, w2i, C_hat,
                                 stopwords_set=stopwords_set,
                                 top_k=int(args.top_k))

        print(f"\nQuery: {line}")
        if not preds:
            print("  (no context words in vocab)")
            continue
        for i, (w, s) in enumerate(preds, 1):
            print(f"  {i:2d}. {w:<15} score={s:.3f}")


if __name__ == "__main__":
    main()
