# probabilistic-matrix-factorization

MAP estimation and matrix completion with probabilistic matrix factorization: notes on the loss landscape and ALS dynamics, and seven solvers compared on the same problems.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/hero-dark.png">
  <img alt="Three panels. (a) A landscape photo with half of its pixels hidden, and the same photo completed by soft-impute. (b) Relative error against iteration for seven solvers on a 300 by 200 matrix with half of its entries missing: ALS, gradient descent and soft-impute settle at 0.085, VB and Gibbs at 0.049, SVT dips to 0.050 and then rises as it fits the noise, the closed-form spectral estimate stays at 0.28. (c) Profiled loss of 200 projected-gradient runs started near a bad-pairing saddle at 35.43: most leave it by step 500, pass a second plateau and reach the optimum 33.25." src="figures/hero.png">
</picture>

Started next to a bad-pairing saddle of the diagonal-prior MAP loss (loss 35.43, optimum 33.25), 182 of 200 perturbed projected-gradient runs have left it by step 500, and 196 of 200 are within a relative 1e-6 of the global minimum by step 20,000.

On MovieLens-100K (random 80/20 splits, 5 seeds, mean ± sd of test RMSE in stars; predicting the training mean gives 1.122):

| solver | rank (chosen on validation) | test RMSE |
|---|---|---|
| Gibbs | 20 | 0.904 ± 0.003 |
| VB | 20 | 0.919 ± 0.002 |
| ALS | 5 | 0.938 ± 0.003 |
| gradient descent | 5 | 0.940 ± 0.004 |
| soft-impute | 5 | 0.942 ± 0.006 |
| SVT | 8 to 16 (early stopping) | 0.960 ± 0.004 |
| closed form, zero-filled | 10 | 1.045 ± 0.004 |

The rank and λ are chosen on a validation split of the training ratings. ALS, gradient descent and soft-impute minimize the same objective with the same λ; within their iteration budgets the last two end between 0.2% below and 0.82% above ALS's objective value. The Bayesian posterior means do better than all three at every rank ([RMSE against rank](figures/movielens_rmse.png)).

## Run it

```sh
git clone https://github.com/lorenzocaputi/probabilistic-matrix-factorization.git
cd probabilistic-matrix-factorization
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make check      # tests and a small run of every experiment; `make reproduce` reruns everything
```

`make reproduce` runs all experiments with 3 worker processes (about 35 minutes on a laptop; MovieLens is downloaded from files.grouplens.org into `data/`, which is not committed) and redraws the figures. `make figures` redraws them from the committed `results/`. The notebooks in `experiments/` need `pip install jupyter`.

## What the notes show

[notes/pmf.pdf](notes/pmf.pdf), 16 pages, Lorenzo Caputi and Omiros Papaspiliopoulos.

- The MAP estimate with Gaussian priors of precision λ on both factors solves the convex problem ½‖Y − Z‖²_F + λ‖Z‖_*, so it is the soft-thresholded SVD Z* = F diag((σ_i − λ)_+) Gᵀ, with factors F diag(√(σ_i − λ)_+) and G diag(√(σ_i − λ)_+) up to rotation.
- Started on scaled singular vectors, ALS never changes direction: each scale follows γ ← σγ / (λ + γ²), converges monotonically to √(σ − λ) (or to 0 when σ ≤ λ), at the linear rate |1 − 2λ/σ| per half-step. With a diagonal prior each coordinate does this on its own.
- In the maximum-likelihood case the stationary points are the R-dimensional invariant subspaces of YᵀY, points on a Grassmann manifold; every one other than the top-R subspace has a descent direction, so it is a saddle.
- In experiments with a 20% mask, soft-impute reached the same solution from all four starting points, including the ones on a saddle subspace, while complete-data ALS stayed in the subspace it started in.

## What I measured

- **ALS rate.** On complete data the ALS error contracts per iteration by max((σ_{R+1}/σ_R)², maxᵢ (1 − 2λ/σᵢ)²): the first term because span(V) follows orthogonal iteration on YᵀY, the second is the notes' scalar rate squared. Over 35 (λ, gap) settings the measured rate is within 1% of this in 29 and within 3% in 31; in the other four the measured rate is lower, by up to 29% ([figure](figures/convergence_rates.png)).
- **Landscape.** For a rank-4 fit with seven singular values above λ, the 99 stationary points all have loss ½‖Y‖²_F − ½ Σ_{i∈I} (σᵢ − λ)². Only the top-4 one has no negative Hessian eigenvalue, and in all 99 the number of negative eigenvalues equals a simple count: one for each pair of a used singular direction and a larger unused one, plus one for each zero column and unused direction above λ ([figure](figures/landscape.png)).
- **Saddle escape.** The 182 of 200 is a count at step 500. The runs leave the bad pairing after a few hundred steps, stall on a second plateau and reach the optimum after a median of 5,336 steps (4 of 200 are not there by step 20,000). Over prior scales 0.25 to 4 and ranks 2 to 8 (100 runs each), every run with R ≤ 3 reaches the optimum although none has left the saddle by step 500. With R = 8, 72 to 88 of 100 leave by step 500, but after 20,000 steps the median run is still 0.009 to 0.023 above the optimum, whose flattest direction has curvature 0.011 to 0.036 ([figure](figures/saddle_escape.png)).
- **Missing data.** ALS started exactly at a saddle of the complete-data loss still leaves it, after a median of 159 iterations, because the saddle amplifies rounding errors; gradient descent, whose gradient there is zero up to rounding, never moves. With 1% of the entries missing ALS reaches the best subspace in a median of 39 iterations, and with 5 to 40% missing in 26 to 30. Apart from that gradient-descent run, every run from every start ends at the best loss found ([figure](figures/missing_data.png)).
- **Convergence (hero, panel b).** With half of the entries missing, ALS, gradient descent and soft-impute reach the same MAP point (relative error 0.085); VB and the Gibbs posterior mean reach 0.049; SVT passes 0.050 at iteration 221 and then fits the noise. To get within 1e-6 of their own limit with 30% missing (median over the λ and gap grid), soft-impute needs 18 iterations, VB 34, ALS 63 and gradient descent 165; SVT does not get there in 3,000.

Sizes were kept small so that `make reproduce` runs on a laptop: synthetic matrices are at most 300 × 200, the saddle grid uses 100 runs per cell, and MovieLens uses 5 seeds.

## Layout

```
pmf/            LowRankMF problem, seven solvers (pmf/solvers/), landscape tools
experiments/    scripts that write results/*.csv, and the original notebooks
plotting/       one script per figure, reading results/
applications/   image inpainting and word co-occurrence completion with soft-impute
notes/          pmf.tex and pmf.pdf
```

## References

- N. Srebro, J. Rennie, T. Jaakkola. Maximum-margin matrix factorization. NeurIPS 2004 (proceedings 2005).
- R. Mazumder, T. Hastie, R. Tibshirani. Spectral regularization algorithms for learning large incomplete matrices. JMLR 11, 2010. (soft-impute)
- J.-F. Cai, E. J. Candès, Z. Shen. A singular value thresholding algorithm for matrix completion. SIAM J. Optim. 20(4), 2010. (SVT)
- S. Nakajima, M. Sugiyama, S. D. Babacan. Global solution of fully-observed variational Bayesian matrix factorization is column-wise independent. NeurIPS 2011.
- Y. J. Lim, Y. W. Teh. Variational Bayesian approach to movie rating prediction. KDD Cup and Workshop 2007. (VB)
- R. Salakhutdinov, A. Mnih. Bayesian probabilistic matrix factorization using Markov chain Monte Carlo. ICML 2008. (Gibbs)
- F. M. Harper, J. A. Konstan. The MovieLens datasets: history and context. ACM TiiS 5(4), 2015.

Cite this repository with [CITATION.cff](CITATION.cff).
