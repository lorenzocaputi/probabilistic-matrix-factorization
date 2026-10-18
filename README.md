<div align="center">

# Saddles, missing data and seven solvers for probabilistic matrix factorization

Lorenzo Caputi

Research assistant work supervised by Prof. Omiros Papaspiliopoulos, Bocconi University · [Notes (PDF)](notes/pmf.pdf)

</div>

**TL;DR.** Probabilistic matrix factorization fills in the missing entries of a matrix by modelling it as the product of two thin matrices. Its loss is not convex, but with complete data every stationary point other than the minimum is a saddle, and once entries are missing the solvers we tried always reach the best fit. On 100,000 movie ratings, seven solvers are compared: those that minimize the loss end up in the same place, and the two Bayesian ones, which average over fits instead of picking one, predict unseen ratings best.

This is one part of the work I did as a research assistant to Prof. Omiros Papaspiliopoulos at Bocconi University, which I found interesting enough to put on GitHub. The notes are written with him.

## Abstract

We study maximum a posteriori (MAP) estimation in probabilistic matrix factorization, where an $n \times p$ matrix with missing entries is modelled as $UV^\top$ plus noise, with Gaussian priors on $U$ and $V$. With every entry observed, the MAP estimate is the singular value decomposition of the data with each singular value shrunk by λ, the solution of a convex nuclear-norm problem, and every other stationary point of the non-convex loss is a saddle that pairs the fit with the wrong singular vectors. Started next to such a saddle, 196 of 200 gradient runs still reach the minimum, and with as little as 1% of the entries missing, alternating least squares, gradient descent and soft-impute reach the best solution from every starting point we tried. We then compare seven solvers on a simulated matrix and on the 100,000 ratings of MovieLens. The three that minimize the MAP loss reach almost the same predictions (test error 0.938 to 0.942 stars), and the two Bayesian methods, variational Bayes and Gibbs sampling, predict better at every rank (0.919 and 0.904 stars, against 1.122 for predicting the average rating).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/hero-dark.png">
  <img alt="Two panels. (a) Error against the true matrix as each algorithm runs on a simulated 300 by 200 matrix with half of its entries hidden: ALS, gradient descent and soft-impute settle at 0.085, VB and Gibbs at 0.049, SVT reaches 0.05 and then rises as it fits the noise, the closed form stays at 0.28. (b) Test error on held-out MovieLens ratings, in stars: Gibbs 0.904, VB 0.919, ALS 0.938, gradient descent 0.940, soft-impute 0.942, SVT 0.960, closed form 1.045, against 1.122 for predicting the average rating." src="figures/hero.png">
</picture>

**Figure 1.** (a) A simulated 300 × 200 matrix of rank 5 plus noise, with half of its entries hidden: how far each solver's estimate is from the true matrix as it runs. (b) The error of each solver when it predicts MovieLens ratings it has not seen.

## 1. Problem

We observe some entries of an $n \times p$ matrix $Y$ and want to predict the rest. In MovieLens [5] the rows are 943 users, the columns are 1,682 movies, and 6% of the entries are observed: the 100,000 ratings, from 1 to 5 stars.

The model gives every row $i$ a vector $u_i$ and every column $j$ a vector $v_j$, both of a small length $r$, and says that each entry is their dot product plus noise:

$$Y_{ij} = u_i^\top v_j + \varepsilon_{ij}, \qquad \varepsilon_{ij} \sim N(0, 1), \qquad u_i,\ v_j \sim N(0, \lambda^{-1} I).$$

So $Y$ is close to a matrix $UV^\top$ of rank $r$, and the prior keeps $U$ and $V$ small. For movies, $v_j$ describes a movie by $r$ numbers, $u_i$ describes how much a user likes each of them, and the predicted rating of an unseen movie is $u_i^\top v_j$. The most probable $U$ and $V$ given the observed entries, the MAP estimate, minimize

$$L(U, V) = \frac{1}{2} \sum_{(i,j) \text{ observed}} \left(Y_{ij} - u_i^\top v_j\right)^2 + \frac{\lambda}{2} \left(\lVert U \rVert_F^2 + \lVert V \rVert_F^2\right),$$

a least-squares fit plus a penalty that shrinks $U$ and $V$ towards zero. $L$ is not convex, so it is not obvious that an algorithm will find its minimum.

## 2. The shape of the loss

With every entry observed, the minimum has a closed form. Minimizing $L$ gives the same matrix $UV^\top$ as the convex problem $\min_Z \frac{1}{2}\lVert Y - Z \rVert_F^2 + \lambda \lVert Z \rVert_*$, whose solution is the singular value decomposition of $Y$ with every singular value σ replaced by max(σ − λ, 0).

Every other stationary point is a saddle, where the fit uses the wrong singular vectors of $Y$ (proved in the notes for λ = 0 and checked numerically for λ > 0). In exact arithmetic, alternating least squares started on wrong singular vectors stays on them, but a small random perturbation is usually enough to escape: in a version of the model with a different λ for each column, 196 of 200 gradient runs started next to a saddle reach the minimum ([figure](figures/saddle_escape.png)).

Missing entries remove the trap. Filling in the missing entries mixes the singular directions, and with even 1% of the entries missing, ALS, gradient descent and soft-impute reach the best solution from every starting point we tried, saddles included ([figure](figures/missing_data.png)).

## 3. Solvers

**Table 1.** The seven solvers.

| Solver | How it fits the model |
|---|---|
| Closed form | the closed form above, applied once with the missing entries set to zero |
| Alternating least squares (ALS) | solves for $U$ with $V$ fixed, then for $V$ with $U$ fixed; each half-step is a ridge regression |
| Gradient descent | takes gradient steps on $U$ and $V$ together |
| Soft-impute [1] | fills the missing entries with the current estimate, then applies the closed form |
| Singular value thresholding, SVT [2] | a related convex method whose rank grows as it runs; stopped when the validation error is lowest |
| Variational Bayes, VB [3] | approximates the posterior of $U$ and $V$ by a simpler distribution and predicts with its mean |
| Gibbs sampling [4] | draws $U$ and $V$ from their posterior and averages the predictions |

The first five give a single estimate of $U$ and $V$; the last two average over all the $U$ and $V$ that fit the data reasonably well.

## 4. Results

**Table 2.** Test error on MovieLens, in stars.

| Solver | Test error |
|---|---:|
| Gibbs sampling | 0.904 |
| Variational Bayes | 0.919 |
| Alternating least squares | 0.938 |
| Gradient descent | 0.940 |
| Soft-impute | 0.942 |
| Singular value thresholding | 0.960 |
| Closed form | 1.045 |
| Predicting the average rating | 1.122 |

The error is the root mean squared error on a random 20% of the ratings held out for testing, so an error of 0.9 means a typical prediction is about 0.9 stars off. Each number is the mean over 5 random splits (standard deviation 0.002 to 0.006), and the rank $r$ and λ were chosen for each solver on part of the training ratings.

ALS, gradient descent and soft-impute minimize the same loss $L$, so, as Section 2 suggests, they end up with almost the same predictions. Gibbs sampling and variational Bayes do not commit to one $U$ and $V$, and that predicts unseen ratings better at every rank ([error against rank](figures/movielens_rmse.png)). The simulated matrix of Figure 1(a) shows the same order: the three MAP solvers stop 8.5% away from the true matrix, the two Bayesian ones 4.9% away, and SVT gets to 5% before it starts fitting the noise.

## 5. Application: filling in a photo

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/inpainting-dark.png">
  <img alt="A photo of a rocket on its launch pad at dusk. In the middle column, three quarters of its pixels are removed at random, and below it the photo completed by soft-impute: the rocket, the towers and the lights are back, slightly grainy. In the right column, five lines of white text are written over the photo, and below it the completed photo with the text gone except for faint traces." src="figures/inpainting.png">
</picture>

**Figure 2.** A photo completed by soft-impute after three quarters of its pixels are removed at random (middle) and after text is written over it (right). Photo: the launch of DSCOVR on a Falcon 9, February 2015, by SpaceX, in the public domain.

A colour photo is three matrices of pixel values, one per colour, and a photo made of long straight lines like this launch pad is close to a low-rank matrix. Filling in hidden pixels is then the same problem as predicting unseen ratings. In the middle column, soft-impute rebuilds the photo from the quarter of the pixels that is left. In the right column, the pixels under the text are treated as missing, which removes the text. The code is in [`applications/image_reconstruction`](applications/image_reconstruction), and this runs it on your own photo:

```sh
python applications/image_reconstruction/matrix_completion_inpaint.py --input photo.jpg --output completed.png --mask random --missing-frac 0.75
```

## Reproducing the results

With Python 3.11 or later:

```sh
git clone https://github.com/lorenzocaputi/probabilistic-matrix-factorization.git
cd probabilistic-matrix-factorization
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make reproduce
```

`make reproduce` reruns every experiment and redraws every figure in about 40 minutes on a laptop; it downloads MovieLens from grouplens.org. `make check` runs the tests and a small version of each experiment in about a minute, and `make figures` redraws the figures from the saved results in `results/`.

## References

1. R. Mazumder, T. Hastie and R. Tibshirani. Spectral regularization algorithms for learning large incomplete matrices. *Journal of Machine Learning Research* 11, 2287–2322, 2010.
2. J.-F. Cai, E. J. Candès and Z. Shen. A singular value thresholding algorithm for matrix completion. *SIAM Journal on Optimization* 20(4), 1956–1982, 2010.
3. Y. J. Lim and Y. W. Teh. Variational Bayesian approach to movie rating prediction. *Proceedings of KDD Cup and Workshop*, 2007.
4. R. Salakhutdinov and A. Mnih. Bayesian probabilistic matrix factorization using Markov chain Monte Carlo. *ICML*, 2008.
5. F. M. Harper and J. A. Konstan. The MovieLens datasets: history and context. *ACM Transactions on Interactive Intelligent Systems* 5(4), 19, 2015.
