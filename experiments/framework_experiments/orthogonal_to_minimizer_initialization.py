from LowRankMF import LowRankMF
from ALSSolver import ALSSolver
import numpy as np

"""
Experiment: Orthogonal-to-Minimizer Initialization for ALS in MAP Matrix Factorization

This experiment studies the effect of initializing ALS in a subspace orthogonal
to the right-singular subspace of the global MAP minimizer.

Setup:
- We consider a ridge-regularized low-rank matrix factorization problem.
- The global MAP minimizer is characterized by the soft-thresholded SVD of Y.
- An initialization V0 is constructed to lie in the orthogonal complement of the
  minimizer's right-singular subspace.
- ALS is run from this initialization and compared to a standard random start.

Observation:
- From random initialization, ALS converges to the global MAP minimizer.
- From the orthogonal initialization, ALS converges to a distinct stationary point
  with strictly higher objective value and small gradient norm.

Interpretation:
- The orthogonal initialization restricts ALS to an invariant subspace that does
  not contain the global minimizer.
- ALS converges to the best solution within this subspace, which corresponds to a
  suboptimal stationary point (a saddle of the full MAP objective).
- This illustrates the presence of algorithm-dependent stability and invariant
  subspaces in the MAP loss landscape.

Conclusion:
- The MAP objective admits suboptimal stationary points indexed by singular
  subspaces.
- ALS can converge to these points when initialized in the corresponding invariant
  subspaces, even when starting orthogonal to the global minimizer.
"""


# SETUP PROBLEM
problem = LowRankMF(
    n=100, p=80,
    true_rank=10,
    R=3,
    lambda_reg=5.0,
    noise_level=0.1,
    seed=0
)

# SETUP INITIALIZATIONS AND SOLVERS

# 1) Random initialization (control)
solver_random = ALSSolver(problem, init_scale=0.1)
solver_random.fit(max_iter=200, tol=1e-8, verbose=True)

# 2) Orthogonal-to-minimizer initialization
V0_orth = problem.init_V_orthogonal_to_global_minimizer(
    matrix="Y",
    init_scale=0.1,
    epsilon=0.0
)
solver_orth = ALSSolver(problem, V0=V0_orth)
solver_orth.fit(max_iter=200, tol=1e-8, verbose=True)

# ANALYZE SOLUTIONS

print("\n=== RANDOM INIT ===")
report_random = solver_random.analyze_solution()
for k, v in report_random.items():
    print(k, ":", v)

print("\n=== ORTHOGONAL-TO-MINIMIZER INIT ===")
report_orth = solver_orth.analyze_solution()
for k, v in report_orth.items():
    print(k, ":", v)
