from LowRankMF import ALSSolver, LowRankMF
import numpy as np


problem = LowRankMF(
    n=100,
    p=80,
    true_rank=10,
    R=3,
    lambda_reg=5.0,
    singular_values=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    noise_level=0.1,
    seed=42,
)


solver = ALSSolver(problem, init_scale=0.01)
solver.fit(max_iter=200, tol=1e-6, verbose=True)

err = problem.relative_signal_error(solver.U, solver.V)
print("Relative error on Y_true:", err)

