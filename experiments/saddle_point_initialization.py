from LowRankMF import LowRankMF
from ALSSolver import ALSSolver
import numpy as np

""" 
This experiment runs ALS on a LowRankMF problem instance from different initializations: 
    - random initialization (should converge to global optimum)
    - saddle point with random scaling (should converge to the saddle point at that same subspace, it just rescales the singular vectors)
    - saddle point correctly scaled with soft-thresholding (should converge to the same saddle point, as it is already stationary).
    
It then analyzes the solutions found.
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

# SETUP INITIALIZATIONS AND SOLVERS

# random initialization
solver = ALSSolver(problem, init_scale=0.1)
solver.fit(max_iter=200, tol=1e-8, verbose=True)

# Choose a "suboptimal" singular subset by hand
index_set = [2, 3, 4]  

# Random diagonal scaling in that subspace
V0_saddle_scaled = problem.init_V_from_svd_directions(
    index_set=index_set,
    matrix="Y",
    scaling="random",
    scale_range=(0.1, 10.0),
)

# Soft-thresholded stationary candidate in that subspace (stationary point that is a saddle)
V0_saddle = problem.init_V_from_svd_directions(
    index_set=index_set,
    matrix="Y",
    scaling="soft_threshold",
)

solver_saddle_scaled = ALSSolver(problem, V0=V0_saddle_scaled)
solver_saddle_scaled.fit(max_iter=200, tol=1e-8, verbose=True)

solver_saddle = ALSSolver(problem, V0=V0_saddle)
solver_saddle.fit(max_iter=200, tol=1e-8, verbose=True)


# ANALYZE SOLUTIONS

# random initialization --> global optimum
report = solver.analyze_solution()
for k, v in report.items():
    print(k, ":", v)

# scaled saddle point initialization --> saddle point (correct scale)
report_saddle_scaled = solver_saddle_scaled.analyze_solution()
for k, v in report_saddle_scaled.items():
    print(k, ":", v)

# saddle point initialization --> same saddle point
report_saddle = solver_saddle.analyze_solution()
for k, v in report_saddle.items():
    print(k, ":", v)
