from LowRankMF import ALSSolver, LowRankMF
import numpy as np


problem = LowRankMF(n=100,
                    p=80, 
                    true_rank=50,
                    R=5, 
                    lambda_reg=5.0, 
                    noise_level=0.1,
                    )

solver = ALSSolver(problem, init_scale=0.01)
solver.fit(max_iter=200, tol=1e-6, verbose=True)

report = solver.analyze_solution()
for k, v in report.items():
    print(k, ":", v)


