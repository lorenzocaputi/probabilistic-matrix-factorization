from LowRankMF import ALSSolver, LowRankMF
import numpy as np


problem = LowRankMF(n=100,
                    p=80, 
                    true_rank=10,
                    R=3, 
                    lambda_reg=5.0, 
                    )

solver = ALSSolver(problem, init_scale=0.01)
solver.fit(max_iter=200, tol=1e-6, verbose=False)

report = solver.analyze_solution()
for k, v in report.items():
    print(k, ":", v)


