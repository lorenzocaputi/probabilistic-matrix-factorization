from LowRankMF import LowRankMF
import numpy as np


Y = np.random.randn(8, 10)
model = LowRankMF(Y, R=3, lambda_reg=0.1)
model.fit_als(20)
print("Final loss:", model.loss())
