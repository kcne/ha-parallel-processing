import numba
import numpy as np

@numba.njit
def ackley(n,x=np.array([[]])):
    sum_sq = 0.0
    sum_cos = 0.0
    for i in range(10):
        sum_sq += x[i] ** 2
        sum_cos += np.cos(2 * np.pi * x[i])
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

# Generate random input data
n = 100
m = 10
x = np.random.rand(n, m)
fitnesses=np.zeros(n,dtype=np.float32)

threads_per_block = 256
blocks_per_grid = (m + (threads_per_block - 1))
# Compute the Ackley function

fitnesses=ackley(x,m)
# fitnesses = ackley(x, m)

print(fitnesses)  # prints (100,)