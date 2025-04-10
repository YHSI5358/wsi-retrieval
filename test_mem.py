import numpy as np

test = np.zeros((3502295, 1024), dtype=np.float32)


print(test.nbytes / 1024 / 1024)

import cupy as cp
from cuvs.neighbors import cagra

n_samples = 3502295
n_features = 1024
n_queries = 1000
k = 10
dataset = cp.random.random_sample((n_samples, n_features),
                                  dtype=cp.float32)

print(dataset.nbytes / 1024 / 1024)
