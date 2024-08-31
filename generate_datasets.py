import numpy as np
from scipy.stats import bernoulli

from params import thetas, block_size, reps

np.random.seed(123456789)
# To replicate random state of original ipynb
_ = bernoulli.rvs(1 / 2, size=20)


for theta in thetas:
    for i in range(reps):
        x = np.array([])
        x = np.append(x, bernoulli.rvs(theta, size=block_size))
        np.savez_compressed(
            f"table2_data/theta{theta}_{i}.npz", x=np.array(x, dtype=bool)
        )
