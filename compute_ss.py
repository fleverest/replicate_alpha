import os

import numpy as np
import multiprocessing as mp

from numpy.random import SeedSequence, default_rng

from alpha_implementations import new_alpha, old_alpha
from params import thetas, etas, ds, reps, alpha

experiments = [
    {"theta": theta, "eta": eta, "d": d, "i": i}
    for theta in thetas
    for eta in etas
    for d in ds
    for i in range(reps)
]


def compute_ss(params):
    print(params)
    theta = params["theta"]
    i = params["i"]
    eta = params["eta"]
    d = params["d"]
    in_filename = f"table2_data/theta{theta}_{i}.npz"
    out_filename = f"table2_results/theta{theta}_eta{eta}_d{d}_{i}.txt"
    if os.path.isfile(out_filename):
        # Skip completed computations
        return
    x = np.array(np.load(in_filename)["x"], dtype=np.float64)
    ss = new_alpha(x, eta, d)
    print(ss)
    with open(out_filename, "w") as f:
        f.write(str(ss))


if __name__ == "__main__":
    pool = mp.Pool()
    pool.map(compute_ss, experiments)
