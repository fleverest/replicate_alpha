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
    x = np.array(np.load(f"table2_data/theta{theta}_{i}.npz")["x"], dtype=np.float64)
    old = old_alpha(x, eta, d)
    old_ss = np.argmax(old > 1 / alpha)
    new = new_alpha(x, eta, d)
    new_ss = np.argmax(new > 1 / alpha)
    np.savez_compressed(
        f"table2_results/theta{theta}_eta{eta}_d{d}_{i}.npz",
        old=old_ss,
        new=new_ss,
    )


if __name__ == "__main__":
    pool = mp.Pool()
    pool.map(compute_ss, experiments)
