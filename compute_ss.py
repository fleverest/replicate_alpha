import os

import numpy as np
import polars as pl
import multiprocessing as mp

from numpy.random import SeedSequence, default_rng

from alpha_implementations import new_alpha, old_alpha
from params import thetas, etas, ds, reps, alpha

datasets = [
    {"theta": theta, "i": i}
    for theta in thetas
    for i in range(reps)
]


def compute_ss(params):
    print(params)
    theta = params["theta"]
    i = params["i"]
    in_filename = f"table2_data/theta{theta}_{i}.npz"
    out_filename = f"table2_results/theta{theta}_{i}.csv"
    if os.path.isfile(out_filename):
        # Skip completed computations
        return
    x = np.array(np.load(in_filename)["x"], dtype=np.float64)
    df_eta = []
    df_d = []
    df_ss = []
    for eta in etas:
        for d in ds:
            df_eta.append(eta)
            df_d.append(d)
            df_ss.append(new_alpha(x, eta, d))
    pl.DataFrame({
        "eta": df_eta,
        "d": df_d,
        "ss": df_ss
    }).write_csv(out_filename)


if __name__ == "__main__":
    pool = mp.Pool()
    pool.map(compute_ss, datasets)
