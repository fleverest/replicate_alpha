import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import polars as pl


def read_martingales(datafile):
    data = np.load(datafile)
    return pl.DataFrame(
        {
            "Time": np.append(np.arange(data["new"].size), np.arange(data["old"].size)),
            "Method": (["New"] * data["new"].size) + (["Old"] * data["old"].size),
            "Martingale": np.append(data["new"], data["old"]),
        }
    )


results_dir = "table2_results/"
filenames = [results_dir + filename for filename in os.listdir(results_dir)]

for fn in filenames[:10]:
    out_filename = fn.split("/")[-1].replace("npz", "png")
    df = read_martingales(fn)
    plt.figure()
    sns.lineplot(df, x="Time", y="Martingale", hue="Method")
    plt.savefig("plots/" + out_filename)
