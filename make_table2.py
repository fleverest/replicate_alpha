import polars as pl

from params import thetas, reps


csvs = [
    pl.read_csv(f"table2_results/theta{theta}_{i}.csv")\
        .with_columns(theta=pl.lit(theta))
    for i in range(reps)
    for theta in thetas
]

table2 = pl.concat(csvs)\
    .group_by("theta", "eta", "d")\
    .mean()\
    .sort("theta", "eta", "d")
