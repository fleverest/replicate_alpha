alpha = 0.05

reps = int(10**3)
block_size = int(10**7)  # number of Bernoulli RVs to generate as a batch
max_size = int(10**7)

thetas = [0.505, 0.51, 0.52, 0.53, 0.54, 0.55, 0.6, 0.65, 0.7]
etas = thetas
ds = [10, 100, 500, 1000]
