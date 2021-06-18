import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt

def likelihood(p):
    return sst.lognorm.pdf((np.sum(np.array(p)**2, axis=0)), s=1)

def prior(p):
    return np.product(sst.uniform.pdf(np.array(p), loc=-5, scale=10), axis=0)

posterior = lambda p: likelihood(p)*prior(p)
# (unnormalized)

# Example:
print("Posterior(1,1) =", posterior([1,1]))

# ----------------------

# Grid plot:

nDims = 2

# meshgrid is fun!
grid_points = [a.flatten() for a in np.meshgrid(*([np.linspace(-6,6,100)]*nDims))]

values = likelihood(grid_points)*prior(grid_points)

plt.hist2d(grid_points[0], grid_points[1], weights=values, bins=100)
plt.show()



# ----------------------

# MCMC:
nDims = 2

# Starting points
mcmc_points = [[-5,-5]]
posterior_values = [posterior(mcmc_points[0])]



for i in range(10000):
    # generate new samples
    # accept/reject them
    # append new/old sample to mcmc_points and (optionally) posterior_values

P = np.array(mcmc_points)
plt.hist2d(P.T[0], P.T[1], bins=100)
plt.show()

plt.plot(posterior_values)
plt.show()
