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
mcmc_points = [np.array([-5]*nDims)]
posterior_values = [posterior(mcmc_points[0])]

scale = 1
accepted_count = 0
rejected_count = 0


imax = 10000
for i in range(imax):
    if i%10000 == 0:
        print(i, imax)
    old_point = mcmc_points[-1]
    old_posterior = posterior_values[-1]
    new_point = old_point + np.random.normal(scale=[scale]*nDims)
    new_posterior = posterior(new_point)
    a = np.random.uniform()
    if a < new_posterior/old_posterior:
        accepted_count += 1
        #print("Accepted :)")
        mcmc_points.append(new_point)
        posterior_values.append(new_posterior)
    else:
        #print("Rejected :(")
        rejected_count += 1
        mcmc_points.append(old_point)
        posterior_values.append(old_posterior)


P = np.array(mcmc_points)
plt.hist2d(P.T[0], P.T[1], bins=100)
plt.show()

plt.plot(posterior_values)
plt.show()
