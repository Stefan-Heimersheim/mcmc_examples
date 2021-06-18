#https://emcee.readthedocs.io/en/stable/tutorials/quickstart/
import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt

def loglikelihood_function(a=0, b=0):
    return sst.norm.logpdf((a**2+b**2), loc=1, scale=0.5)

def logprior_function(a=0, b=0):
    return sst.uniform.logpdf(a, loc=-5, scale=10)+sst.uniform.logpdf(b, loc=-5, scale=10)

# emcee wants:
##  input = array of parameters
##  output = logarithm of posterior
def logposterior_function(p):
    return loglikelihood_function(p[0], p[1]) + logprior_function(p[0], p[1])

import emcee

# Dimension of parameter space
nDim = 2

# Simultaneous "walkers" (separate chains)
nWalkers = 10

# Starting points for each walker in parameter space (usually prior)
p0 = np.random.uniform(low=-5,high=5,size=(nWalkers,nDim))

# "EnsembleSampler" class to do runs with
emcee_sampler = emcee.EnsembleSampler(nWalkers, nDim, logposterior_function)




# Check if chain is converged using autocorrelation length.
# Autocorrelation code is taken from emcee (MIT license) documentation, see below for details.
# From here: https://emcee.readthedocs.io/en/stable/user/autocorr/

# Run chain for 10000 points and track autocorrelation "tau"
max_n = 10000
index = 0
autocorr = np.empty(max_n)
old_tau = np.inf
was_converged = False
for sample in emcee_sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100/1000 steps
    if not was_converged and emcee_sampler.iteration % 100:
        continue
    if was_converged and emcee_sampler.iteration % 1000:
        continue
    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = emcee_sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
    # Check convergence
    converged = np.all(tau * 100 < emcee_sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged and not was_converged:
        was_converged = True
        print("Converged at iteration", emcee_sampler.iteration)
    old_tau = tau


# Plot autocorrelation time, estimate converges to true autocorrelation time.
# Aim to get N > 100*autocorrelation time

n = 100 * np.arange(1, index + 1)
y = autocorr[:index]
plt.plot(n, n / 100.0, "--k", label=r'$N = 100 \hat{\tau}$')
plt.plot(n, y, label='Chain')
plt.xlabel("number of steps $N$")
plt.ylabel(r"mean autocorrelation time $\hat{\tau}$");
plt.legend()
plt.show()



"""
The MIT License (MIT)

Copyright (c) 2010-2019 Daniel Foreman-Mackey & contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""