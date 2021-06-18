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




# Loading and saving

filename = "chains_2/emcee.h5"
load_from_file = False

if not load_from_file:
    # Load savefile & delete old data in there
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nWalkers, nDim)
    
    # Do the usual thing, run a bit for burn in and the actual chain
    # Will automatically be saved!
    emcee_sampler = emcee.EnsembleSampler(nWalkers, nDim, logposterior_function, backend=backend)
    state = emcee_sampler.run_mcmc(p0, 1000, progress=True)
    emcee_sampler.reset()
    emcee_sampler.run_mcmc(state, 5000, progress=True)

    # Usual stuff for plotting
    chain = emcee_sampler.get_chain()
    flatchain = emcee_sampler.flatchain
    log_prob = emcee_sampler.get_log_prob()
else:
    # Load results from file
    backend = emcee.backends.HDFBackend(filename)

    # Get most things similarly
    chain = backend.get_chain()
    log_prob = backend.get_log_prob()
    # .flatchain doesn't work but you can make it:
    s = np.shape(chain) #number of chains, length of chains, dimensions
    flatchain = chain.reshape([s[0]*s[1],s[2]])


# Plotting:

samples = flatchain.T
plt.hist2d(samples[0], samples[1], bins=100)
plt.show()


# Will Handley's anesthetic

import anesthetic
anesthetic_samples = anesthetic.samples.MCMCSamples(data=flatchain, columns=['a', 'b'], tex={'a': r'$\alpha$', 'b': r'$\beta$'})
anesthetic_samples.plot_2d(['a', 'b'])
plt.show()