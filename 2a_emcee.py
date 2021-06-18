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


# Simple
emcee_sampler.run_mcmc(p0, 500, progress=True)

# Look at convergence / burn-in
chain = emcee_sampler.get_chain()
logprob = emcee_sampler.get_log_prob()

for i in range(10):
    plt.plot(logprob[:,i])

plt.show()


# Run a bit to cut off burn-in
emcee_sampler = emcee.EnsembleSampler(nWalkers, nDim, logposterior_function)

# Save that last (random) position and throw away chain
intermediate_point = emcee_sampler.run_mcmc(p0, 500, progress=True)
emcee_sampler.reset()

# Rerun starting from the last position
emcee_sampler.run_mcmc(intermediate_point, 5000, progress=True)

# Look at convergence / burn-in
chain = emcee_sampler.get_chain()
logprob = emcee_sampler.get_log_prob()

for i in range(10):
    plt.plot(logprob[:,i])

plt.show()


# Simple plotting:
samples = emcee_sampler.flatchain.T
plt.hist2d(samples[0], samples[1], bins=100)
plt.show()





# With backend to store:

nDim = 2
nWalkers = 10    
p0 = np.random.uniform(low=-5,high=5,size=(nWalkers,nDim))

filename = "chains_2/emcee.h5"
rerun = True

if rerun:
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nWalkers, nDim)
    
    emcee_sampler = emcee.EnsembleSampler(nWalkers, nDim, logposterior_function, backend=backend)
    state = emcee_sampler.run_mcmc(p0, 1000, progress=True)
    emcee_sampler.reset()
    emcee_sampler.run_mcmc(state, 5000, progress=True)

    chain = emcee_sampler.get_chain()
    flatchain = emcee_sampler.flatchain
    log_prob = emcee_sampler.get_log_prob()
else:
    backend = emcee.backends.HDFBackend(filename)

    log_prob = backend.get_log_prob()
    chain = backend.get_chain()
    s=np.shape(chain) #number of chains, length of chains, dimensions
    flatchain = chain.reshape([s[0]*s[1],s[2]])

#emcee_to_anesthetic = lambda s: anesthetic.samples.MCMCSamples(data=s.flatchain, columns=paramNames, tex=texDict)
#emcee_results = vemcee_to_anesthetic(emcee_sampler)

import anesthetic
anesthetic_samples = anesthetic.samples.MCMCSamples(data=flatchain, columns=['a', 'b'], tex={'a': r'$\alpha$', 'b': r'$\beta$'})

anesthetic_samples.plot_2d(['a', 'b'])
plt.show()



check_convergence = True
if check_convergence:
    nDim = 2
    nWalkers = 10    
    p0 = np.random.uniform(low=-5,high=5,size=(nWalkers,nDim))

    emcee_sampler = emcee.EnsembleSampler(nWalkers, nDim, logposterior_function)
    max_n = 10000
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    # This will be useful to testing convergence
    old_tau = np.inf
    was_converged = False
    # Now we'll sample for up to max_n steps
    for sample in emcee_sampler.sample(p0, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
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

if check_convergence:
    n = 100 * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.plot(n, n / 100.0, "--k", label=r'$N = 100 \hat{\tau}$')
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    plt.xlabel("number of steps $N$")
    plt.legend()
    plt.ylabel(r"mean autocorrelation time $\hat{\tau}$");
    plt.show()

chain = emcee_sampler.get_chain()
flatchain = emcee_sampler.flatchain
log_prob = emcee_sampler.get_log_prob()
anesthetic_samples = anesthetic.samples.MCMCSamples(data=flatchain, columns=['a', 'b'], tex={'a': r'$\alpha$', 'b': r'$\beta$'})

anesthetic_samples.plot_2d(['a', 'b'])
plt.show()