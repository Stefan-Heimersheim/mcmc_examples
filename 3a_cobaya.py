# https://cobaya.readthedocs.io/en/latest/example.html
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst


def loglikelihood_function(a=0, b=0):
#def likelihood_function(**kwargs):
    return sst.norm.logpdf((a**2+b**2), loc=1, scale=0.5)

# Dictionary to tell cobaya what to do
info = {
    "likelihood": {
        "name_of_your_likelihood_here_does_not_matter": {
            "external": loglikelihood_function
            # Optionally you can set input parameters to save writing all the kwargs in the function:
            # input_params = ['a', 'b']
        },
    },
    "params": {
        "a": {
            "prior": {"min": -5, "max": 5},
            "proposal": 0.1, # automatically guessed from prior, reduce if getting stuck
            "latex": r"\alpha",
        },
        "b": {
            "prior": {"min": -5, "max": 5},
            "proposal": 0.1, # automatically guessed from prior, reduce if getting stuck
            "latex": r"\beta",
        },
    },
    "sampler": {
        "mcmc": {
            # Useful settings to just manually run for a certain length:
            #"max_samples": 100000,
            #"Rminus1_stop": 0,
            #"Rminus1_cl_stop": 0,
        },
#        "polychord": {
#            # This is basically the accuracy argument for polychord
#            "nlive": 300,
#        },

    },
}

# Run cobaya with this information
from cobaya.run import run
full_info, sampler = run(info)

# "sampler" contains the results, "full_info" includes the input dictionary + all default settings used

# Plotting:
samples = sampler.products()['sample']
a_samples = samples['a']
b_samples = samples['b']
weights = samples['weight']
posterior_values = np.exp(-samples['minuslogpost'])

plt.hist2d(a_samples, b_samples, weights=weights, bins=100)
plt.show()



# anesthetic

import anesthetic
data = np.array([a_samples, b_samples]).T
anesthetic_samples = anesthetic.samples.MCMCSamples(data=data, weights=weights, columns=['a', 'b'], tex={'a': r'$\alpha$', 'b': r'$\beta$'})
anesthetic_samples.plot_2d(['a', 'b'])
plt.show()




# Another plotting tool

from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt

gd_sample = MCSamplesFromCobaya(full_info, sampler.products()["sample"])
gdplot = gdplt.get_subplot_plotter()
gdplot.triangle_plot(gd_sample, ["a", "b"], filled=True)
plt.show()

