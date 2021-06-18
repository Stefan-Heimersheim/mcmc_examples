# https://cobaya.readthedocs.io/en/latest/example.html
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst


def loglikelihood_function(a=0, b=0):
    return sst.norm.logpdf((a**2+b**2), loc=1, scale=0.5)

# Dictionary to tell cobaya what to do
info = {
    # Where to store outputs?
    "output": "chains_3/cobaya_test",
    # Resume from existing outputs?
    "resume": False,
    # Force overwrite existing outputs
    "force": True,
    # The usual stuff
    "likelihood": {
        "name_of_your_likelihood_here_does_not_matter": {
            "external": loglikelihood_function
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
        "polychord": {
            # This is basically the accuracy argument for polychord
            "nlive": 100,
        },

    },
}

# Run cobaya with this information
from cobaya.run import run
full_info, sampler = run(info)


# Load samples from file
import anesthetic

# Load nested samples from polychord
anesthetic.samples.NestedSamples(root="chains_3/cobaya_test_polychord_raw/cobaya_test")

# Loading MCMC samples isn't that great yet
anesthetic.samples.MCMCSamples(root="chains_3/cobaya_test")

