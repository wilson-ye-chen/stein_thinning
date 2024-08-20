"""Post-process output from Stan."""

import numpy as np
import matplotlib.pyplot as plt

import stan
from stein_thinning.thinning import thin

if __name__ == '__main__':
    # Simple bivariate Gaussian model
    mc = """
    parameters {vector[2] x;}
    model {x ~ multi_normal([0, 0], [[1, 0.8], [0.8, 1]]);}
    """
    sm = stan.build(mc, random_seed=12345)
    fit = sm.sample(num_samples=1000)

    # Extract sampled points and gradients
    sample = fit['x'].T
    gradient = np.apply_along_axis(lambda x: sm.grad_log_prob(x.tolist()), 1, sample)

    # Obtain a subset of 40 points
    idx = thin(sample, gradient, 40)

    # Plot point-set over trace
    plt.figure()
    plt.scatter(sample[:, 0], sample[:, 1], color='lightgray')
    plt.scatter(sample[idx, 0], sample[idx, 1], color='red')

    plt.savefig('stein_thinning/demo/pystan.png')
    plt.show()
