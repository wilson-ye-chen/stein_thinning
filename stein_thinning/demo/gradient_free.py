"""Stein thinning for a Gaussian mixture model using the gradient-free approach"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from stein_thinning.thinning import thin, thin_gf


class GaussianMixture:

    def __init__(self, weights, means, covs):
        self.n_components, self.d = means.shape
        assert weights.shape == (self.n_components,)
        assert covs.shape == (self.n_components, self.d, self.d)
        self.weights = weights
        self.means = means
        self.covs = covs
        self.covs_inv = np.linalg.inv(covs)

    def rvs(self, size, random_state):
        """Generate random variates"""
        component_samples = [
            mvn.rvs(
                mean=self.means[i], cov=self.covs[i], size=size, random_state=random_state
            ) for i in range(self.n_components)
        ]
        indices = rng.choice(self.n_components, size=size, p=self.weights)
        return np.take_along_axis(
            np.stack(component_samples, axis=1), indices.reshape(size, 1, 1), axis=1
        ).squeeze()

    def logpdf(self, x):
        """Calculate log-pdf"""
        f = np.stack([mvn.pdf(x, mean=means[i], cov=covs[i]) for i in range(self.n_components)]).reshape(self.n_components, -1)
        return np.log(np.einsum('i,il->l', self.weights, f))

    def score(self, x):
        """Calculate score (gradient of log-pdf)"""
        # centered sample
        xc = x[np.newaxis, :, :] - self.means[:, np.newaxis, :]
        # pdf evaluations for all components and all elements of the sample
        f = np.stack([mvn.pdf(x, mean=self.means[i], cov=self.covs[i]) for i in range(self.n_components)]).reshape(self.n_components, -1)
        # numerator of the score function
        num = np.einsum('i,il,ijk,ilk->lj', self.weights, f, self.covs_inv, xc)
        # denominator of the score function
        den = np.einsum('i,il->l', self.weights, f)
        return -num / den[:, np.newaxis]


w = np.array([0.3, 0.7])
means = np.array([
    [-1., -1.],
    [1., 1.],
])
covs = np.array([
    [
        [0.5, 0.25],
        [0.25, 1.],
    ],
    [
        [2.0, -np.sqrt(3.) * 0.8],
        [-np.sqrt(3.) * 0.8, 1.5],
    ]
])

mixture = GaussianMixture(w, means, covs)

rng = np.random.default_rng(12345)
sample_size = 1000
sample = mixture.rvs(sample_size, random_state=rng)

# Numerically calculate the gradient
gradient = mixture.score(sample)

# Apply Stein thinning
thinned_size = 40
idx = thin(sample, gradient, thinned_size)

# For the proxy distribution, use a simple Gaussian with sample mean and covariance
sample_mean = np.mean(sample, axis=0)
sample_cov = np.cov(sample, rowvar=False, ddof=1)

# Gradient-free Stein thinning requires us to provide the log-pdf of the proxy
# distribution and its score function:
log_q = mvn.logpdf(sample, mean=sample_mean, cov=sample_cov)
gradient_q = -np.einsum('ij,kj->ki', np.linalg.inv(sample_cov), sample - sample_mean)

# We also need the log-pdf of the target distribution
log_p = mixture.logpdf(sample)

# Apply gradient-free Stein thinning
idx_gf = thin_gf(sample, log_p, log_q, gradient_q, thinned_size)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(sample[:, 0], sample[:, 1], alpha=0.3, color='gray')
axs[0].scatter(sample[idx, 0], sample[idx, 1], color='red')
axs[0].set_title('Stein thinning')
axs[1].scatter(sample[:, 0], sample[:, 1], alpha=0.3, color='gray')
axs[1].scatter(sample[idx_gf, 0], sample[idx_gf, 1], color='red')
axs[1].set_title('Gradient-free Stein thinning')

plt.show()
