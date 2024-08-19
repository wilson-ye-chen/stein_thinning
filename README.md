# Stein Thinning
This Python package implements an algorithm for optimally compressing
sampling algorithm outputs by minimising a kernel Stein discrepancy.
Please see the accompanying paper "Optimal Thinning of MCMC Output"
([arXiv](https://arxiv.org/pdf/2005.03952.pdf)) for details of the
algorithm.

# Installing the package

The latest stable version can be installed via pip:
```
pip install stein-thinning
```

To install the current development version, use this command:
```
pip install git+https://github.com/wilson-ye-chen/stein_thinning
```

# Getting Started
For example, correlated samples from a posterior distribution are
obtained using a MCMC algorithm and stored in the NumPy array `smpl`,
and the corresponding gradients of the log-posterior are stored in
another NumPy array `grad`. One can then perform Stein Thinning to
obtain a subset of 40 sample points by running the following code:
```python
from stein_thinning.thinning import thin
idx = thin(smpl, grad, 40)
```
The `thin` function returns a NumPy array containing the row indices
in `smpl` (and `grad`) of the selected points. Please refer to `demo.py`
as a starting example.

The default usage requires no additional user input and is based on
the identity (`id`) preconditioning matrix and standardised sample.
Alternatively, the user can choose to specify which heuristic to use
for computing the preconditioning matrix by setting the option string
to either `id`, `med`,  `sclmed`, or `smpcov`. Standardisation can be
disabled by setting `stnd=False`. For example, the default setting
corresponds to:
```python
idx = thin(smpl, grad, 40, stnd=True, pre='id')
```
The details for each of the heuristics are documented in Section 2.3 of
the accompanying paper.

# PyStan Example
As an illustration of how Stein Thinning can be used to post-process
output from [Stan](https://mc-stan.org/users/interfaces/pystan), consider
the following simple Stan script that produces correlated samples from a
bivariate Gaussian model:
```python
from pystan import StanModel
mc = """
parameters {vector[2] x;}
model {x ~ multi_normal([0, 0], [[1, 0.8], [0.8, 1]]);}
"""
sm = stan.build(mc, random_seed=12345)
fit = sm.sample(num_samples=1000)
```
The bivariate Gaussian model is used for illustration, but regardless of
the complexity of the model being sampled the output of Stan will always
be a `fit` object (StanFit instance). The sampled points and the
log-posterior gradients can be extracted from the returned `fit` object:
```python
import numpy as np
sample = fit['x'].T
gradient = np.apply_along_axis(lambda x: sm.grad_log_prob(x.tolist()), 1, sample)
idx = thin(sample, gradient, 40)
```
The selected points can then be plotted:
```python
plt.figure()
plt.scatter(sample[:, 0], sample[:, 1], color='lightgray')
plt.scatter(sample[idx, 0], sample[idx, 1], color='red')
plt.show()
```

The above example can be found in `pystan/demo.py`.
