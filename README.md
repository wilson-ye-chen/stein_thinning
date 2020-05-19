# Stein Thinning
This Python package implements an algorithm for optimally compressing
sampling algorithm outputs by minimising a kernel Stein discrepancy.
Please see the accompanying paper "Optimal Thinning of MCMC Output"
([arXiv](https://arxiv.org/pdf/2005.03952.pdf)) for details of the
algorithm.

# Installing via Git
One can pip install the package directly from this repository:
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
the `sclmed` heuristic. Alternatively, the user can choose to specify
which heuristic to use for computing the preconditioning matrix by
setting the option string `pre` to either `med`,  `sclmed`, `smpcov`,
`bayesian`, or `avehess`. For example, the default setting corresponds
to:
```python
x, g = thin(smpl, grad, 40, pre='sclmed')
```
The details for each of the heuristics are documented in Section 3.4 of
the accompanying paper.

# PyStan Example
As an illustration of how Stein Thinning can be used to post-process
output from Stan, consider the following simple Stan script that produces
correlated samples from a bivariate Gaussian model:
```python
from pystan import StanModel
mc = """
parameters {vector[2] x;}
model {x ~ multi_normal([0, 0], [[1, 0.8], [0.8, 1]]);}
"""
sm = StanModel(model_code=mc)
fit = sm.sampling(iter=1000)
```
The bivariate Gaussian model is used for illustration, but regardless of
the complexity of the model being sampled the output of Stan will always
be a `fit` object (StanFit instance). The sampled points and the
log-posterior gradients can be extracted from the returned `fit` object:
```python
import numpy as np
smpl = fit['x']
grad = np.apply_along_axis(fit.grad_log_prob, 1, smpl)
```
The above example can be found in `pystan/demo.py`.
