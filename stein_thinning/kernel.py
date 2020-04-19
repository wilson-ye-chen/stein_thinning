"""Kernel functions."""

import jax.numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import pdist
from stein_thinning.util import isfloat

def fk_imq(a, b, linv):
    amb = a - b
    return (1 + np.dot(np.dot(amb, linv), amb)) ** (-0.5)

def make_precon(smp, pre='sclmed'):
    # Sample size and dimension
    sz, dm = smp.shape

    # Squared pairwise median
    def med2(m):
        if sz > m:
            sub = smp[np.linspace(0, sz - 1, m, dtype=int)]
        else:
            sub = smp
        return np.median(pdist(sub)) ** 2

    # Select preconditioner
    m = 1000
    if pre == 'med':
        l = med2(m) * np.identity(dm)
    elif pre == 'sclmed':
        l = med2(m) / np.log(np.minimum(m, sz)) * np.identity(dm)
    elif pre == 'smpcov':
        l = np.cov(smp, rowvar=False)
    elif pre == 'bayesian':
        c = np.cov(smp, rowvar=False)
        l = 1 / (sz - dm - 1) * (np.identity(dm) + (sz - 1) * c)
    elif isfloat(pre):
        l = float(pre) * np.identity(dm)
    else:
        raise ValueError('incorrect preconditioner type.')
    return l

def make_imq(smp, pre='sclmed'):
    linv = inv(make_precon(smp, pre))
    def fk(a, b):
        return fk_imq(a, b, linv)
    return fk
