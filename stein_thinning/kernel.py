"""Kernel functions."""

import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import pdist
from stein_thinning.util import isfloat

def vfk0_imq(a, b, sa, sb, linv):
    amb = a.T - b.T
    qf = 1 + np.sum(np.dot(linv, amb) * amb, axis=0)
    t1 = -3 * np.sum(np.dot(np.dot(linv, linv), amb) * amb, axis=0) / (qf ** 2.5)
    t2 = (np.trace(linv) + np.sum(np.dot(linv, sa.T - sb.T) * amb, axis=0)) / (qf ** 1.5)
    t3 = np.sum(sa.T * sb.T, axis=0) / (qf ** 0.5)
    return t1 + t2 + t3

def make_precon(smp, scr, pre='sclmed'):
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
        linv = inv(med2(m) * np.identity(dm))
    elif pre == 'sclmed':
        linv = inv(med2(m) / np.log(np.minimum(m, sz)) * np.identity(dm))
    elif pre == 'smpcov':
        linv = inv(np.cov(smp, rowvar=False))
    elif pre == 'bayesian':
        c = np.cov(smp, rowvar=False)
        linv = inv(1 / (sz - dm - 1) * (np.identity(dm) + (sz - 1) * c))
    elif isfloat(pre):
        linv = inv(float(pre) * np.identity(dm))
    else:
        raise ValueError('incorrect preconditioner type.')
    return linv

def make_imq(smp, scr, pre='sclmed'):
    linv = make_precon(smp, scr, pre)
    def vfk0(a, b, sa, sb):
        return vfk0_imq(a, b, sa, sb, linv)
    return vfk0
