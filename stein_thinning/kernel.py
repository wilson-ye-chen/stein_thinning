"""Kernel functions."""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
from scipy.spatial.distance import pdist
from stein_thinning.util import isfloat


def vfk0_imq(a, b, sa, sb, linv):
    amb = a.T - b.T
    qf = 1 + np.sum(np.dot(linv, amb) * amb, axis=0)
    t1 = -3 * np.sum(np.dot(np.dot(linv, linv), amb) * amb, axis=0) / (qf ** 2.5)
    t2 = (np.trace(linv) + np.sum(np.dot(linv, sa.T - sb.T) * amb, axis=0)) / (qf ** 1.5)
    t3 = np.sum(sa.T * sb.T, axis=0) / (qf ** 0.5)
    return t1 + t2 + t3


def make_precon(sample, pre='id'):
    # Sample size and dimension
    sz, dm = sample.shape

    # Squared pairwise median
    def med2(m):
        if sz > m:
            sub = sample[np.linspace(0, sz - 1, m, dtype=int)]
        else:
            sub = sample
        return np.median(pdist(sub)) ** 2

    # Select preconditioner
    m = 1000
    if pre == 'id':
        linv = np.identity(dm)
    elif pre == 'med':
        m2 = med2(m)
        if m2 == 0:
            raise Exception('Too few unique samples in smp.')
        linv = inv(m2 * np.identity(dm))
    elif pre == 'sclmed':
        m2 = med2(m)
        if m2 == 0:
            raise Exception('Too few unique samples in smp.')
        linv = inv(m2 / np.log(np.minimum(m, sz)) * np.identity(dm))
    elif pre == 'smpcov':
        c = np.cov(sample, rowvar=False)
        if not all(eig(c)[0] > 0):
            raise Exception('Too few unique samples in smp.')
        linv = inv(c)
    elif isfloat(pre):
        linv = inv(float(pre) * np.identity(dm))
    else:
        raise ValueError('Incorrect preconditioner type.')
    return linv


def make_imq(smp, pre='id'):
    linv = make_precon(smp, pre)
    def vfk0(a, b, sa, sb):
        return vfk0_imq(a, b, sa, sb, linv)
    return vfk0
