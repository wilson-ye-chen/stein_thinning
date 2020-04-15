"""Core functions of Stein Thinning."""

import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import pdist
from stein_thinning.stein import greedy, fmin_grid, fk_imq

def thin(smp, scr, n, pre='sclmed'):
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
        l = 1 / (sz - dm - 1) * (np.identity(dm) + (sz - 1) * c)
        linv = inv(l)
    else:
        raise ValueError('incorrect preconditioner type.')

    # Eliminate duplicates
    smp_uni, idx_uni = np.unique(smp, return_index=True, axis=0)
    scr_uni = scr[idx_uni]

    # Run SP using grid search
    vfs = lambda x: scr_uni
    fk = lambda a, b: fk_imq(a, b, linv)
    fmin = lambda vf, x, vfs: fmin_grid(vf, x, vfs, smp_uni)
    return greedy(dm, vfs, fk, fmin, n)
