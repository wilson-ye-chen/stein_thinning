"""Core functions of Stein Thinning."""

import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import pdist
from stein_thinning.stein import greedy, fmin_grid
from stein_thinning.kernel import make_imq

def thin(smp, scr, n, pre='sclmed'):
    # Sample dimension
    dm = smp.shape[1]

    # Eliminate duplicates
    smp_uni, idx_uni = np.unique(smp, return_index=True, axis=0)
    scr_uni = scr[idx_uni]

    # Run SP using grid search
    vfs = lambda x: scr_uni
    fk = make_imq(smp, pre)
    fmin = lambda vf, x, vfs: fmin_grid(vf, x, vfs, smp_uni)
    return greedy(dm, vfs, fk, fmin, n)
