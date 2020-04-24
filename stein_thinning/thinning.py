"""Core functions of Stein Thinning."""

import numpy as np
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
    fk0 = make_imq(smp, scr, pre)
    fmin = lambda vf, x, vfs: fmin_grid(vf, x, vfs, smp_uni)
    x, s, _ = greedy(dm, vfs, fk0, fmin, n)
    return x, s
