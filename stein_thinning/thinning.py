import numpy as np
from stein_thinning.kernel import make_imq

def thin(smp, scr, m, stnd=True, pre='id'):
    """
    Optimally select m points from n > m samples generated from a target
    distribution of d dimensions.

    Args:
    smp  - n x d array where each row is a sample point.
    scr  - n x d array where each row is a gradient of the log target.
    m    - integer specifying the desired number of points.
    stnd - optional logical, either 'True' (default) or 'False', indicating
           whether or not to standardise the colums of smp.
    pre  - optional string, either 'id' (default), 'med', 'sclmed', or
           'smpcov', specifying the preconditioner to be used. Alternatively,
           a numeric string can be passed as the single length-scale parameter
           of an isotropic kernel.

    Returns:
    array shaped (m,) containing the row indices in smp (and scr) of the
    selected points.
    """

    # Vectorised Stein kernel function
    vfk0 = make_imq(smp, scr, pre)

    # Pre-allocate arrays
    n = smp.shape[0]
    k0 = np.empty((n, m))
    idx = np.empty(m, dtype=np.uint32)

    # Populate columns of k0 as new points are selected
    k0[:, 0] = vfk0(smp, smp, scr, scr)
    idx[0] = np.argmin(k0[:, 0])
    print(f'THIN: {1} of {m}')
    for i in range(1, m):
        smp_last = np.tile(smp[idx[i - 1]], (n, 1))
        scr_last = np.tile(scr[idx[i - 1]], (n, 1))
        k0[:, i] = vfk0(smp, smp_last, scr, scr_last)
        idx[i] = np.argmin(k0[:, 0] + 2 * np.sum(k0[:, 1:(i + 1)], axis=1))
        print(f'THIN: {i + 1} of {m}')
    return idx
