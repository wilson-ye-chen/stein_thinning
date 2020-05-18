import numpy as np
from stein_thinning.kernel import make_imq

def thin(smp, scr, m, pre='sclmed'):
    """
    Optimally select m points from n > m samples generated from a target
    distribution of d dimensions.

    Args:
    smp - n x d array where each row is a sample point.
    scr - n x d array where each row is a gradient of the log target.
    m   - integer specifying the desired number of points.
    pre - string specifying the heuristic for computing the preconditioning
          matrix, either 'med', 'sclmed', 'smpcov', 'bayesian' or 'avehess'.

    Returns:
    array shaped (m,) containing the row indices in smp (and scr) of the
    selected points.
    """

    # Vectorised Stein kernel function
    vfk0 = make_imq(smp, scr, pre)

    # Pre-allocate arrays
    n = smp.shape[0]
    k0 = np.empty((n, m))
    iSel = np.empty(m, dtype=np.uint32)

    # Populate columns of k0 as new points are selected
    k0[:, 0] = vfk0(smp, smp, scr, scr)
    iSel[0] = np.argmin(k0[:, 0])
    print(f'{1} of {m}')
    for i in range(1, m):
        smpLast = np.tile(smp[iSel[i - 1]], (n, 1))
        scrLast = np.tile(scr[iSel[i - 1]], (n, 1))
        k0[:, i] = vfk0(smp, smpLast, scr, scrLast)
        iSel[i] = np.argmin(k0[:, 0] + 2 * np.sum(k0[:, 1:(i + 1)], axis=1))
        print(f'{i + 1} of {m}')
    return iSel
