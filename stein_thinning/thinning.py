"""Core functions of Stein Thinning."""

import numpy as np
from stein_thinning.kernel import make_imq

def thin(smp, scr, m, pre='sclmed'):
    # Sample dimension
    n = smp.shape[0]

    # Vectorised Stein kernel
    vfk0 = make_imq(smp, scr, pre)

    k0 = np.empty((n, m))
    return x, s
