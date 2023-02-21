"""Core functions of Stein Points."""

import numpy as np
from stein_thinning.util import mirror_lower

def fmin_grid(vf, x, vfs, grid):
    s = vfs(grid)
    val = vf(grid, s)
    i = np.argmin(val)
    return grid[i], s[i], grid.shape[0]

def vfps(x_new, s_new, x, s, i, vfk0):
    k0aa = vfk0(x_new, x_new, s_new, s_new)
    if i > 0:
        n_new = x_new.shape[0]
        a = np.tile(x_new, (i, 1))
        b = np.repeat(x[0:i], n_new, 0)
        sa = np.tile(s_new, (i, 1))
        sb = np.repeat(s[0:i], n_new, 0)
        k0ab = np.reshape(vfk0(a, b, sa, sb), (-1, n_new))
        return np.sum(k0ab, axis=0) * 2 + k0aa
    else:
        return k0aa

def ksd(x, s, vfk0, verb=False):
    """
    Compute a cumulative sequence of KSD values.

    Args:
    x    - n x d array where each row is a d-dimensional sample point.
    s    - n x d array where each row is a gradient of the log target.
    vfk0 - vectorised Stein kernel function.
    verb - optional logical, either 'True' or 'False' (default), indicating
           whether or not to be verbose about the KSD evaluation progress.

    Returns:
    array shaped (n,) containing the sequence of KSD values.
    """

    n = x.shape[0]
    ks = np.empty(n)
    ps = 0.
    for i in range(n):
        x_i = np.tile(x[i], (i + 1, 1))
        s_i = np.tile(s[i], (i + 1, 1))
        k0 = vfk0(x_i, x[0:(i + 1)], s_i, s[0:(i + 1)])
        ps += 2 * np.sum(k0[0:i]) + k0[i]
        ks[i] = np.sqrt(ps) / (i + 1)
        if verb:
            print(f'KSD: {i + 1} of {n}')
    return ks

def kmat(x, s, vfk0):
    """
    Compute a Stein kernel matrix.

    Args:
    x    - n x d array where each row is a d-dimensional sample point.
    s    - n x d array where each row is a gradient of the log target.
    vfk0 - vectorised Stein kernel function.

    Returns:
    n x n array containing the Stein kernel matrix.
    """

    n = x.shape[0]
    k0 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            k0[i, j] = vfk0(x[i], x[j], s[i], s[j])
    mirror_lower(k0)
    return k0

def greedy(d, vfs, vfk0, fmin, n):
    x = np.empty((n, d))
    s = np.empty((n, d))
    e = np.empty(n)
    for i in range(n):
        vf = lambda x_new, s_new: vfps(x_new, s_new, x, s, i, vfk0)
        x[i], s[i], e[i] = fmin(vf, x, vfs)
        print(f'i = {i}')
    return x, s, e
