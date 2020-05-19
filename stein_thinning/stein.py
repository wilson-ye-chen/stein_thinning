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

def ksd(x, s, vfk0):
    n, d = x.shape
    ks = np.empty(n)
    ps = 0.
    for i in range(n):
        x_new = x[i].reshape((-1, d))
        s_new = s[i].reshape((-1, d))
        ps += vfps(x_new, s_new, x, s, i, vfk0)
        ks[i] = np.sqrt(ps) / (i + 1)
        print(f'i = {i}')
    return ks

def kmat(x, s, vfk0):
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
