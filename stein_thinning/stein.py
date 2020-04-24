"""Core functions of Stein Points."""

import numpy as np
from stein_thinning.util import mirror_lower

def fmin_grid(vf, x, vfs, grid):
    s = vfs(grid)
    val = vf(grid, s)
    i = np.argmin(val)
    return grid[i], s[i], grid.shape[0]

def vfps(x_new, s_new, x, s, i, fk0):
    n_new = x_new.shape[0]
    ps = np.empty(n_new)
    for j in range(n_new):
        ab_sum = 0.
        for k in range(i):
            ab_sum += fk0(x_new[j], x[k], s_new[j], s[k])
        aa = fk0(x_new[j], x_new[j], s_new[j], s_new[j])
        ps[j] = ab_sum * 2 + aa
    return ps

def ksd(x, s, fk0):
    n, d = x.shape
    ks = np.empty(n)
    ps = 0.
    for i in range(n):
        x_new = x[i].reshape((-1, d))
        s_new = s[i].reshape((-1, d))
        ps += vfps(x_new, s_new, x, s, i, fk0)
        ks[i] = np.sqrt(ps) / (i + 1)
        print(f'i = {i}')
    return ks

def kmat(x, s, fk0):
    n = x.shape[0]
    k0 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            k0[i, j] = fk0(x[i], x[j], s[i], s[j])
    mirror_lower(k0)
    return k0

def greedy(d, vfs, fk0, fmin, n):
    x = np.empty((n, d))
    s = np.empty((n, d))
    e = np.empty(n)
    for i in range(n):
        vf = lambda x_new, s_new: vfps(x_new, s_new, x, s, i, fk0)
        x[i], s[i], e[i] = fmin(vf, x, vfs)
        print(f'i = {i}')
    return x, s, e
