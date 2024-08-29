"""Kernel matrix functions"""

from typing import Any, Callable

import numpy as np


IndexerT = Any


def kmat(integrand: Callable[[IndexerT, IndexerT], np.ndarray], n: int) -> np.ndarray:
    """Compute a Stein kernel matrix

    The matrix is obtained by evaluating the provided Stein kernel
    on a Cartesian square of `sample`.

    Parameters
    ----------
    integrand: Callable[[IndexerT, IndexerT], np.ndarray]
        vectorised function returning the values of the integrand in the KSD
        integral for the given indices (rows and columns).
    n: int
        size of the matrix to return

    Returns
    -------
    np.ndarray
        n x n array containing the Stein kernel matrix.
    """
    k0 = np.zeros((n, n))
    ind1, ind2 = np.triu_indices(n)
    v = integrand(ind1, ind2).squeeze()
    k0[ind1, ind2] = v
    k0[ind2, ind1] = v
    return k0


def ksd(integrand: Callable[[IndexerT, IndexerT], np.ndarray], n: int) -> np.ndarray:
    """Compute a cumulative sequence of KSD values.

    KSD values are calculated from sums of elements in each i x i square in the top-left
    corner of the kernel Stein matrix.

    Parameters
    ----------
    integrand: Callable[[IndexerT, IndexerT], np.ndarray]
        vectorised function returning the values of the integrand in the KSD
        integral for the given indices (rows and columns).
    n: int
        number of terms to calculate

    Returns
    -------
    np.ndarray
        array shaped (n,) containing the sequence of KSD values.
    """
    assert n > 0
    cum_sum = np.zeros(n)
    cum_sum[0] = integrand(0, 0)
    for i in range(1, n):
        vals = integrand([i], slice(0, i + 1))
        cum_sum[i] = cum_sum[i - 1] + 2 * np.sum(vals) - vals[-1]
    return np.sqrt(cum_sum) / np.arange(1, n + 1)
