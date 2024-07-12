"""Core functions of Stein Points."""

from typing import Callable

import numpy as np


def kmat(
        sample: np.ndarray,
        gradient: np.ndarray,
        stein_kernel: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    ) -> np.ndarray:
    """Compute a Stein kernel matrix

    The matrix is obtained by evaluating the provided Stein kernel
    on a Cartesian square of `sample`.

    Parameters
    ----------
    sample: np.ndarray
        n x d array where each row is a d-dimensional sample point.
    gradient: np.ndarray
        n x d array where each row is a gradient of the log target.
    stein_kernel: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
        vectorised Stein kernel function.

    Returns
    -------
    np.ndarray
        n x n array containing the Stein kernel matrix.
    """
    n = sample.shape[0]
    k0 = np.zeros((n, n))
    ind1, ind2 = np.triu_indices(n)
    v = stein_kernel(sample[ind1], sample[ind2], gradient[ind1], gradient[ind2]).squeeze()
    k0[ind1, ind2] = v
    k0[ind2, ind1] = v
    return k0


def ksd(
        sample: np.ndarray,
        gradient: np.ndarray,
        stein_kernel: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    ) -> np.ndarray:
    """Compute a cumulative sequence of KSD values.

    KSD values are calculated from sums of elements in each i x i square in the top-left
    corner of the kernel Stein matrix.

    Parameters
    ----------
    sample: np.ndarray
        n x d array where each row is a d-dimensional sample point.
    gradient: np.ndarray
        n x d array where each row is a gradient of the log target.
    stein_kernel: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
        vectorised Stein kernel function.

    Returns
    -------
    np.ndarray
        array shaped (n,) containing the sequence of KSD values.
    """
    n = sample.shape[0]
    km = kmat(sample, gradient, stein_kernel)
    ind1, ind2 = np.tril_indices(n, -1)  # indices of elements below diagonal
    ks = np.cumsum(np.diag(km)) + 2 * np.concatenate([[0], np.cumsum(km[ind1, ind2])[np.cumsum(np.arange(1, n)) - 1]])
    return np.sqrt(ks) / np.arange(1, n + 1)
