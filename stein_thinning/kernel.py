"""Kernel functions."""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
from scipy.spatial.distance import pdist


def vfk0_imq(a, b, sa, sb, linv, c=1.0, beta=-0.5):
    amb = a.T - b.T
    qf = c + np.sum(np.dot(linv, amb) * amb, axis=0)
    t1 = -4 * beta * (beta - 1) * np.sum(np.dot(np.dot(linv, linv), amb) * amb, axis=0) / (qf ** (-beta + 2))
    t2 = -2 * beta * (np.trace(linv) + np.sum(np.dot(linv, sa.T - sb.T) * amb, axis=0)) / (qf ** (-beta + 1))
    t3 = np.sum(sa.T * sb.T, axis=0) / (qf ** (-beta))
    return t1 + t2 + t3


def _isfloat(value):
    """Test if value can be converted to float"""
    try:
        float(value)
        return True
    except ValueError:
        return False


def make_precon(sample: np.ndarray, preconditioner: str = 'id') -> np.ndarray:
    """Create preconditioner matrix

    Parameters
    ----------
    sample: np.ndarray
        n x d array where each row is a d-dimensional sample point.
    preconditioner: str
        optional string, either 'id' (default), 'med', 'sclmed', or
        'smpcov', specifying the preconditioner to be used. Alternatively,
        a numeric string can be passed as the single length-scale parameter
        of an isotropic kernel.

    Returns
    -------
    np.ndarray
        d x d array containing the preconditioner matrix
    """
    # Sample size and dimension
    N, d = sample.shape

    # Squared pairwise median
    def med2(m: int) -> float:
        if N > m:
            sub = sample[np.linspace(0, N - 1, m, dtype=int)]
        else:
            sub = sample
        return np.median(pdist(sub)) ** 2

    # Select preconditioner
    m = 1000
    if preconditioner == 'id':
        return np.identity(d)
    elif preconditioner == 'med':
        m2 = med2(m)
        assert m2 > 0, 'Too few unique samples.'
        return np.identity(d) / m2
    elif preconditioner == 'sclmed':
        m2 = med2(m)
        assert m2 > 0, 'Too few unique samples.'
        return np.identity(d) / m2 * np.log(np.minimum(m, N))
    elif preconditioner == 'smpcov':
        c = np.cov(sample, rowvar=False)
        assert np.all(eig(c)[0] > 0), 'Covariance matrix of sample is singular.'
        return inv(c)
    elif _isfloat(preconditioner):
        return np.identity(d) / float(preconditioner)
    else:
        raise ValueError('Incorrect preconditioner type.')


def make_imq(sample: np.ndarray, preconditioner: str = 'id'):
    preconditioner = make_precon(sample, preconditioner)
    def vfk0(sample1, sample2, gradient1, gradient2):
        return vfk0_imq(sample1, sample2, gradient1, gradient2, preconditioner)
    return vfk0
