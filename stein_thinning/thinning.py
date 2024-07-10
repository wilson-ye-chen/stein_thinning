import numpy as np
from stein_thinning.kernel import make_imq


def thin(
        sample: np.ndarray,
        gradient: np.ndarray,
        n_points: int,
        standardize: bool = True,
        preconditioner: str = 'id',
        verbose: bool = False,
) -> np.ndarray:
    """Optimally select m points from n > m samples generated from a target distribution of d dimensions.

    Parameters
    ----------
    sample: np.ndarray
        n x d array where each row is a sample point.
    gradient: np.ndarray
        n x d array where each row is a gradient of the log target.
    n_points: int
        integer specifying the desired number of points.
    standardize: bool
        optional logical, either 'True' (default) or 'False', indicating 
        whether or not to standardise the columns of `sample` around means
        using the mean absolute deviation from the mean as the scale.
    preconditioner: str
        optional string, either 'id' (default), 'med', 'sclmed', or
        'smpcov', specifying the preconditioner to be used. Alternatively,
        a numeric string can be passed as the single length-scale parameter
        of an isotropic kernel.
    verbose: bool
        optional logical, either 'True' or 'False' (default), indicating
        whether or not to be verbose about the thinning progress.

    Returns
    -------
    np.ndarray
        array shaped (m,) containing the row indices in `sample` (and `gradient`) of the
        selected points.
    """
    # Argument checks
    assert sample.ndim == 2 and gradient.ndim == 2, 'sample or gradient is not two-dimensional.'
    n, d = sample.shape
    assert n > 0 and d > 0, 'sample is empty.'
    assert gradient.shape[0] == n and gradient.shape[1] == d, 'Dimensions of sample and gradient are inconsistent.'
    assert not np.any(np.isnan(sample)) and not np.any(np.isnan(gradient)), 'sample or gradient contains NaNs.'
    assert not np.any(np.isinf(sample)) and not np.any(np.isinf(gradient)), 'sample or gradient contains infs.'

    # Standardisation
    if standardize:
        loc = np.mean(sample, axis=0)
        scl = np.mean(np.abs(sample - loc), axis=0)
        assert np.min(scl) > 0, 'Too few unique samples in smp.'
        sample = sample / scl
        gradient = gradient * scl

    # Vectorised Stein kernel function
    vfk0 = make_imq(sample, preconditioner)

    # Pre-allocate arrays
    k0 = np.empty((n, n_points))
    idx = np.empty(n_points, dtype=np.uint32)

    # Populate columns of k0 as new points are selected
    k0[:, 0] = vfk0(sample, sample, gradient, gradient)
    idx[0] = np.argmin(k0[:, 0])
    if verbose:
        # TODO: use logging instead
        print(f'THIN: {1} of {n_points}')
    for i in range(1, n_points):
        smp_last = sample[[idx[i - 1]]]
        scr_last = gradient[[idx[i - 1]]]
        k0[:, i] = vfk0(sample, smp_last, gradient, scr_last)
        idx[i] = np.argmin(k0[:, 0] + 2 * np.sum(k0[:, 1:(i + 1)], axis=1))
        if verbose:
            # TODO: use logging instead
            print('THIN: %{i + 1} of {n_points}')
    return idx
