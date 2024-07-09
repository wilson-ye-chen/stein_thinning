import numpy as np
import pytest

from stein_thinning.kernel import make_precon


def test_make_precon():
    sample = np.array([
        [1., 2.],
        [3., 5.],
        [6., 9.],
    ])
    precon = make_precon(sample)
    np.testing.assert_array_equal(precon, np.identity(2))

    precon = make_precon(sample, 'med')
    np.testing.assert_array_almost_equal(precon, np.identity(2) / 25)

    precon = make_precon(sample, 'sclmed')
    np.testing.assert_array_almost_equal(precon, np.identity(2) / 25 * np.log(3))

    precon = make_precon(sample, 'smpcov')
    np.testing.assert_array_almost_equal(precon, np.linalg.inv(np.cov(sample, rowvar=False)))

    precon = make_precon(sample, 2.)
    np.testing.assert_array_almost_equal(precon, np.identity(2) / 2.)

    precon = make_precon(sample, '3.5')
    np.testing.assert_array_almost_equal(precon, np.identity(2) / 3.5)

    with pytest.raises(ValueError):
        precon = make_precon(sample, 'foo')
