import numpy as np
import pytest

from stein_thinning.kernel import make_precon, vfk0_imq


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


def test_vfk0_imq():
    x1 = np.array([1., 2., 3.])
    x2 = np.array([2., 3., 4.])
    s1 = np.array([0.5, 0.75, 1.5])
    s2 = np.array([1., 1.5, 3.])
    np.testing.assert_approx_equal(vfk0_imq(x1, x2, s1, s2, np.identity(3)), 3.5)
