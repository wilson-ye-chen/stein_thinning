import numpy as np

from stein_thinning.kernel import make_imq
from stein_thinning.stein import kmat, ksd
from stein_thinning.thinning import _make_stein_integrand


def test_kmat():
    x = np.array([1.0, 2.0, 5.0, 7.0])
    def integrand(ind1, ind2):
        return (x[ind1] - x[ind2]) ** 2
    result = kmat(integrand, len(x))
    expected = np.array([
        [0.,  1.,  16., 36.],
        [1.,  0.,  9.,  25.],
        [16., 9.,  0.,  4. ],
        [36., 25., 4.,  0],
    ])
    np.testing.assert_array_equal(result, expected)


def test_ksd():
    x = np.array([1.0, 2.0, 5.0, 7.0])
    def integrand(ind1, ind2):
        return (x[ind1] - x[ind2]) ** 2
    s = np.zeros(4).reshape(4, 1)
    result = ksd(integrand, len(x))
    expected = np.array([0., np.sqrt(2) / 2, np.sqrt(52) / 3, np.sqrt(182) / 4])
    np.testing.assert_array_equal(result, expected)


def test_demo_kmat(demo_smp, demo_scr, demo_kmat):
    integrand = _make_stein_integrand(demo_smp, demo_scr, standardize=False)
    result = kmat(integrand, demo_smp.shape[0])
    np.testing.assert_array_almost_equal(result, demo_kmat)
