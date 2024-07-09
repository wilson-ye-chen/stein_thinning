import numpy as np

from stein_thinning.stein import kmat, ksd


def test_kmat():
    def kernel(x, y, sx, sy):
        return (x - y) ** 2
    x = np.array([1.0, 2.0, 5.0, 7.0])
    s = np.zeros(4)
    result = kmat(x, s, kernel)
    expected = np.array([
        [0.,  1.,  16., 36.],
        [1.,  0.,  9.,  25.],
        [16., 9.,  0.,  4. ],
        [36., 25., 4.,  0],
    ])
    np.testing.assert_array_equal(result, expected)


def test_ksd():
    def kernel(x, y, sx, sy):
        return (x - y) ** 2
    x = np.array([1.0, 2.0, 5.0, 7.0])
    s = np.zeros(4).reshape(4, 1)
    result = ksd(x, s, kernel)
    expected = np.array([0., np.sqrt(2) / 2, np.sqrt(52) / 3, np.sqrt(182) / 4])
    np.testing.assert_array_equal(result, expected)
