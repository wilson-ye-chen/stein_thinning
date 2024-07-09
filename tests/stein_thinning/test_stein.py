import numpy as np

from stein_thinning.stein import kmat


def test_kmat():
    def kernel(x, y, sx, sy):
        return (x - y) ** 2
    x = np.arange(4)
    s = np.zeros(4)
    result = kmat(x, s, kernel)
    expected = np.array([
        [0, 1, 4, 9],
        [1, 0, 1, 4],
        [4, 1, 0, 1],
        [9, 4, 1, 0],
    ])
    np.testing.assert_array_equal(result, expected)
