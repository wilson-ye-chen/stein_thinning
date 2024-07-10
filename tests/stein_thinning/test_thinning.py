from os.path import join, dirname
from pathlib import Path

import numpy as np
import pytest

from stein_thinning.thinning import thin


@pytest.fixture
def demo_data_dir():
    return Path('stein_thinning') / 'sample_chains' / 'gmm'


@pytest.fixture
def demo_smp(demo_data_dir):
    return np.genfromtxt(demo_data_dir / 'smp.csv', delimiter=',')


@pytest.fixture
def demo_scr(demo_data_dir):
    return np.genfromtxt(demo_data_dir / 'scr.csv', delimiter=',')


def test_thin(demo_smp, demo_scr):
    idx = thin(demo_smp, demo_scr, 40)
    expected = np.array([
        68, 322, 268, 234, 161, 292, 229, 275, 259, 131, 400, 486, 207,
        120, 443, 430, 376, 411,  98, 293, 111, 372, 285, 427, 406, 246,
        148, 260, 296, 208,  79, 430, 369, 363, 462, 393, 321, 460, 373,
        114
    ])
    np.testing.assert_array_equal(idx, expected)
