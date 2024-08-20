from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def demo_data_dir():
    return Path('stein_thinning') / 'demo' / 'data'


@pytest.fixture
def test_data_dir():
    return Path('tests') / 'stein_thinning' / 'data'


@pytest.fixture
def demo_smp(demo_data_dir):
    return np.genfromtxt(demo_data_dir / 'smp.csv', delimiter=',')


@pytest.fixture
def demo_scr(demo_data_dir):
    return np.genfromtxt(demo_data_dir / 'scr.csv', delimiter=',')


@pytest.fixture
def demo_kmat(test_data_dir):
    return np.genfromtxt(test_data_dir / 'demo_kmat.csv', delimiter=',')
