"""Test Stein Thinning."""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname
from stein_thinning.thinning import thin

# Read MCMC output from files
dir = join(dirname(__file__), 'sample_chains/gmm')
smp = np.genfromtxt(join(dir, 'smp.csv'), delimiter=',')
scr = np.genfromtxt(join(dir, 'scr.csv'), delimiter=',')

# Run Stein Thinning
x, s, e = thin(smp, scr, 40)

# Plot point-set over trace
plt.plot(smp[:,0], smp[:,1], color=(0.4, 0.4, 0.4), linewidth=1)
plt.plot(x[:,0], x[:,1], 'r.', markersize=16)
plt.show()
