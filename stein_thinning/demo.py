"""Test Stein Thinning."""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname
from stein_thinning.stein import ksd, kmat
from stein_thinning.thinning import thin
from stein_thinning.kernel import make_imq, make_precon

# Read MCMC output from files
dir = join(dirname(__file__), 'sample_chains/gmm')
smp = np.genfromtxt(join(dir, 'smp.csv'), delimiter=',')
scr = np.genfromtxt(join(dir, 'scr.csv'), delimiter=',')

# Run Stein Thinning
iSel = thin(smp, scr, 40)

# Plot point-set over trace
plt.figure()
plt.plot(smp[:,0], smp[:,1], color=(0.4, 0.4, 0.4), linewidth=1)
plt.plot(smp[iSel, 0], smp[iSel, 1], 'r.', markersize=16)

# Compute KSD
vfk0 = make_imq(smp, scr, pre='sclmed')
ks_smp = ksd(smp, scr, vfk0)
ks_x = ksd(smp[iSel], scr[iSel], vfk0)

# Print out the inverse of the preconditioner matrix
print(make_precon(smp, scr, pre='sclmed'))

# Visualise the Stein kernel matrix
plt.matshow(kmat(smp[iSel], scr[iSel], vfk0))

# Plot KSD curves
plt.figure()
h = np.log(range(1, ks_smp.size + 1))
plt.plot(h, np.log(ks_smp), 'k-', linewidth=2)
h = np.log(range(1, ks_x.size + 1))
plt.plot(h, np.log(ks_x), 'r-', linewidth=2)
plt.show()
