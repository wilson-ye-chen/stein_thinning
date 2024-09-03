"""Stein thinning for a Gaussian mixture model."""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname
from stein_thinning.stein import ksd, kmat
from stein_thinning.thinning import thin, _make_stein_integrand
from stein_thinning.kernel import make_imq, make_precon

if __name__ == '__main__':
    # Read MCMC output from files
    dir = join(dirname(__file__), 'data')
    smp = np.genfromtxt(join(dir, 'smp.csv'), delimiter=',')
    scr = np.genfromtxt(join(dir, 'scr.csv'), delimiter=',')

    # Run Stein Thinning
    idx = thin(smp, scr, 40)

    # Plot point-set over trace
    plt.figure()
    plt.plot(smp[:,0], smp[:,1], color=(0.4, 0.4, 0.4), linewidth=1)
    plt.plot(smp[idx, 0], smp[idx, 1], 'r.', markersize=16)

    # Compute KSD
    vfk0 = make_imq(smp, preconditioner='sclmed')
    ks_smp = ksd(_make_stein_integrand(smp, scr, vfk0=vfk0), smp.shape[0])
    idx_integrand = _make_stein_integrand(smp[idx], scr[idx], vfk0=vfk0)
    ks_x = ksd(idx_integrand, len(idx))

    # Print out the inverse of the preconditioner matrix
    print(make_precon(smp, preconditioner='sclmed'))

    # Visualise the Stein kernel matrix
    plt.matshow(kmat(idx_integrand, len(idx)))

    # Plot KSD curves
    plt.figure()
    h = np.log(range(1, ks_smp.size + 1))
    plt.plot(h, np.log(ks_smp), 'k-', linewidth=2)
    h = np.log(range(1, ks_x.size + 1))
    plt.plot(h, np.log(ks_x), 'r-', linewidth=2)
    plt.show()
