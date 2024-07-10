import numpy as np
from scipy.stats import multivariate_normal as mvn

from stein_thinning.kernel import vfk0_imq, make_precon
from stein_thinning.thinning import thin, thin_gf, _make_stein_integrand, _greedy_search


def test_thin(demo_smp, demo_scr):
    idx = thin(demo_smp, demo_scr, 40)
    expected = np.array([
        68, 322, 268, 234, 161, 292, 229, 275, 259, 131, 400, 486, 207,
        120, 443, 430, 376, 411,  98, 293, 111, 372, 285, 427, 406, 246,
        148, 260, 296, 208,  79, 430, 369, 363, 462, 393, 321, 460, 373,
        114
    ])
    np.testing.assert_array_almost_equal(idx, expected)

    preconditioner = make_precon(demo_smp, 'id')
    def kernel1(sample1, sample2, gradient1, gradient2):
        return vfk0_imq(sample1, sample2, gradient1, gradient2, preconditioner)
    integrand = _make_stein_integrand(demo_smp, demo_scr, vfk0=kernel1)
    idx = _greedy_search(40, integrand)
    np.testing.assert_array_almost_equal(idx, expected)

    def kernel2(sample1, sample2, gradient1, gradient2):
        return vfk0_imq(sample1, sample2, gradient1, gradient2, preconditioner, beta=-0.75)
    integrand = _make_stein_integrand(demo_smp, demo_scr, vfk0=kernel2)
    idx = _greedy_search(40, integrand)
    expected = np.array([
        68, 322, 268, 234, 161, 292, 229, 276, 259, 131, 207, 431, 486,
        120, 457, 430, 412, 376, 111, 101,  97, 332, 394, 123, 429, 109,
        349,  79, 466, 114, 458, 371, 296, 284,  89, 317, 485, 392, 261,
        246
    ])
    np.testing.assert_array_almost_equal(idx, expected)


def test_thin_gf():
    rng = np.random.default_rng(12345)

    n = 1000
    m = 20

    # obtain a sample from a bivariate Gaussian distribution
    means = np.array([0., 0.])
    covs = np.array([
        [1., 0.8],
        [0.8, 1.],
    ])

    sample = mvn.rvs(mean=means, cov=covs, size=n, random_state=rng)
    gradient = np.einsum('kj,ij->ik', np.linalg.inv(covs), means - sample)

    # use the standard Stein thinning
    idx = thin(sample, gradient, m)
    expected = np.array([164, 60, 859, 821, 66, 870, 327, 885, 677, 944, 111, 601, 950,
       249, 584, 795, 174, 317, 792, 637])
    np.testing.assert_array_equal(idx, expected)

    # confirm that the gradient-free thinning is equivalent to standard thinning
    # if the true distribution is used as the proxy
    log_p = mvn.logpdf(sample, mean=means, cov=covs)
    idx2 = thin_gf(sample, log_p, log_p, gradient, m)
    np.testing.assert_array_equal(idx2, idx)

    # construct a simple bivariate Gaussian proxy based on the sample mean and covariance
    sample_mean = np.mean(sample, axis=0)
    sample_cov = np.cov(sample, rowvar=False, ddof=means.shape[0])
    log_q = mvn.logpdf(sample, mean=sample_mean, cov=sample_cov)
    gradient_q = np.einsum('kj,ij->ik', np.linalg.inv(sample_cov), sample_mean - sample)

    idx3 = thin_gf(sample, log_p, log_q, gradient_q, m)
    expected = np.array([302, 995, 914, 931, 889, 918, 65, 714, 885, 46, 601, 88, 111,
        16, 478, 462, 750, 79, 783, 739])
    np.testing.assert_array_equal(idx3, expected)
