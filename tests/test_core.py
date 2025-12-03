import numpy as np

from dmlkappa.core import (
    compute_kappa_from_u,
    compute_se_from_scores,
    compute_se_from_u_eps,
    effective_sample_size,
)


def test_kappa_basic():
    u = np.array([1.0, 2.0, -1.0])
    k = compute_kappa_from_u(u)
    assert np.isfinite(k)
    assert k == 3 / np.sum(u ** 2)


def test_se_from_scores_vs_u_eps():
    u = np.array([1.0, 2.0, -1.0])
    eps = np.array([0.5, -0.2, 0.1])
    psi = u * eps
    k = compute_kappa_from_u(u)
    se1 = compute_se_from_scores(k, psi)
    se2 = compute_se_from_u_eps(u, eps)
    assert np.allclose(se1, se2)


def test_effective_sample_size():
    n = 100
    kappa = 2.0
    neff = effective_sample_size(kappa, n)
    assert neff == n / (1 + kappa ** 2)
