"""Core utilities for dmlkappa.

Implements the DML condition number kappa and SE estimators.

Notation follows the note and derivations: n is sample size, U_i are residualized
treatments (D_i - m_hat(X_i)), eps_i are residualized outcome errors, and
psi_i = U_i * eps_i is the orthogonal score. The DML condition number is
    kappa_DML = n / sum_i U_i^2

The plug-in SE estimator implemented is
    SE_hat = (kappa / sqrt(n)) * sqrt( (1/n) * sum_i psi_i^2 )
which is algebraically equivalent to the expression in the note using U_i^2 * eps_i^2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def compute_kappa_from_u(u: np.ndarray) -> float:
    """Compute kappa_DML = n / sum_i U_i^2.

    Parameters
    - u: 1d array of residualized treatments U_i = D_i - m_hat(X_i).

    Returns
    - kappa (float)

    Raises
    - ValueError if `u` is empty or all zeros (would lead to division by zero).
    """
    u = np.asarray(u)
    if u.ndim != 1:
        raise ValueError("`u` must be a 1D array of residualized treatments")
    n = u.shape[0]
    if n == 0:
        raise ValueError("`u` is empty")
    s = np.sum(u ** 2)
    if s == 0:
        raise ValueError("sum of U_i^2 is zero; cannot compute kappa")
    return float(n / s)


def compute_se_from_scores(kappa: float, scores: np.ndarray) -> float:
    """Compute plug-in standard error from orthogonal scores psi_i = U_i * eps_i.

    Formula:
        SE_hat = (kappa / sqrt(n)) * sqrt( (1/n) * sum_i psi_i^2 )

    Parameters
    - kappa: computed kappa_DML
    - scores: 1d array of psi_i = U_i * eps_i

    Returns
    - se_hat (float)
    """
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("`scores` must be a 1D array")
    n = scores.shape[0]
    if n == 0:
        raise ValueError("`scores` is empty")
    var_hat = np.mean(scores ** 2)
    se = (kappa / np.sqrt(n)) * np.sqrt(var_hat)
    return float(se)


def compute_se_from_u_eps(u: np.ndarray, eps: np.ndarray) -> float:
    """Compute the SE_hat using U and eps arrays directly.

    Implements the expression in the note
        SE_hat = (kappa / sqrt(n)) * sqrt( (1/n) * sum_i U_i^2 * eps_i^2 ).

    This function calls `compute_kappa_from_u` internally.
    """
    u = np.asarray(u)
    eps = np.asarray(eps)
    if u.shape != eps.shape:
        raise ValueError("`u` and `eps` must have the same shape")
    kappa = compute_kappa_from_u(u)
    n = u.shape[0]
    if n == 0:
        raise ValueError("empty arrays")
    quantity = np.mean((u ** 2) * (eps ** 2))
    se = (kappa / np.sqrt(n)) * np.sqrt(quantity)
    return float(se)


def effective_sample_size(kappa: float, n: int) -> float:
    """Heuristic effective sample size that shrinks as kappa grows.

    This is a heuristic: n_eff = n / (1 + kappa**2).

    Parameters
    - kappa: kappa_DML
    - n: nominal sample size

    Returns
    - n_eff (float)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return float(n / (1.0 + float(kappa) ** 2))
