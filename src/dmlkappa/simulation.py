"""Simulation engine for PLR DGP and a cross-fitted DML-PLR estimator.

Implements a simple PLR DGP used in the note and a cross-fitted DML estimator
that returns theta_hat, standard error, confidence interval, kappa, and residuals.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
# sklearn imports are deferred to function scope to avoid import-time issues in some environments

from .core import compute_kappa_from_u, compute_se_from_scores, compute_se_from_u_eps


def _toeplitz_cov(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(np.subtract.outer(idx, idx))


def simulate_plr_once(n: int, overlap: str, rho: float, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one PLR dataset.

    DGP:
      X ~ N(0, Sigma) where Sigma is Toeplitz with parameter rho (p=10)
      D = X^T beta_D + U, U ~ N(0, sigma_U^2)
      Y = D * theta_0 + g0(X) + eps, eps ~ N(0,1)

    Overlap levels determine sigma_U (larger sigma_U -> more overlap):
      - 'high': sigma_U = 2.0
      - 'moderate': sigma_U = 1.0
      - 'low': sigma_U = 0.3

    Returns (Y, D, X).
    """
    rs = np.random.RandomState(random_state)
    p = 10
    Sigma = _toeplitz_cov(p, rho)
    X = rs.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    # coefficients
    beta_D = np.ones(p) * 0.5 / p
    gamma = rs.normal(scale=0.5, size=p)
    theta_0 = 1.0
    # overlap mapping
    overlap_map = {"high": 2.0, "moderate": 1.0, "low": 0.3}
    if overlap not in overlap_map:
        raise ValueError("overlap must be one of 'high', 'moderate', 'low'")
    sigma_U = overlap_map[overlap]
    U = rs.normal(scale=sigma_U, size=n)
    D = X.dot(beta_D) + U
    g0 = X.dot(gamma * np.sin(1.0))  # simple transformation; uses sin(X_j) flavor
    eps = rs.normal(scale=1.0, size=n)
    Y = D * theta_0 + g0 + eps
    return Y, D, X


def fit_dml_plr(Y: np.ndarray, D: np.ndarray, X: np.ndarray, *, n_folds: int = 5, learner: str = "rf", random_state: Optional[int] = None) -> Dict[str, Any]:
    """Fit a cross-fitted DML-PLR estimator.

    Returns a dict containing:
      - theta_hat
      - se_hat
      - ci: (lower, upper)
      - kappa
      - u: residualized treatment (out-of-fold)
      - eps: eps_hat as defined in the note (Y - g_hat - theta_hat * U)
    """
    n = Y.shape[0]
    if X.shape[0] != n or D.shape[0] != n:
        raise ValueError("Shapes of Y, D, X must align on first axis")
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    m_oof = np.zeros(n)
    g_oof = np.zeros(n)

    # import learners lazily to avoid import-time failures in some environments
    if learner not in {"rf"}:
        learner = "rf"
    from sklearn.ensemble import RandomForestRegressor

    for train_idx, test_idx in kf.split(X):
        m = RandomForestRegressor(random_state=random_state)
        g = RandomForestRegressor(random_state=random_state)
        m.fit(X[train_idx], D[train_idx])
        g.fit(X[train_idx], Y[train_idx])
        m_oof[test_idx] = m.predict(X[test_idx])
        g_oof[test_idx] = g.predict(X[test_idx])

    u_hat = D - m_oof
    v_hat = Y - g_oof  # initial residualized outcome
    denom = np.sum(u_hat ** 2)
    if denom == 0:
        raise ValueError("Denominator sum U_i^2 is zero in theta_hat computation")
    theta_hat = float(np.sum(u_hat * v_hat) / denom)
    eps_hat = Y - g_oof - theta_hat * u_hat
    psi = u_hat * eps_hat
    kappa = compute_kappa_from_u(u_hat)
    # SE: either from scores or u and eps
    se_hat = compute_se_from_scores(kappa, psi)
    ci_lower = theta_hat - 1.96 * se_hat
    ci_upper = theta_hat + 1.96 * se_hat
    return {
        "theta_hat": theta_hat,
        "se_hat": se_hat,
        "ci": (ci_lower, ci_upper),
        "kappa": kappa,
        "u": u_hat,
        "eps": eps_hat,
    }


def run_plr_simulation_grid(n_list, overlap_list, rho_list, n_rep: int, random_state: Optional[int] = None):
    """Run a simulation grid and return detailed and summary DataFrames.

    Returns (results_df, summary_df)
    """
    rows = []
    rng = np.random.RandomState(random_state)
    for n in n_list:
        for overlap in overlap_list:
            for rho in rho_list:
                for rep in range(int(n_rep)):
                    rs_seed = rng.randint(0, 2 ** 31 - 1)
                    Y, D, X = simulate_plr_once(n=int(n), overlap=overlap, rho=float(rho), random_state=int(rs_seed))
                    out = fit_dml_plr(Y, D, X, n_folds=5, random_state=rs_seed)
                    theta_hat = out["theta_hat"]
                    se_hat = out["se_hat"]
                    ci_low, ci_high = out["ci"]
                    kappa = out["kappa"]
                    coverage = float((ci_low <= 1.0) and (1.0 <= ci_high))
                    rows.append({
                        "n": int(n),
                        "overlap": overlap,
                        "rho": float(rho),
                        "rep": int(rep),
                        "theta_hat": float(theta_hat),
                        "se_hat": float(se_hat),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "kappa": float(kappa),
                        "coverage": coverage,
                    })
    results_df = pd.DataFrame(rows)
    # summary
    summary = (
        results_df.groupby(["n", "overlap", "rho"])
        .agg(kappa_mean=("kappa", "mean"), coverage=("coverage", "mean"), rmse=("theta_hat", lambda x: float(np.sqrt(np.mean((x - 1.0) ** 2)))))
        .reset_index()
    )
    return results_df, summary
