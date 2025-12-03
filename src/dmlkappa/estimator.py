"""DMLKappaPLR: Main estimator class for DML-PLR with κ_DML diagnostics.

Implements cross-fitted DML for the Partially Linear Regression model
with built-in condition number diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .core import compute_kappa_from_u, compute_se_from_scores, effective_sample_size
from .diagnostics import classify_regime, KappaRegime


@dataclass
class DMLKappaPLRResult:
    """Container for DMLKappaPLR fit results."""

    theta_hat: float
    se: float
    kappa: float
    n: int
    u_hat: np.ndarray
    eps_hat: np.ndarray
    psi: np.ndarray


class DMLKappaPLR:
    """Cross-fitted DML estimator for the Partially Linear Regression model.

    Implements the estimator from Chernozhukov et al. (2018) with finite-sample
    conditioning diagnostics based on κ_DML (Saco, 202X).

    The PLR model is:
        Y = D θ₀ + g₀(X) + ε,   E[ε | D, X] = 0

    The DML estimator uses cross-fitting to estimate nuisance functions:
        m₀(X) = E[D | X]   (propensity/treatment model)
        ℓ₀(X) = E[Y | X]   (reduced-form outcome model)

    and computes:
        θ̂ = Σᵢ Ûᵢ V̂ᵢ / Σᵢ Ûᵢ²

    where Ûᵢ = Dᵢ - m̂(Xᵢ) and V̂ᵢ = Yᵢ - ĝ(Xᵢ).

    The condition number κ_DML = n / Σᵢ Ûᵢ² measures finite-sample stability.

    Parameters
    ----------
    learner_m : estimator
        Scikit-learn compatible regressor for m₀(X) = E[D | X].
    learner_g : estimator
        Scikit-learn compatible regressor for ℓ₀(X) = E[Y | X].
    n_splits : int, default=5
        Number of folds for cross-fitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    theta_hat_ : float
        Estimated treatment effect θ̂.
    se_ : float
        Standard error of θ̂.
    kappa_ : float
        DML condition number κ_DML.
    u_hat_ : np.ndarray
        Residualized treatments Ûᵢ = Dᵢ - m̂(Xᵢ).
    eps_hat_ : np.ndarray
        Residuals ε̂ᵢ = Yᵢ - ĝ(Xᵢ) - θ̂ Ûᵢ.
    diagnostics_ : dict
        Dictionary with diagnostic information.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from dmlkappa import DMLKappaPLR, simulate_plr
    >>> X, D, Y, info = simulate_plr(n=500, overlap="moderate")
    >>> model = DMLKappaPLR(
    ...     learner_m=RandomForestRegressor(random_state=0),
    ...     learner_g=RandomForestRegressor(random_state=0),
    ...     n_splits=5,
    ...     random_state=0
    ... )
    >>> model.fit(X, D, Y)
    >>> print(f"θ̂ = {model.theta_hat_:.3f}, κ = {model.kappa_:.2f}")
    """

    def __init__(
        self,
        learner_m,
        learner_g,
        n_splits: int = 5,
        random_state: Optional[int] = None,
    ):
        self.learner_m = learner_m
        self.learner_g = learner_g
        self.n_splits = n_splits
        self.random_state = random_state

        # Fitted attributes (set by fit)
        self.theta_hat_: Optional[float] = None
        self.se_: Optional[float] = None
        self.kappa_: Optional[float] = None
        self.u_hat_: Optional[np.ndarray] = None
        self.eps_hat_: Optional[np.ndarray] = None
        self.psi_: Optional[np.ndarray] = None
        self.n_: Optional[int] = None
        self.diagnostics_: Optional[Dict[str, Any]] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> "DMLKappaPLR":
        """Fit the DML-PLR model with cross-fitting.

        Parameters
        ----------
        X : np.ndarray of shape (n, p)
            Covariate matrix.
        D : np.ndarray of shape (n,)
            Treatment variable.
        Y : np.ndarray of shape (n,)
            Outcome variable.

        Returns
        -------
        self : DMLKappaPLR
            Fitted estimator.
        """
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        X = np.asarray(X)
        D = np.asarray(D).ravel()
        Y = np.asarray(Y).ravel()

        n = X.shape[0]
        if D.shape[0] != n or Y.shape[0] != n:
            raise ValueError("X, D, Y must have the same number of observations")

        self.n_ = n

        # Cross-fitting
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        m_oof = np.zeros(n)
        g_oof = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            # Clone learners for each fold
            m = clone(self.learner_m)
            g = clone(self.learner_g)

            # Fit on training fold
            m.fit(X[train_idx], D[train_idx])
            g.fit(X[train_idx], Y[train_idx])

            # Predict on test fold
            m_oof[test_idx] = m.predict(X[test_idx])
            g_oof[test_idx] = g.predict(X[test_idx])

        # Residualized treatment and outcome
        u_hat = D - m_oof
        v_hat = Y - g_oof

        # DML estimator: θ̂ = Σ Û V̂ / Σ Û²
        denom = np.sum(u_hat**2)
        if denom == 0:
            raise ValueError("Sum of Û² is zero; cannot compute θ̂")

        theta_hat = float(np.sum(u_hat * v_hat) / denom)

        # Residuals and scores
        eps_hat = Y - g_oof - theta_hat * u_hat
        psi = u_hat * eps_hat

        # Condition number and SE
        kappa = compute_kappa_from_u(u_hat)
        se = compute_se_from_scores(kappa, psi)

        # Store results
        self.theta_hat_ = theta_hat
        self.se_ = se
        self.kappa_ = kappa
        self.u_hat_ = u_hat
        self.eps_hat_ = eps_hat
        self.psi_ = psi

        # Build diagnostics dict
        regime = classify_regime(kappa)
        eff_n = effective_sample_size(kappa, n)
        self.diagnostics_ = {
            "kappa": kappa,
            "regime": regime.value,
            "effective_n": eff_n,
            "u_hat_var": float(np.var(u_hat)),
            "u_hat_mean": float(np.mean(u_hat)),
        }

        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

    def ci_(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Compute confidence interval for θ.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI).

        Returns
        -------
        (lower, upper) : tuple of float
            Confidence interval bounds.
        """
        self._check_fitted()
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        lower = self.theta_hat_ - z * self.se_
        upper = self.theta_hat_ + z * self.se_
        return (lower, upper)

    def regime_label(
        self, thresholds: Tuple[float, float] = (1.0, 3.0)
    ) -> str:
        """Return human-readable regime label based on κ_DML.

        Parameters
        ----------
        thresholds : tuple of (float, float), default=(1.0, 3.0)
            (thr_moderate, thr_severe) thresholds for regime classification.

        Returns
        -------
        str
            One of "well-conditioned", "moderately ill-conditioned",
            or "severely ill-conditioned".
        """
        self._check_fitted()
        thr_mod, thr_sev = thresholds
        regime = classify_regime(self.kappa_, thr_moderate=thr_mod, thr_severe=thr_sev)
        # Convert enum to readable string
        return regime.value.replace("_", " ")

    def summary(self, alpha: float = 0.05) -> str:
        """Return a human-readable summary of the fit.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for the confidence interval.

        Returns
        -------
        str
            Multi-line summary string.
        """
        self._check_fitted()
        ci_low, ci_high = self.ci_(alpha)
        lines = [
            "=" * 60,
            "DML-PLR Estimation Results with κ_DML Diagnostics",
            "=" * 60,
            f"  θ̂ (treatment effect)    : {self.theta_hat_:.6f}",
            f"  Standard error          : {self.se_:.6f}",
            f"  {int((1 - alpha) * 100)}% Confidence interval : [{ci_low:.6f}, {ci_high:.6f}]",
            "-" * 60,
            "Finite-Sample Conditioning Diagnostics",
            "-" * 60,
            f"  κ_DML                   : {self.kappa_:.4f}",
            f"  Regime                  : {self.regime_label()}",
            f"  Effective sample size   : {self.diagnostics_['effective_n']:.1f}",
            f"  Var(Û)                  : {self.diagnostics_['u_hat_var']:.6f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"DMLKappaPLR(θ̂={self.theta_hat_:.4f}, κ={self.kappa_:.2f}, "
                f"regime='{self.regime_label()}')"
            )
        return "DMLKappaPLR(not fitted)"
