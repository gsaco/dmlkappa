"""Diagnostics utilities for dmlkappa.

Provides regime classification and a small dataclass summarizing diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .core import effective_sample_size, compute_kappa_from_u


class KappaRegime(str, Enum):
    WELL_CONDITIONED = "well_conditioned"
    MODERATELY_ILL_CONDITIONED = "moderately_ill_conditioned"
    SEVERELY_ILL_CONDITIONED = "severely_ill_conditioned"


@dataclass
class KappaDiagnostic:
    kappa: float
    n: int
    regime: KappaRegime
    effective_n: float
    coverage_estimate: Optional[float] = None


def classify_regime(kappa: float, *, thr_moderate: float = 1.0, thr_severe: float = 3.0) -> KappaRegime:
    """Classify the kappa_DML regime.

    Defaults follow the note: kappa < 1 -> well, 1<=kappa<3 -> moderate, kappa>=3 -> severe.
    """
    if kappa < thr_moderate:
        return KappaRegime.WELL_CONDITIONED
    if kappa < thr_severe:
        return KappaRegime.MODERATELY_ILL_CONDITIONED
    return KappaRegime.SEVERELY_ILL_CONDITIONED


def diagnostic_report(kappa: float, n: int) -> str:
    """Return a human-readable multi-line diagnostic string.

    Summarizes kappa, regime, heuristic effective sample size, and brief recommendations.
    References to note.tex / derivations.tex notation: κ_DML, S_n, B_n, etc.
    """
    regime = classify_regime(kappa)
    eff_n = effective_sample_size(kappa, n)
    lines = []
    lines.append(f"κ_DML = {kappa:.4g} (n = {n})")
    lines.append(f"Regime: {regime.value}")
    lines.append(f"Heuristic effective sample size: n_eff ≈ {eff_n:.2f}")
    if regime == KappaRegime.WELL_CONDITIONED:
        lines.append("Recommendation: inference appears well-conditioned; standard DML SEs are likely reliable.")
    elif regime == KappaRegime.MODERATELY_ILL_CONDITIONED:
        lines.append("Recommendation: exercise caution — finite-sample adjustments or more folds may help.")
    else:
        lines.append("Recommendation: severely ill-conditioned; consider stronger regularization, improved overlap, or alternative estimators.")
    lines.append("See note.tex and derivations.tex for detailed justification and proofs.")
    return "\n".join(lines)


def diagnostic_from_u(u: np.ndarray) -> KappaDiagnostic:
    """Compute kappa and return a KappaDiagnostic from residualized treatments `u`.

    Parameters
    - u: array of U_i residualized treatments
    """
    u = np.asarray(u)
    n = u.shape[0]
    kappa = compute_kappa_from_u(u)
    regime = classify_regime(kappa)
    eff_n = effective_sample_size(kappa, n)
    return KappaDiagnostic(kappa=kappa, n=n, regime=regime, effective_n=eff_n)
