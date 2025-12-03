"""dmlkappa

Finite-Sample Conditioning Diagnostics for Double Machine Learning.

This package implements the condition number κ_DML and related diagnostics
for the DML-PLR estimator, as described in:

    Saco (202X), "Finite-Sample Conditioning in Double Machine Learning:
    A Short Communication"

Main API:
    - DMLKappaPLR: Main estimator class with cross-fitting and diagnostics
    - simulate_plr: Simulate data from a PLR model
    - compute_kappa_from_u: Compute κ_DML from residualized treatments
"""
from .core import (
    compute_kappa_from_u,
    compute_se_from_scores,
    compute_se_from_u_eps,
    effective_sample_size,
)
from .diagnostics import (
    KappaRegime,
    KappaDiagnostic,
    classify_regime,
    diagnostic_report,
    diagnostic_from_u,
)
from .estimator import DMLKappaPLR
from .simulation import simulate_plr, simulate_plr_once, fit_dml_plr

__version__ = "0.1.0"

__all__ = [
    # Estimator
    "DMLKappaPLR",
    # Simulation
    "simulate_plr",
    "simulate_plr_once",
    "fit_dml_plr",
    # Core functions
    "compute_kappa_from_u",
    "compute_se_from_scores",
    "compute_se_from_u_eps",
    "effective_sample_size",
    # Diagnostics
    "KappaRegime",
    "KappaDiagnostic",
    "classify_regime",
    "diagnostic_report",
    "diagnostic_from_u",
]
