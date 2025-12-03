import numpy as np

from dmlkappa.diagnostics import (
    classify_regime,
    diagnostic_report,
    diagnostic_from_u,
    KappaRegime,
)


def test_classify_regime_default_thresholds():
    assert classify_regime(0.5) == KappaRegime.WELL_CONDITIONED
    assert classify_regime(1.0) == KappaRegime.MODERATELY_ILL_CONDITIONED
    assert classify_regime(2.5) == KappaRegime.MODERATELY_ILL_CONDITIONED
    assert classify_regime(3.0) == KappaRegime.SEVERELY_ILL_CONDITIONED
    assert classify_regime(10.0) == KappaRegime.SEVERELY_ILL_CONDITIONED


def test_diagnostic_report_contains_keywords():
    report = diagnostic_report(2.0, 100)
    assert "κ_DML" in report or "kappa" in report.lower()
    assert "Regime" in report
    assert "effective" in report.lower()
    assert "Recommendation" in report or "recommendation" in report.lower()


def test_diagnostic_from_u():
    u = np.array([0.5, 1.0, -0.5])
    diag = diagnostic_from_u(u)
    assert diag.n == 3
    assert np.isfinite(diag.kappa)
    assert diag.regime in KappaRegime
    assert diag.effective_n > 0
