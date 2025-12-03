"""Tests for DMLKappaPLR estimator class."""
import pytest


def test_dmlkappa_plr_basic():
    """Test basic fit and attributes of DMLKappaPLR."""
    # Skip if sklearn import fails in this environment
    try:
        import sklearn  # noqa: F401
    except Exception:
        pytest.skip("sklearn not available or import fails in this environment")

    import numpy as np
    from dmlkappa import simulate_plr, DMLKappaPLR
    from sklearn.linear_model import Ridge

    # Use a fast linear model for testing
    X, D, Y, info = simulate_plr(n=200, p=5, overlap="high", random_state=0)

    model = DMLKappaPLR(
        learner_m=Ridge(alpha=1.0),
        learner_g=Ridge(alpha=1.0),
        n_splits=2,
        random_state=0
    )
    model.fit(X, D, Y)

    # Check fitted attributes exist and are finite
    assert model.theta_hat_ is not None
    assert np.isfinite(model.theta_hat_)
    assert np.isfinite(model.se_)
    assert np.isfinite(model.kappa_)
    assert model.kappa_ > 0

    # Check CI
    ci_low, ci_high = model.ci_(alpha=0.05)
    assert ci_low < ci_high

    # Check regime label is a string
    regime = model.regime_label()
    assert isinstance(regime, str)
    assert "conditioned" in regime

    # Check summary is a string
    summary = model.summary()
    assert isinstance(summary, str)
    assert "θ̂" in summary or "theta" in summary.lower()


def test_simulate_plr_returns_info():
    """Test that simulate_plr returns correct structure."""
    try:
        import sklearn  # noqa: F401
    except Exception:
        pytest.skip("sklearn not available")

    from dmlkappa import simulate_plr

    X, D, Y, info = simulate_plr(n=100, p=5, overlap="moderate", random_state=42)

    assert X.shape == (100, 5)
    assert D.shape == (100,)
    assert Y.shape == (100,)
    assert "theta0" in info
    assert "overlap" in info
    assert info["overlap"] == "moderate"
