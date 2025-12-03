import pytest


def test_simulation_grid_small():
    # try importing sklearn; skip test if import raises any exception in this env
    try:
        import sklearn  # noqa: F401
    except Exception:
        pytest.skip("sklearn not available or import fails in this environment")
    from dmlkappa.simulation import run_plr_simulation_grid

    n_list = [100]
    overlap_list = ["moderate"]
    rho_list = [0.2]
    results, summary = run_plr_simulation_grid(n_list, overlap_list, rho_list, n_rep=2, random_state=0)
    # basic sanity checks
    assert not results["kappa"].isnull().any()
    assert (results["coverage"].between(0, 1)).all()
    assert summary.shape[0] == 1
