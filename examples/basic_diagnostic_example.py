"""Basic example for dmlkappa.

Simulate a PLR dataset, fit DML-PLR, and print diagnostics.
"""
from dmlkappa import compute_kappa_from_u
from dmlkappa.diagnostics import diagnostic_report
from dmlkappa.simulation import simulate_plr_once, fit_dml_plr


def main():
    n = 500
    overlap = "moderate"
    rho = 0.5
    Y, D, X = simulate_plr_once(n=n, overlap=overlap, rho=rho, random_state=123)
    out = fit_dml_plr(Y, D, X, n_folds=5, random_state=123)
    kappa = out["kappa"]
    print(diagnostic_report(kappa, n))


if __name__ == "__main__":
    main()
