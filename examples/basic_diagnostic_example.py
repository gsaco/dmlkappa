"""Basic example for dmlkappa.

Simulate a PLR dataset, fit DML-PLR with κ_DML diagnostics.

This example demonstrates the main API of the dmlkappa package.
"""
from sklearn.ensemble import RandomForestRegressor

from dmlkappa import simulate_plr, DMLKappaPLR


def main():
    # 1. Simulate PLR data with moderate overlap
    print("Simulating PLR data...")
    X, D, Y, info = simulate_plr(
        n=500,
        p=10,
        rho=0.5,
        overlap="moderate",
        theta0=1.0,
        random_state=123
    )
    print(f"  True θ₀ = {info['theta0']}")
    print(f"  Overlap = {info['overlap']}")
    print()

    # 2. Fit DML-PLR with Random Forest learners
    print("Fitting DML-PLR estimator...")
    model = DMLKappaPLR(
        learner_m=RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=123),
        learner_g=RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=123),
        n_splits=5,
        random_state=123
    )
    model.fit(X, D, Y)
    print()

    # 3. Print summary with κ_DML diagnostics
    print(model.summary())


if __name__ == "__main__":
    main()
