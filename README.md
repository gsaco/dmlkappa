# dmlkappa

dmlkappa: a small diagnostic package implementing the condition number κ_DML and
finite-sample simulations from the note “Finite-Sample Conditioning in Double Machine Learning”.

Install (editable):

```bash
pip install -e .
```

Minimal usage:

```python
from dmlkappa.simulation import simulate_plr_once, fit_dml_plr
from dmlkappa.diagnostics import diagnostic_report

Y, D, X = simulate_plr_once(500, "moderate", 0.5, random_state=123)
out = fit_dml_plr(Y, D, X, n_folds=5, random_state=123)
print(diagnostic_report(out['kappa'], len(Y)))
```

Theory and proofs are contained in `note.tex` and `derivations.tex` in this repository.
# dmlkappa