"""Numerically verify closed-form logistic results using grouped-binomial MLE."""
import csv
import json
from pathlib import Path

import numpy as np


def fit(design, successes, totals):
    """Fit grouped-binomial logistic regression with Newton's method."""
    x = np.asarray(design, dtype=float)
    a = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(x @ beta)))
        # The score gives the update direction; Fisher information scales it.
        fisher = x.T @ ((n*p*(1-p))[:, None] * x)
        step = np.linalg.solve(fisher, x.T @ (a-n*p))
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    else:
        raise AssertionError("Numerical MLE did not converge")
    p = 1 / (1 + np.exp(-(x @ beta)))
    # The inverse information matrix gives coefficient variances.
    covariance = np.linalg.inv(x.T @ ((n*p*(1-p))[:, None]*x))
    return beta, np.sqrt(np.diag(covariance))


out = Path(__file__).parent / "outputs"
regressions = json.loads((out / "regressions.json").read_text())
with (out / "figure3.csv").open() as f:
    question_effects = list(csv.DictReader(f))
checked = 0
# Refit each saved two-group result numerically, independent of the closed form.
for row in [*regressions.values(), *question_effects]:
    if "baseline_agree" not in row:
        continue
    beta, se = fit([[1, 0], [1, 1]],
                   [row["baseline_agree"], row["framed_agree"]],
                   [row["baseline_total"], row["framed_total"]])
    np.testing.assert_allclose(beta[1], float(row["beta"]), atol=1e-12)
    np.testing.assert_allclose(se[1], float(row["se"]), atol=1e-12)
    checked += 1

for scope, technical in [("two_types", "rq3_technical_two_types"),
                          ("all_technical", "rq1_all_technical")]:
    t, p = regressions[technical], regressions["rq3_plain_only"]
    # Columns: intercept, framed, plain, framed*plain.
    beta, se = fit([[1, 0, 0, 0], [1, 1, 0, 0],
                    [1, 0, 1, 0], [1, 1, 1, 1]],
                   [t["baseline_agree"], t["framed_agree"],
                    p["baseline_agree"], p["framed_agree"]],
                   [t["baseline_total"], t["framed_total"],
                    p["baseline_total"], p["framed_total"]])
    target = regressions[f"rq3_interaction_{scope}"]
    np.testing.assert_allclose(beta[3], target["beta"], atol=1e-12)
    np.testing.assert_allclose(se[3], target["se"], atol=1e-12)
    checked += 1

checks = json.loads((out / "checks.json").read_text())
assert checks["table3_cells_matching"] == checks["table3_cells_total"] == 32
assert checks["figure3_lowest_odds_ratio"] == "Effectiveness"
print(f"PASS: {checked} regression specifications agree with numerical MLE.")
print("PASS: all 32 Table 3 cells and Figure 3 ordering checks.")
