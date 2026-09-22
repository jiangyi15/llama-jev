#!/usr/bin/env python3
"""Significance of calibration: is accuracy = confidence?

Two tests, using the per-item confidences saved by ``check_calibration.py``:

1. Per-window Wald test. For each sliding window [x-0.05, x+0.05], test
   H0: frac = x, with z = (frac - x) / sigma_hess, sigma_hess = sqrt(V),
   V = 1/(N1/p^2 + N2/(1-p)^2).

2. Global Hosmer-Lemeshow test over 10 equal-width confidence bins:
   chi2 = sum_b (O_b - E_b)^2 / (N_b * pbar_b * (1 - pbar_b)),
   where E_b = sum of confidences in bin b. dof = (#bins used) - 2.

Usage: python3 check_calibration_significance.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy import stats

from _bootstrap import DATA

CENTERS = np.arange(0.05, 1.0, 0.1)


def sigma_hessian(n1: float, n2: float) -> float:
    n = n1 + n2
    if n == 0:
        return 0.0
    p = n1 / n
    if p <= 0.0 or p >= 1.0:
        return 1.0 / math.sqrt(n)
    return 1.0 / math.sqrt(n1 / p ** 2 + n2 / (1.0 - p) ** 2)


def window_test(conf, correct):
    rows = []
    for x in CENTERS:
        mask = (conf >= x - 0.05) & (conf <= x + 0.05)
        n = int(mask.sum())
        if n == 0:
            continue
        n1 = float(correct[mask].sum())
        n2 = n - n1
        frac = n1 / n
        s = sigma_hessian(n1, n2)
        if s <= 0:
            continue
        z = (frac - x) / s
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append((x, n1, n2, frac, s, z, p))
    return rows


def hosmer_lemeshow(conf, correct, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    chi2 = 0.0
    used = 0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        observed = float(correct[mask].sum())
        expected = float(conf[mask].sum())
        pbar = expected / n
        if pbar <= 0.0 or pbar >= 1.0:
            continue
        chi2 += (observed - expected) ** 2 / (n * pbar * (1 - pbar))
        used += 1
    dof = max(1, used - 2)
    return chi2, dof, float(stats.chi2.sf(chi2, dof))


def main() -> int:
    data = json.load(open(os.path.join(DATA, "calibration.json")))
    print("H0: accuracy = confidence  (per-window Wald, then global Hosmer-Lemeshow)\n")
    for task, d in data.items():
        conf = np.array(d["confidence"])
        correct = np.array(d["correct"], float)
        rows = window_test(conf, correct)
        sig = sum(1 for r in rows if r[6] < 0.05)
        print(f"{task}  (n={len(correct)})")
        print(f"  {'x':>6}{'N1':>5}{'N2':>5}{'frac':>8}{'sigma':>8}{'z':>7}{'p':>8}  sig95")
        for x, n1, n2, frac, s, z, p in rows:
            print(f"  {x:>6.2f}{n1:>5.0f}{n2:>5.0f}{frac:>8.3f}{s:>8.3f}{z:>7.2f}{p:>8.3f}"
                  f"  {'YES' if p < 0.05 else 'no'}")
        chi2, dof, pval = hosmer_lemeshow(conf, correct)
        verdict = "calibrated" if pval > 0.05 else "MISCALIBRATED"
        direction = ""
        if pval <= 0.05:
            direction = " (underconfident)" if correct.mean() > conf.mean() else " (overconfident)"
        print(f"  -> {sig}/{len(rows)} windows differ from y=x at 95%;  "
              f"HL chi2={chi2:.2f} dof={dof} p={pval:.4f}  {verdict}{direction}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
