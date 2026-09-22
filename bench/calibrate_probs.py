#!/usr/bin/env python3
"""Fit simple calibration formulas to the model's confidence and measure the ECE drop.

Input: per-item (confidence, correct) from ``data/calibration.json``.
Calibration maps the raw confidence p to a corrected probability q:

  power        q = p^gamma
  temperature  q = sigmoid( logit(p) / T )
  platt        q = sigmoid( a * logit(p) + b )
  beta         q = sigmoid( a*ln p - b*ln(1-p) + c )
  isotonic     monotone step fit (sklearn IsotonicRegression)

Each map is fit by minimising the negative log-likelihood on the training folds
and scored with 5-fold cross-validated ECE / NLL / Brier. Parameters are also
refit on the full data for reporting.

Usage: python3 calibrate_probs.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from _bootstrap import DATA

EPS = 1e-6


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return 100.0 * total


def nll(q, y):
    q = clip(q)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def brier(q, y):
    return float(np.mean((np.asarray(q) - np.asarray(y)) ** 2))


# --- calibration maps ------------------------------------------------------- #
def fit_power(p, y):
    res = minimize(lambda g: nll(clip(p) ** np.exp(g[0]), y), [0.0], method="Nelder-Mead")
    return ("power", np.exp(res.x[0]))


def fit_temperature(p, y):
    res = minimize(lambda t: nll(expit(logit(clip(p)) / np.exp(t[0])), y), [0.0], method="Nelder-Mead")
    return ("temperature", np.exp(res.x[0]))


def fit_platt(p, y):
    x = logit(clip(p))
    res = minimize(lambda w: nll(expit(w[0] * x + w[1]), y), [1.0, 0.0], method="Nelder-Mead")
    return ("platt", res.x)


def fit_beta(p, y):
    lp, lq = np.log(clip(p)), np.log(clip(1 - p))
    res = minimize(lambda w: nll(expit(w[0] * lp - w[1] * lq + w[2]), y),
                   [1.0, 1.0, 0.0], method="Nelder-Mead")
    return ("beta", res.x)


def fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return ("isotonic", iso)


def apply_map(fitted, p):
    name, params = fitted
    if name == "power":
        return clip(p) ** params
    if name == "temperature":
        return expit(logit(clip(p)) / params)
    if name == "platt":
        a, b = params
        return expit(a * logit(clip(p)) + b)
    if name == "beta":
        a, b, c = params
        return expit(a * np.log(clip(p)) - b * np.log(clip(1 - p)) + c)
    if name == "isotonic":
        return clip(params.predict(p))
    raise ValueError(name)


FITTERS = [fit_power, fit_temperature, fit_platt, fit_beta, fit_isotonic]


def cv_scores(p, y, fitter, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    q = np.zeros_like(p, float)
    for tr, te in skf.split(p, y):
        fitted = fitter(p[tr], y[tr])
        q[te] = apply_map(fitted, p[te])
    return {"ece": ece(q, y), "nll": nll(q, y), "brier": brier(q, y)}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=os.path.join(DATA, "calibration.json"))
    ap.add_argument("--out", dest="dst", default=os.path.join(DATA, "calibration_fits.json"))
    args = ap.parse_args()
    data = json.load(open(args.src))
    out = {}
    for task, d in data.items():
        p = np.array(d["confidence"], float)
        y = np.array(d["correct"], float)
        row = {"n": len(y), "raw": {"ece": ece(p, y), "nll": nll(p, y), "brier": brier(p, y)},
               "methods": {}}
        print(f"\n=== {task} (n={len(y)}) ===")
        print(f"  {'method':<12}{'ECE':>7}{'NLL':>8}{'Brier':>8}   fitted params")
        print(f"  {'raw':<12}{row['raw']['ece']:>7.2f}{row['raw']['nll']:>8.4f}{row['raw']['brier']:>8.4f}")
        for fitter in FITTERS:
            scores = cv_scores(p, y, fitter)
            fitted = fitter(p, y)  # refit on all data for the reported parameters
            name, params = fitted
            if name in ("power", "temperature"):
                pstr = f"{params:.3f}"
            elif name == "platt":
                pstr = f"a={params[0]:.3f}, b={params[1]:.3f}"
            elif name == "beta":
                pstr = f"a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}"
            else:
                pstr = "step function"
            print(f"  {name:<12}{scores['ece']:>7.2f}{scores['nll']:>8.4f}{scores['brier']:>8.4f}   {pstr}")
            stored = "step function" if name == "isotonic" else list(np.atleast_1d(params))
            row["methods"][name] = {"cv": scores, "params": stored}
        out[task] = row

    with open(args.dst, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
