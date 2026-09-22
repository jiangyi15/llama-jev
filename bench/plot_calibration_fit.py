#!/usr/bin/env python3
"""Per-task calibration before/after: raw confidence vs Platt-calibrated.

For each task we fit Platt scaling (q = sigmoid(a*logit(p)+b)) with 5-fold
cross-validation, then plot the reliability curve (accuracy in [x-0.05, x+0.05])
for the raw confidence and for the calibrated probability, against the diagonal.

Usage: python3 plot_calibration_fit.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from _bootstrap import DATA
from calibrate_probs import apply_map, ece, fit_platt

CENTERS = np.arange(0.05, 1.0, 0.1)
TITLES = {
    "pubmedqa": "PubMedQA · noul (2 options)",
    "imdb": "IMDB · noul (2 options)",
    "helpsteer2": "HelpSteer2 · score (5 levels)",
    "choice5": "Banking77 · choice (K=5)",
    "choice10": "Banking77 · choice (K=10)",
}


def sigma_hess(n1: float, n2: float) -> float:
    n = n1 + n2
    if n == 0:
        return 0.0
    p = n1 / n
    if p <= 0.0 or p >= 1.0:
        return 1.0 / math.sqrt(n)
    return 1.0 / math.sqrt(n1 / p ** 2 + n2 / (1.0 - p) ** 2)


def curve(conf, correct):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    xs, ys, ns, sig = [], [], [], []
    for x in CENTERS:
        m = (conf >= x - 0.05) & (conf <= x + 0.05)
        n = int(m.sum())
        if n == 0:
            continue
        n1 = float(correct[m].sum())
        n2 = n - n1
        xs.append(x)
        ys.append(n1 / n)
        ns.append(n)
        sig.append(sigma_hess(n1, n2))
    return np.array(xs), np.array(ys), np.array(ns), np.array(sig)


def cv_platt(p, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    q = np.zeros_like(p, float)
    for tr, te in skf.split(p, y):
        q[te] = apply_map(fit_platt(p[tr], y[tr]), p[te])
    return q


def main() -> int:
    data = json.load(open(os.path.join(DATA, "calibration.json")))
    tasks = [t for t in ("pubmedqa", "imdb", "helpsteer2", "choice5", "choice10") if t in data]

    cols = 2
    rows = (len(tasks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.4 * rows), squeeze=False)
    for i, task in enumerate(tasks):
        ax = axes[i // cols][i % cols]
        p = np.array(data[task]["confidence"], float)
        y = np.array(data[task]["correct"], float)
        q = cv_platt(p, y)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")

        for values, color, label in ((p, "#999999", "raw"), (q, "#1f77b4", "Platt-calibrated")):
            xs, ys, ns, sig = curve(values, y)
            ax.errorbar(xs, ys, yerr=sig, marker="o", ms=5, lw=1.3, capsize=3,
                        color=color, ecolor=color, alpha=0.9, label=label)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence x"); ax.set_ylabel("accuracy in [x-0.05, x+0.05]")
        ax.set_title(f"{TITLES.get(task, task)}\n"
                     f"ECE raw {ece(p, y):.1f} -> calibrated {ece(q, y):.1f} pts   (n={len(y)})",
                     fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    for j in range(len(tasks), rows * cols):
        axes[j // cols][j % cols].axis("off")

    out = os.path.join(DATA, "calibration_fit_plot.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")

    # text summary
    print(f"\n{'task':<12}{'raw ECE':>9}{'cal ECE':>9}{'a':>8}{'b':>8}")
    for task in tasks:
        p = np.array(data[task]["confidence"], float)
        y = np.array(data[task]["correct"], float)
        q = cv_platt(p, y)
        _, (a, b) = fit_platt(p, y)
        print(f"{task:<12}{ece(p, y):>9.2f}{ece(q, y):>9.2f}{a:>8.2f}{b:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
