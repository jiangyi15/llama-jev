#!/usr/bin/env python3
"""Reliability plot with uncertainty: accuracy vs confidence in [x-0.05, x+0.05].

For each window centre x we take the N decisions whose confidence is in
[x-0.05, x+0.05], with N1 = Nt correct and N2 = Nf = N - Nt wrong:

    frac      = N1 / (N1 + N2)
    sigma_Ni  = sqrt(Ni)
    sigma_frac^2 = (d frac/dN1 * sigma_N1)^2 + (d frac/dN2 * sigma_N2)^2
                 = (N2/N^2)^2 * N1 + (N1/N^2)^2 * N2
                 = N1 * N2 / N^3
    sigma_frac   = sqrt(N1 * N2) / N^1.5

The error bar is frac +/- sigma_frac. Marker size scales with N. The dashed
diagonal is perfect calibration (accuracy = confidence).

Usage: python3 plot_calibration.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import DATA

CENTERS = np.arange(0.05, 1.0, 0.1)
TITLES = {
    "pubmedqa": "PubMedQA · noul (2 options)",
    "imdb": "IMDB · noul (2 options)",
    "helpsteer2": "HelpSteer2 · score (5 levels)",
    "choice5": "Banking77 · choice (K=5)",
    "choice10": "Banking77 · choice (K=10)",
}


def sigma_propagation(n1: float, n2: float) -> float:
    """Error propagation with sigma_N = sqrt(N): sqrt(N1*N2) / N^1.5."""
    n = n1 + n2
    return math.sqrt(n1 * n2) / (n ** 1.5) if n > 0 else 0.0


def sigma_hessian(n1: float, n2: float) -> float:
    """V = H^-1 from lnL = N1 ln p + N2 ln(1-p); sigma = sqrt(V).

    V = 1 / (N1/p^2 + N2/(1-p)^2). At the MLE p=N1/N this equals N1*N2/N^3
    (identical to the propagation result), but it stays non-zero at the
    boundaries (N1=0 or N2=0) where the plug-in variance would be 0.
    """
    n = n1 + n2
    if n == 0:
        return 0.0
    p = n1 / n
    if p <= 0.0 or p >= 1.0:          # limit N1->0 or N2->0 => I = N, V = 1/N
        return 1.0 / math.sqrt(n)
    return 1.0 / math.sqrt(n1 / p ** 2 + n2 / (1.0 - p) ** 2)


def curve(conf, correct):
    """Window centres, frac = N1/(N1+N2), counts, and both sigma estimates."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    xs, ys, ns, sp, sh = [], [], [], [], []
    for x in CENTERS:
        mask = (conf >= x - 0.05) & (conf <= x + 0.05)
        n = int(mask.sum())
        if n == 0:
            continue
        n1 = float(correct[mask].sum())
        n2 = n - n1
        xs.append(x)
        ys.append(n1 / (n1 + n2))
        ns.append(n)
        sp.append(sigma_propagation(n1, n2))
        sh.append(sigma_hessian(n1, n2))
    return np.array(xs), np.array(ys), np.array(ns), np.array(sp), np.array(sh)


def print_table(name, conf, correct):
    xs, ys, ns, sp, sh = curve(conf, correct)
    print(f"\n{name}  (window [x-0.05, x+0.05])")
    print(f"  {'x':>6}{'range':>14}{'N1':>5}{'N2':>5}{'frac':>8}{'sig_prop':>10}{'sig_hess':>10}{'frac+/-hess':>20}")
    for x, y, n, a, b in zip(xs, ys, ns, sp, sh):
        n1 = round(y * n)
        print(f"  {x:>6.2f}{f'[{x-0.05:.2f},{x+0.05:.2f}]':>14}{n1:>5}{n-n1:>5}"
              f"{y:>8.3f}{a:>10.3f}{b:>10.3f}{f'{y:.3f} +/- {b:.3f}':>20}")
    return xs, ys, ns, sp, sh


def main() -> int:
    data = json.load(open(os.path.join(DATA, "calibration.json")))
    tasks = [t for t in ("pubmedqa", "imdb", "helpsteer2", "choice5", "choice10") if t in data]

    # combined figure
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    # horizontal offsets so the datasets' error bars do not overlap
    offsets = np.linspace(-0.03, 0.03, len(tasks)) if len(tasks) > 1 else np.array([0.0])
    for off, task in zip(offsets, tasks):
        xs, ys, ns, sp, sh = curve(data[task]["confidence"], data[task]["correct"])
        ax.errorbar(xs + off, ys, yerr=sh, marker="o", ms=6, lw=1.3, capsize=3,
                    label=f"{TITLES.get(task, task)}  (acc={data[task]['accuracy']:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("confidence  x  (datasets offset horizontally by up to ±0.03)")
    ax.set_ylabel("accuracy in [x-0.05, x+0.05]")
    ax.set_title("Calibration with uncertainty (V = H^-1)\n"
                 "V = 1/(N1/p^2 + N2/(1-p)^2)   |   window width 0.10")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    combined = os.path.join(DATA, "calibration_plot_combined.png")
    fig.tight_layout(); fig.savefig(combined, dpi=150); plt.close(fig)
    print(f"wrote {combined}")

    # per-task grid
    n = len(tasks)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.3 * rows), squeeze=False)
    for i, task in enumerate(tasks):
        ax = axes[i // cols][i % cols]
        xs, ys, ns, sp, sig = print_table(TITLES.get(task, task),
                                          data[task]["confidence"], data[task]["correct"])
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        d = 0.012   # shift the two methods apart so their bars don't overlap
        ax.errorbar(xs - d, ys, yerr=sp, fmt="o", ms=5, lw=1.0, capsize=2,
                    color="#999999", ecolor="#999999", alpha=0.9, zorder=2,
                    label="propagation")
        ax.errorbar(xs + d, ys, yerr=sig, fmt="o", ms=5, lw=1.4, capsize=3,
                    color="#1f77b4", ecolor="#1f77b4", alpha=0.9, zorder=3,
                    label="V = H^-1")
        ax.scatter(xs + d, ys, s=30 + ns * 4, c="#1f77b4", alpha=0.35,
                   edgecolors="white", zorder=4)
        for x, y, k, a, b in zip(xs, ys, ns, sp, sig):
            ax.annotate(f"N={k}\n±{b:.2f}", (x + d, y), fontsize=7, ha="center",
                        va="bottom", xytext=(0, 6), textcoords="offset points", zorder=4)
        ax.axhline(data[task]["accuracy"], color="#d62728", lw=1, ls=":",
                   label=f"overall acc={data[task]['accuracy']:.2f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence x  (methods offset ±0.012)"); ax.set_ylabel("accuracy")
        ax.set_title(f"{TITLES.get(task, task)}\nECE={data[task]['ece']:.1f} pts, n={len(data[task]['correct'])}",
                     fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    grid = os.path.join(DATA, "calibration_plot.png")
    fig.tight_layout(); fig.savefig(grid, dpi=150); plt.close(fig)
    print(f"\nwrote {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
