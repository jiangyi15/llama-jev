#!/usr/bin/env python3
"""Compare the llama-jev API against Jev (jevals.com release 2026-09-18).

Everything is scored with Jevals' own metric so the numbers are directly
comparable:

    Decision Score = 100 * (1 - L_system / L_prior)

where the per-item loss is the multiclass Brier score (choice / noul) or the
ranked probability score (score), averaged over repeats; ``L_prior`` is the
same loss for the label base rates on the same items.

Data comes from the cloned ``data/jevals-data`` repo (suites + Jev run logs)
and the source datasets in ``./data``.

Usage::

    python3 compare_with_jev.py --llama-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np

from _bootstrap import DATA, JE, SUITE

import jev_server as jev

# 79 single characters (A-Z a-z 0-9 + safe symbols) so single-token grammars can
# address up to 79 options, enough for Banking77's 77 intents.
ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits + "!#$%&()*+/:;<=>?@"


# --------------------------------------------------------------------------- #
# Jevals metrics (validated to reproduce the published board)
# --------------------------------------------------------------------------- #
def brier(probs: list[float], target: int) -> float:
    return sum((p - (1.0 if i == target else 0.0)) ** 2 for i, p in enumerate(probs))


def rps(probs: list[float], target: int) -> float:
    K = len(probs)
    cum, running = [], 0.0
    for p in probs:
        running += p
        cum.append(running)
    return sum((cum[i] - (1.0 if target <= i else 0.0)) ** 2 for i in range(K - 1)) / (K - 1)


def decision_score(system_losses: list[float], prior_losses: list[float]) -> float:
    ls = float(np.mean(system_losses))
    lp = float(np.mean(prior_losses))
    return 100.0 * (1.0 - ls / lp) if lp > 0 else float("nan")


def grade(probs: list[list[float]], targets: list[int], loss_fn) -> dict[str, float]:
    K = len(probs[0])
    counts = np.bincount(targets, minlength=K) / len(targets)
    prior = [float(c) for c in counts]
    sys_losses = [loss_fn(p, t) for p, t in zip(probs, targets)]
    prior_losses = [loss_fn(prior, t) for t in targets]
    picks = [int(np.argmax(p)) for p in probs]
    return {
        "n": len(targets),
        "accuracy": float(np.mean([p == t for p, t in zip(picks, targets)])),
        "decision_score": decision_score(sys_losses, prior_losses),
        "mean_confidence": float(np.mean([max(p) for p in probs])),
    }


# --------------------------------------------------------------------------- #
# Jev's own numbers, recomputed from its run logs
# --------------------------------------------------------------------------- #
def jev_scores(bench: str, loss_fn) -> dict[str, float]:
    rows = []
    with open(os.path.join(JE, "runs", f"jev__{bench}__0.1.0.jsonl")) as handle:
        for line in handle:
            d = json.loads(line)
            if d.get("type") != "header":
                rows.append(d)
    by_item: dict[str, list[dict]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)

    probs, targets = [], []
    for _, rs in by_item.items():
        targets.append(rs[0]["target"])
        if bench == "pubmedqa":
            p_yes = float(np.mean([r["output"]["noul"] for r in rs]))
            probs.append([1.0 - p_yes, p_yes])
        elif bench == "helpsteer2":
            cols = [[float(r["output"]["probabilities"].get(str(i), 0.0)) for i in range(5)] for r in rs]
            probs.append([float(np.mean(c)) for c in zip(*cols)])
        else:  # banking77 (unused here)
            opts = json.load(open(os.path.join(SUITE, "banking77.json")))["options"]
            cols = [[float(r["output"]["probabilities"].get(o, 0.0)) for o in opts] for r in rs]
            probs.append([float(np.mean(c)) for c in zip(*cols)])
    return grade(probs, targets, loss_fn)


# --------------------------------------------------------------------------- #
# local model runs
# --------------------------------------------------------------------------- #
def run_local(items: list[tuple[str, int]], question: dict, cfg: jev.Config,
              to_probs, workers: int) -> tuple[list[list[float]], list[int]]:
    def worker(item: tuple[str, int]):
        state, target = item
        answer, _ = jev.answer_question(state, question, cfg)
        return to_probs(answer), target

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(worker, items))
    return [p for p, _ in results], [t for _, t in results]


def load_pubmedqa_exact() -> list[tuple[str, int]]:
    """The exact 300 states Jev ran (reconstructed from the HF revision by sha256)."""
    path = os.path.join(DATA, "pubmedqa_hf", "exact_items.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [(r["state"], int(r["target"])) for r in rows]


def load_pubmedqa(n_yes: int, n_no: int, seed: int) -> list[tuple[str, int]]:
    ori = json.load(open(os.path.join(DATA, "pubmedqa", "data", "ori_pqal.json"), encoding="utf-8"))
    yes = [v for v in ori.values() if v["final_decision"] == "yes"]
    no = [v for v in ori.values() if v["final_decision"] == "no"]
    rng = random.Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    out = []
    for v in yes[:n_yes] + no[:n_no]:
        state = f"{v['QUESTION']}\n\n" + "\n".join(v["CONTEXTS"])
        out.append((state[:6000], 1 if v["final_decision"] == "yes" else 0))
    rng.shuffle(out)
    return out


def load_helpsteer2_exact() -> list[tuple[str, int]]:
    """Reconstruct the exact 300 states from the suite (verified via state_sha256)."""
    suite = json.load(open(os.path.join(SUITE, "helpsteer2.json")))
    rows = [json.loads(l) for l in gzip.open(os.path.join(DATA, "helpsteer2_val.jsonl.gz"), "rt")]
    out = []
    for item in suite["items"]:
        row = rows[item["row_idx"]]
        state = json.dumps({"prompt": row["prompt"], "response": row["response"]},
                           ensure_ascii=False, separators=(",", ":"))
        out.append((state, int(item["target"])))
    return out


def load_banking77_aligned() -> tuple[list[tuple[str, int]], list[str]]:
    """Banking77 items whose row alignment is verified (options[target] == label_text)."""
    suite = json.load(open(os.path.join(SUITE, "banking77.json")))
    options = suite["options"]
    rows = [json.loads(l) for l in open(os.path.join(DATA, "banking77_test.jsonl"), encoding="utf-8")]
    items = []
    for item in suite["items"]:
        row = rows[item["row_idx"]]
        if row.get("label_text") == options[item["target"]]:
            items.append((row["text"][:6000], int(item["target"])))
    return items, options


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--tasks", default="pubmedqa,helpsteer2,choice")
    ap.add_argument("--choice-strategy", default="grammar", choices=["grammar", "pointwise"])
    ap.add_argument("--out", default=os.path.join(DATA, "jev_comparison.json"))
    args = ap.parse_args()

    cfg = jev.Config(llama_url=args.llama_url, model="llama-jev",
                     question_first=True, max_workers=args.workers)

    report: dict[str, Any] = {"release": "2026-09-18", "llama_url": args.llama_url}
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    # ---- noul / PubMedQA ----
    if "pubmedqa" in tasks:
        pm_suite = json.load(open(os.path.join(SUITE, "pubmedqa.json")))
        pm_q = {"type": "noul", "instructions": pm_suite["instructions"], "criteria": pm_suite["criteria"]}
        exact = os.path.join(DATA, "pubmedqa_hf", "exact_items.jsonl")
        if os.path.exists(exact):
            pm_items = load_pubmedqa_exact()
        else:
            pm_items = load_pubmedqa(186, 114, args.seed)
        pm_probs, pm_targets = run_local(pm_items, pm_q, cfg,
                                         lambda a: [1.0 - a["noul"], a["noul"]], args.workers)
        report["pubmedqa"] = {"local": grade(pm_probs, pm_targets, brier),
                              "jev": jev_scores("pubmedqa", brier)}

    # ---- score / HelpSteer2 (exact 300 items) ----
    if "helpsteer2" in tasks:
        hs_suite = json.load(open(os.path.join(SUITE, "helpsteer2.json")))
        hs_q = {"type": "score", "instructions": hs_suite["instructions"], "criteria": hs_suite["criteria"]}
        hs_items = load_helpsteer2_exact()
        hs_probs, hs_targets = run_local(
            hs_items, hs_q, cfg,
            lambda a: [a["probabilities"].get(str(i), 0.0) for i in range(5)], args.workers)
        report["helpsteer2"] = {"local": grade(hs_probs, hs_targets, rps),
                                "jev": jev_scores("helpsteer2", rps)}

    # ---- choice / Banking77 (77 options via widened single-token alphabet) ----
    if "choice" in tasks:
        bk_suite = json.load(open(os.path.join(SUITE, "banking77.json")))
        options = bk_suite["options"]
        bk_q = {"type": "choice", "instructions": bk_suite["instructions"],
                "criteria": {o: "" for o in options}}
        bk_items, _ = load_banking77_aligned()
        cfg_choice = replace(cfg, letters=ALPHABET[: len(options)], n_probs=256,
                             choice_strategy=args.choice_strategy, max_workers=1)
        bk_probs, bk_targets = run_local(
            bk_items, bk_q, cfg_choice,
            lambda a: [a["probabilities"].get(o, 0.0) for o in options], args.workers)
        report["banking77"] = {"local": grade(bk_probs, bk_targets, brier),
                               "jev": jev_scores("banking77", brier)}

    # ---- print table ----
    print(f"\nJevals release {report['release']}  |  local model: Qwen3.5-0.8B via jev_server (chat, question-first)\n")
    print(f"{'task':<12}{'system':<10}{'accuracy':>10}{'DecisionScore':>15}{'mean conf':>11}{'n':>6}")
    for task in ("pubmedqa", "helpsteer2", "banking77"):
        if task not in report:
            continue
        for system in ("local", "jev"):
            m = report[task][system]
            print(f"{task:<12}{system:<10}{m['accuracy']:>10.3f}{m['decision_score']:>15.2f}"
                  f"{m['mean_confidence']:>11.3f}{m['n']:>6}")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
