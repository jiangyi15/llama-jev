#!/usr/bin/env python3
"""Calibration check with larger samples for a meaningful significance test.

For each task we collect per-item (confidence, correct) and report the
reliability of the confidence, ECE, and the raw data for
``check_calibration_significance.py``.

Sizes are configurable; defaults use much more data than the first pass:
  pubmedqa   noul, up to 500 (balanced yes/no)
  imdb       noul, up to 500 (balanced pos/neg)
  helpsteer2 score, up to 1038 (full validation split)
  choice5    choice, 5 intents  (all test items of the busiest classes)
  choice10   choice, 10 intents

Usage: python3 check_calibration.py --llama-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import numpy as np

from _bootstrap import DATA, SUITE

import jev_server as jev


def ece(confs, corrects, bins: int = 10) -> float:
    confs = np.asarray(confs)
    corrects = np.asarray(corrects, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / len(confs)) * abs(corrects[mask].mean() - confs[mask].mean())
    return 100.0 * total


def reliability_table(confs, corrects, bins: int = 5) -> list[dict]:
    confs = np.asarray(confs)
    corrects = np.asarray(corrects, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                     "mean_conf": round(float(confs[mask].mean()), 3),
                     "accuracy": round(float(corrects[mask].mean()), 3)})
    return rows


def run(items, question, to_probs, cfg, workers=4):
    def work(item):
        state, target = item
        answer, _ = jev.answer_question(state, question, cfg)
        return to_probs(answer), target

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, items))
    return [p for p, _ in results], [t for _, t in results]


def report(name, probs, targets, extra=""):
    confs = [max(p) for p in probs]
    corrects = [int(np.argmax(p)) == t for p, t in zip(probs, targets)]
    acc = float(np.mean(corrects))
    print(f"\n=== {name} (n={len(targets)}) {extra}===")
    print(f"  accuracy={acc:.3f}  mean_confidence={np.mean(confs):.3f}  ECE={ece(confs, corrects):.1f} pts")
    print(f"  {'bin':>8}{'n':>6}{'mean_conf':>11}{'accuracy':>10}{'gap':>8}")
    for row in reliability_table(confs, corrects):
        print(f"  {row['bin']:>8}{row['n']:>6}{row['mean_conf']:>11.3f}{row['accuracy']:>10.3f}"
              f"{row['mean_conf']-row['accuracy']:>8.3f}")
    return {"accuracy": acc, "mean_confidence": float(np.mean(confs)),
            "ece": ece(confs, corrects), "reliability": reliability_table(confs, corrects),
            "confidence": [float(c) for c in confs], "correct": [int(c) for c in corrects]}


def load_pubmedqa(n_yes: int, n_no: int, seed: int):
    ori = json.load(open(os.path.join(DATA, "pubmedqa", "data", "ori_pqal.json"), encoding="utf-8"))
    yes = [v for v in ori.values() if v["final_decision"] == "yes"]
    no = [v for v in ori.values() if v["final_decision"] == "no"]
    rng = __import__("random").Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    out = []
    for v in yes[:n_yes] + no[:n_no]:
        state = f"{v['QUESTION']}\n\n" + "\n".join(v["CONTEXTS"])
        out.append((state[:6000], 1 if v["final_decision"] == "yes" else 0))
    rng.shuffle(out)
    return out


def load_helpsteer2(n: int):
    rows = [json.loads(l) for l in gzip.open(os.path.join(DATA, "helpsteer2_val.jsonl.gz"), "rt")]
    out = []
    for r in rows[:n]:
        state = json.dumps({"prompt": r["prompt"], "response": r["response"]},
                           ensure_ascii=False, separators=(",", ":"))
        out.append((state, int(r["helpfulness"])))
    return out


def load_banking77_classes(k: int, cap_per_class: int):
    """All test items of the k busiest intents, from the full Banking77 test set."""
    rows = [json.loads(l) for l in open(os.path.join(DATA, "banking77_test.jsonl"), encoding="utf-8")]
    counts = Counter(r["label_text"] for r in rows)
    chosen = [c for c, _ in counts.most_common(k)]
    pos = {c: i for i, c in enumerate(chosen)}
    items = []
    for c in chosen:
        texts = [r["text"] for r in rows if r["label_text"] == c][:cap_per_class]
        items += [(" ".join(t.split())[:1500], pos[c]) for t in texts]
    return items, chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pubmed-n", type=int, default=500)
    ap.add_argument("--imdb-n", type=int, default=500)
    ap.add_argument("--helpsteer-n", type=int, default=1038)
    ap.add_argument("--choice-cap", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(DATA, "calibration.json"))
    args = ap.parse_args()

    cfg = jev.Config(llama_url=args.llama_url, model="llama-jev", mode="chat",
                     question_first=True, max_workers=1)
    report_data: dict[str, object] = {}

    # 1) PubMedQA noul (2 options)
    pm = json.load(open(os.path.join(SUITE, "pubmedqa.json")))
    pm_q = {"type": "noul", "instructions": pm["instructions"], "criteria": pm["criteria"]}
    half = args.pubmed_n // 2
    probs, targets = run(load_pubmedqa(min(half, 553), min(half, 254), 20260918), pm_q,
                         lambda a: [1 - a["noul"], a["noul"]], cfg, args.workers)
    report_data["pubmedqa"] = report("pubmedqa noul", probs, targets, "[Jev ECE 5.1]")

    # 2) IMDB noul (2 options)
    rows = [json.loads(l) for l in open(os.path.join(DATA, "imdb_test.jsonl"), encoding="utf-8")]
    half = args.imdb_n // 2
    pos = [r["text"] for r in rows if r["label"] == 1][:half]
    neg = [r["text"] for r in rows if r["label"] == 0][:half]
    imdb_items = [(" ".join(t.split())[:1500], 1) for t in pos] + \
                 [(" ".join(t.split())[:1500], 0) for t in neg]
    imdb_q = {"type": "noul", "instructions": "Is this movie review positive?",
              "criteria": {"true": "positive", "false": "negative"}}
    probs, targets = run(imdb_items, imdb_q, lambda a: [1 - a["noul"], a["noul"]], cfg, args.workers)
    report_data["imdb"] = report("imdb noul", probs, targets)

    # 3) HelpSteer2 score (5 levels)
    hs = json.load(open(os.path.join(SUITE, "helpsteer2.json")))
    hs_q = {"type": "score", "instructions": hs["instructions"], "criteria": hs["criteria"]}
    probs, targets = run(load_helpsteer2(args.helpsteer_n), hs_q,
                         lambda a: [a["probabilities"].get(str(i), 0.0) for i in range(5)],
                         cfg, args.workers)
    report_data["helpsteer2"] = report("helpsteer2 score", probs, targets, "[Jev ECE 19.6]")

    # 4) choice with K intents (from the full test set)
    bk_instr = json.load(open(os.path.join(SUITE, "banking77.json")))["instructions"]
    for K in (5, 10):
        items, names = load_banking77_classes(K, args.choice_cap)
        q = {"type": "choice", "instructions": bk_instr, "criteria": {n: "" for n in names}}
        probs, targets = run(items, q,
                             lambda a, names=names: [a["probabilities"].get(n, 0.0) for n in names],
                             cfg, args.workers)
        report_data[f"choice{K}"] = report(f"banking77 choice K={K}", probs, targets)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
