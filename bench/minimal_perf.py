#!/usr/bin/env python3
"""Minimal vs full: does the ~40-line `simple_jev.py` approach keep up?

Same items, two implementations:
  minimal  simple_jev.decide()  — raw /completion, grammar, no chat template/system prompt
  full     jev_server           — chat template + question-first (+ choice system prompt)

Run::

    python3 bench/minimal_perf.py --n 400 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _bootstrap import DATA
from regroup_choice import GROUPS, INTENT_TO_GROUP

import jev_server as jev
import simple_jev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(DATA, "banking77_test.jsonl"), encoding="utf-8")]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    groups = list(GROUPS)
    pos = {g: i for i, g in enumerate(groups)}
    choice_items = [(" ".join(r["text"].split())[:1500], pos[INTENT_TO_GROUP[r["label_text"]]])
                    for r in rows]

    imdb = [json.loads(l) for l in open(os.path.join(DATA, "imdb_test.jsonl"), encoding="utf-8")]
    half = args.n // 2
    pos_t = [r for r in imdb if r["label"] == 1][:half]
    neg_t = [r for r in imdb if r["label"] == 0][:half]
    noul_items = [(" ".join(r["text"].split())[:1500], int(r["label"])) for r in pos_t + neg_t]

    cfg = jev.Config(llama_url="http://127.0.0.1:8080", model="llama-jev", mode="chat",
                     question_first=True, max_workers=1)

    choice_instr = "Which category best describes the customer's request?"
    choice_labels = [f"{g}: {GROUPS[g][1]}" for g in groups]
    full_choice_q = {"type": "choice", "instructions": choice_instr,
                     "criteria": {g: GROUPS[g][1] for g in groups}}
    full_noul_q = {"type": "noul", "instructions": "Is this movie review positive?",
                   "criteria": {"true": "positive", "false": "negative"}}

    def run(fn, items, workers):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(fn, items))
        return np.array([ok for ok, _ in res], bool), float(np.mean([c for _, c in res])), \
            (time.perf_counter() - t0) / len(items) * 1000

    def min_choice(item):
        state, gold = item
        probs = simple_jev.decide(state, choice_instr, choice_labels)
        best = max(probs, key=lambda k: probs[k])
        return groups.index(best.split(":")[0]) == gold, max(probs.values())

    def min_noul(item):
        state, gold = item
        p = simple_jev.decide(state, "Is this movie review positive?", ["yes", "no"])["yes"]
        return int(p >= 0.5) == gold, max(p, 1 - p)

    def full_choice(item):
        state, gold = item
        a, _ = jev.answer_question(state, full_choice_q, cfg)
        p = np.array([a["probabilities"].get(g, 0.0) for g in groups])
        return int(np.argmax(p)) == gold, float(p.max())

    def full_noul(item):
        state, gold = item
        a, _ = jev.answer_question(state, full_noul_q, cfg)
        p = float(a["noul"])
        return int(p >= 0.5) == gold, max(p, 1 - p)

    print(f"n={args.n}  workers={args.workers}\n")
    print(f"{'task':<20}{'impl':<10}{'accuracy':>10}{'mean conf':>11}{'ms/item':>10}")
    for task, items, minfn, fullfn in (
        ("choice (10 groups)", choice_items, min_choice, full_choice),
        ("noul (imdb)", noul_items, min_noul, full_noul),
    ):
        for name, fn in (("minimal", minfn), ("full", fullfn)):
            ok, conf, ms = run(fn, items, args.workers)
            print(f"{task:<20}{name:<10}{ok.mean():>10.3f}{conf:>11.3f}{ms:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
