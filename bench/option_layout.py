#!/usr/bin/env python3
"""Does the option layout change accuracy? Inline vs one-option-per-line, and label style.

Variants control two things: how each option is written ("A. <text>") and how the
options are joined (", " inline vs "\\n" one per line).

Run::

    python3 bench/option_layout.py --n 400 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

from _bootstrap import DATA
from regroup_choice import GROUPS, INTENT_TO_GROUP

import jev_server as jev

# (per-option formatter, separator)
VARIANTS = {
    "inline A.": (lambda l, t: f"{l}. {t}" if t else f"{l}.", ", "),
    "\\n A.": (lambda l, t: f"{l}. {t}" if t else f"{l}.", "\n"),
    "\\n A)": (lambda l, t: f"{l}) {t}" if t else f"{l})", "\n"),
    "\\n (A)": (lambda l, t: f"({l}) {t}" if t else f"({l})", "\n"),
    "\\n Option A:": (lambda l, t: f"Option {l}: {t}" if t else f"Option {l}", "\n"),
}


def run(items, question, cfg, workers):
    is_noul = question["type"] == "noul"

    def work(item):
        state, gold = item
        answer, _ = jev.answer_question(state, question, cfg)
        if is_noul:
            p = float(answer["noul"])
            return int(p >= 0.5) == gold, max(p, 1 - p)
        probs = np.array([answer["probabilities"].get(k, 0.0) for k in question["criteria"]])
        return int(np.argmax(probs)) == gold, float(probs.max())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(work, items))
    return np.array([ok for ok, _ in res], bool), float(np.mean([c for _, c in res]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
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
    choice_q = {"type": "choice", "instructions": "Which category best describes the customer's request?",
                "criteria": {g: GROUPS[g][1] for g in groups}}

    imdb = [json.loads(l) for l in open(os.path.join(DATA, "imdb_test.jsonl"), encoding="utf-8")]
    half = args.n // 2
    pos_t = [r for r in imdb if r["label"] == 1][:half]
    neg_t = [r for r in imdb if r["label"] == 0][:half]
    noul_items = [(" ".join(r["text"].split())[:1500], int(r["label"])) for r in pos_t + neg_t]
    noul_q = {"type": "noul", "instructions": "Is this movie review positive?",
              "criteria": {"true": "positive", "false": "negative"}}

    cfg = jev.Config(llama_url=args.llama_url, model="llama-jev",
                     question_first=True, max_workers=1)

    original = jev._render_options
    try:
        for task, items, question in (("choice (10 groups)", choice_items, choice_q),
                                      ("noul (imdb)", noul_items, noul_q)):
            print(f"\n=== {task}  n={len(items)} ===\n")
            print(f"{'layout':<16}{'accuracy':>10}{'mean conf':>11}{'vs inline (McNemar p)':>24}")
            baseline = None
            for name, (fmt, sep) in VARIANTS.items():
                jev._render_options = lambda letters, texts, fmt=fmt, sep=sep: sep.join(
                    fmt(letters[i], texts[i]) for i in range(len(texts)))
                ok, conf = run(items, question, cfg, args.workers)
                if baseline is None:
                    baseline, extra = ok, ""
                else:
                    b = int(np.sum(baseline & ~ok)); c = int(np.sum(~baseline & ok))
                    p = stats.binomtest(c, b + c, 0.5).pvalue if (b + c) else 1.0
                    extra = f"p={p:.3f} ({c-b:+d})"
                print(f"{name:<16}{ok.mean():>10.3f}{conf:>11.3f}{extra:>24}")
    finally:
        jev._render_options = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
