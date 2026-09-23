#!/usr/bin/env python3
"""Does a system prompt that shapes the output improve decisions?

Runs the same items under several system prompts (JEV_SYSTEM) and reports
accuracy / mean confidence, plus a paired McNemar test vs no system prompt.

Run::

    python3 bench/system_prompt.py --n 400 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
from scipy import stats

from _bootstrap import DATA
from regroup_choice import GROUPS

import jev_server as jev

PROMPTS = {
    "none": "",
    "classifier": "You are a precise text classifier.",
    "single_letter": "You are a precise text classifier. Reply with a single letter and nothing else.",
    "expert_pick": "You are an expert analyst. Read the state and the options, then choose the single "
                   "best option. Reply with only its letter.",
    "format_rule": "Classify the state. Output exactly one letter (A, B, C, ...). No explanation.",
    "banking_role": "You are a helpful banking assistant that routes requests to the right category.",
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
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    args = ap.parse_args()
    chosen = {k: PROMPTS[k] for k in args.prompts.split(",") if k in PROMPTS}

    rows = [json.loads(l) for l in open(os.path.join(DATA, "banking77_test.jsonl"), encoding="utf-8")]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    groups = list(GROUPS)
    pos = {g: i for i, g in enumerate(groups)}
    from regroup_choice import INTENT_TO_GROUP
    choice_items = [(" ".join(r["text"].split())[:1500], pos[INTENT_TO_GROUP[r["label_text"]]])
                    for r in rows]

    choice_q = {"type": "choice",
                "instructions": "Which category best describes the customer's request?",
                "criteria": {g: GROUPS[g][1] for g in groups}}

    imdb_all = [json.loads(l) for l in open(os.path.join(DATA, "imdb_test.jsonl"), encoding="utf-8")]
    half = args.n // 2
    pos = [r for r in imdb_all if r["label"] == 1][:half]
    neg = [r for r in imdb_all if r["label"] == 0][:half]
    noul_items = [(" ".join(r["text"].split())[:1500], int(r["label"])) for r in pos + neg]
    noul_q = {"type": "noul", "instructions": "Is this movie review positive?",
              "criteria": {"true": "positive", "false": "negative"}}

    base = jev.Config(llama_url=args.llama_url, model="llama-jev",
                      question_first=True, max_workers=1)

    for task, items, question in (("choice (10 groups)", choice_items, choice_q),
                                  ("noul (imdb)", noul_items, noul_q)):
        print(f"\n=== {task}  n={len(items)} ===\n")
        print(f"{'system prompt':<16}{'accuracy':>10}{'mean conf':>11}{'vs none (McNemar p)':>22}")
        baseline = None
        for name, sp in chosen.items():
            cfg = replace(base, system_prompt=sp)
            ok, conf = run(items, question, cfg, args.workers)
            if baseline is None:
                baseline = ok
                extra = ""
            else:
                b = int(np.sum(baseline & ~ok))
                c = int(np.sum(~baseline & ok))
                p = stats.binomtest(c, b + c, 0.5).pvalue if (b + c) else 1.0
                extra = f"p={p:.3f} ({c-b:+d})"
            print(f"{name:<16}{ok.mean():>10.3f}{conf:>11.3f}{extra:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
