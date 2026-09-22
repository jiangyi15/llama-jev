#!/usr/bin/env python3
"""Two-stage choice for Banking77: group first, then intent within the group.

  stage 1: 10 described groups            (chance 0.10)
  stage 2: intents inside the chosen group (<= 18 options)

Reports end-to-end top-1 accuracy (vs the 77-way single call), the stage-1
group accuracy, and the conditional stage-2 accuracy when the group is right.

Run::

    python3 bench/two_stage_choice.py --n 1000 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _bootstrap import DATA
from regroup_choice import GROUPS, INTENT_TO_GROUP

import jev_server as jev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    group_q = {
        "type": "choice",
        "instructions": "Which category best describes the customer's request?",
        "criteria": {g: desc for g, (_, desc) in GROUPS.items()},
    }

    def intent_q(group: str) -> dict:
        return {
            "type": "choice",
            "instructions": f"Which specific issue within the '{group}' category "
                            "best matches the customer's request?",
            "criteria": {intent: "" for intent in GROUPS[group][0]},
        }

    rows = [json.loads(l) for l in open(os.path.join(DATA, "banking77_test.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["label_text"] in INTENT_TO_GROUP]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    items = [(" ".join(r["text"].split())[:1500], r["label_text"]) for r in rows]

    cfg = jev.Config(llama_url=args.llama_url, model="llama-jev", mode="chat",
                     question_first=True, max_workers=1)

    def work(item):
        state, gold_intent = item
        gold_group = INTENT_TO_GROUP[gold_intent]
        answer1, _ = jev.answer_question(state, group_q, cfg)
        group = answer1["choice"]
        answer2, _ = jev.answer_question(state, intent_q(group), cfg)
        return answer2["choice"] == gold_intent, group == gold_group

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        res = list(pool.map(work, items))

    end_to_end = float(np.mean([ok for ok, _ in res]))
    stage1 = float(np.mean([g for _, g in res]))
    stage2_given = float(np.mean([ok for ok, g in res if g])) if any(g for _, g in res) else 0.0

    print(f"n={len(items)}  (77-way chance = {1/77:.3f})\n")
    print(f"  stage-1 group accuracy          : {stage1:.3f}")
    print(f"  stage-2 accuracy | group correct: {stage2_given:.3f}")
    print(f"  end-to-end intent accuracy      : {end_to_end:.3f}")
    print(f"\n  single-stage 10-group baseline  : 0.267  (n=1000)")
    print(f"  single-stage 77-way baseline    : 0.025  (n=277)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
