#!/usr/bin/env python3
"""Benchmark the llama-jev API on the three Jev task shapes.

Uses real labeled datasets downloaded from ModelScope into ``./data``:

============  ==========================================  =========================
task          dataset                                     metric
============  ==========================================  =========================
``choice``    Banking77 intent classification (subset)    accuracy  (baseline 1/K)
``noul``      IMDB sentiment -> "is the review positive?" accuracy @0.5, ROC-AUC
``score``     HelpSteer2 helpfulness (0-4 ordinal)        MAE, Pearson r
============  ==========================================  =========================

The local server is exercised in-process through ``jev_server.answer_question``
(the same code path the HTTP API uses), pointed at a running ``llama-server``.

Example::

    python3 benchmark_jev.py --llama-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from _bootstrap import DATA

import jev_server as jev


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_gzip_jsonl(path: str) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truncate(text: str, max_chars: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= max_chars else flat[:max_chars] + " …"


def run_items(
    items: list[Any], worker: Callable[[Any], Any], workers: int
) -> list[Any]:
    if workers <= 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, items))


def roc_auc(scores: list[float], labels: list[int]) -> float:
    scores_a, labels_a = np.asarray(scores), np.asarray(labels)
    pos, neg = scores_a[labels_a == 1], scores_a[labels_a == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def pearson(a: list[float], b: list[float]) -> float:
    a_a, b_a = np.asarray(a, float), np.asarray(b, float)
    if a_a.std() == 0 or b_a.std() == 0:
        return float("nan")
    return float(np.corrcoef(a_a, b_a)[0, 1])


# --------------------------------------------------------------------------- #
# task benchmarks
# --------------------------------------------------------------------------- #
def bench_choice(args, cfg, rng) -> dict[str, Any]:
    rows = read_jsonl(os.path.join(DATA, "banking77_test.jsonl"))
    by_class: dict[str, list[str]] = {}
    for row in rows:
        by_class.setdefault(row["label_text"], []).append(row["text"])

    classes = sorted(by_class, key=lambda c: -len(by_class[c]))[: args.choice_classes]
    labels = [c.replace("_", " ") for c in classes]
    criteria = {label: "" for label in labels}

    samples: list[tuple[str, str]] = []
    for cls, label in zip(classes, labels):
        texts = by_class[cls][:]
        rng.shuffle(texts)
        samples += [(truncate(t, args.max_chars), label) for t in texts[: args.choice_per_class]]

    question = {
        "type": "choice",
        "instructions": "Classify the customer's banking support message into exactly one intent.",
        "criteria": criteria,
    }

    def worker(sample: tuple[str, str]):
        state, gold = sample
        answer, _ = jev.answer_question(state, question, cfg)
        return answer["choice"], gold, answer["confidence"]

    t0 = time.time()
    results = run_items(samples, worker, args.workers)
    elapsed = time.time() - t0
    correct = [(p, g, c) for p, g, c in results if p == g]
    confusion: dict[str, int] = {}
    for pred, gold, _ in results:
        if pred != gold:
            confusion[f"{gold} -> {pred}"] = confusion.get(f"{gold} -> {pred}", 0) + 1
    return {
        "task": "choice",
        "dataset": f"banking77 (top {len(classes)} intents)",
        "n": len(samples),
        "accuracy": len(correct) / len(results),
        "baseline_majority": 1.0 / len(classes),
        "mean_confidence": sum(c for _, _, c in results) / len(results),
        "mean_confidence_correct": (sum(c for _, _, c in correct) / len(correct)) if correct else 0.0,
        "top_confusions": dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:5]),
        "classes": labels,
        "seconds": round(elapsed, 1),
        "ms_per_item": round(1000 * elapsed / len(samples), 1),
    }


def bench_noul(args, cfg, rng) -> dict[str, Any]:
    rows = read_jsonl(os.path.join(DATA, "imdb_test.jsonl"))
    pos = [r["text"] for r in rows if r["label"] == 1]
    neg = [r["text"] for r in rows if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    samples = [(truncate(t, args.max_chars), 1) for t in pos[: args.noul_per_class]]
    samples += [(truncate(t, args.max_chars), 0) for t in neg[: args.noul_per_class]]

    question = {
        "type": "noul",
        "instructions": "Is this movie review positive?",
        "criteria": {"true": "the review is positive", "false": "the review is negative"},
    }

    def worker(sample: tuple[str, int]):
        state, gold = sample
        answer, _ = jev.answer_question(state, question, cfg)
        return answer["noul"], gold

    t0 = time.time()
    results = run_items(samples, worker, args.workers)
    elapsed = time.time() - t0
    probs = [p for p, _ in results]
    golds = [g for _, g in results]
    predicted = [1 if p >= 0.5 else 0 for p in probs]
    return {
        "task": "noul",
        "dataset": "imdb (binary sentiment)",
        "n": len(samples),
        "accuracy": sum(p == g for p, g in zip(predicted, golds)) / len(golds),
        "roc_auc": roc_auc(probs, golds),
        "mean_p_yes": sum(probs) / len(probs),
        "mean_p_yes_positive": sum(p for p, g in results if g == 1) / max(1, len(pos[: args.noul_per_class])),
        "mean_p_yes_negative": sum(p for p, g in results if g == 0) / max(1, len(neg[: args.noul_per_class])),
        "seconds": round(elapsed, 1),
        "ms_per_item": round(1000 * elapsed / len(samples), 1),
    }


def bench_score(args, cfg, rng) -> dict[str, Any]:
    rows = read_gzip_jsonl(os.path.join(DATA, "helpsteer2_val.jsonl.gz"))
    rng.shuffle(rows)
    rows = rows[: args.score_n]

    levels = [
        "not helpful: ignores the request or gives wrong/harmful information",
        "slightly helpful: touches the request but with major gaps or errors",
        "moderately helpful: addresses the request adequately with minor issues",
        "very helpful: fully and accurately addresses the request",
        "extremely helpful: fully addresses the request with exceptional clarity and detail",
    ]
    question = {
        "type": "score",
        "instructions": "How helpful is the assistant response to the user request?",
        "criteria": levels,
    }
    samples = [
        (truncate(f"User request: {r['prompt']}\n\nAssistant response: {r['response']}", args.max_chars),
         float(r["helpfulness"]))
        for r in rows
    ]

    def worker(sample: tuple[str, float]):
        state, gold = sample
        answer, _ = jev.answer_question(state, question, cfg)
        return answer["score"], gold

    t0 = time.time()
    results = run_items(samples, worker, args.workers)
    elapsed = time.time() - t0
    preds = [p for p, _ in results]
    golds = [g for _, g in results]
    mae = float(np.mean(np.abs(np.asarray(preds) - np.asarray(golds))))
    gold_counts = {str(int(v)): int(np.sum(np.asarray(golds) == v)) for v in sorted(set(golds))}
    return {
        "task": "score",
        "dataset": "helpsteer2 (helpfulness 0-4)",
        "n": len(samples),
        "mae": mae,
        "pearson_r": pearson(preds, golds),
        "mean_pred": float(np.mean(preds)),
        "mean_gold": float(np.mean(golds)),
        "std_gold": float(np.std(golds)),
        "gold_distribution": gold_counts,
        "seconds": round(elapsed, 1),
        "ms_per_item": round(1000 * elapsed / len(samples), 1),
    }


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the llama-jev API")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    parser.add_argument("--choice-classes", type=int, default=10)
    parser.add_argument("--choice-per-class", type=int, default=8)
    parser.add_argument("--noul-per-class", type=int, default=40)
    parser.add_argument("--score-n", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--question-first", action="store_true")
    parser.add_argument("--system", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default="choice,noul,score")
    parser.add_argument("--out", default=os.path.join(DATA, "benchmark_results.json"))
    args = parser.parse_args()

    cfg = jev.Config(
        llama_url=args.llama_url,
        model="llama-jev",
        timeout=120.0,
        question_first=args.question_first,
        system_prompt=args.system,
    )
    rng = random.Random(args.seed)
    wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}

    benchmarks = {"choice": bench_choice, "noul": bench_noul, "score": bench_score}
    summary: dict[str, Any] = {"llama_url": args.llama_url, "config": vars(args), "results": []}

    for name in ("choice", "noul", "score"):
        if name not in wanted:
            continue
        print(f"\n>>> running {name} ...", flush=True)
        result = benchmarks[name](args, cfg, rng)
        summary["results"].append(result)
        print(json.dumps(result, indent=2), flush=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
