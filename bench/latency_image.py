#!/usr/bin/env python3
"""Latency of image (vision) decisions vs text-only, on the 10-group choice task.

Renders each customer message to a PNG (matplotlib), sends it as a multimodal
user message through the wrapper's image path, and measures per-decision wall
time (median / p95 / throughput), with 1 and 4 workers.

Run::

    python3 bench/latency_image.py --llama-url http://127.0.0.1:8080 --n 40
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _bootstrap import DATA
from regroup_choice import GROUPS, INTENT_TO_GROUP

import jev_server as jev


def render_png(text: str) -> str:
    """Render text to a PNG and return it as a data URL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 1.6))
    plt.text(0.02, 0.5, text, fontsize=10, wrap=True)
    plt.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def measure(items, question, groups, cfg, workers, use_image):
    def work(item):
        gold = groups.index(item[1])
        t0 = time.perf_counter()
        if use_image:
            payload = {**question, "image": item[2]}
            answer, _ = jev.answer_question(item[0], payload, cfg)
        else:
            answer, _ = jev.answer_question(item[0], question, cfg)
        dt = time.perf_counter() - t0
        probs = np.array([answer["probabilities"].get(g, 0.0) for g in groups])
        return dt, int(np.argmax(probs)) == gold, float(probs.max())

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(work, items))
    wall = time.perf_counter() - t0
    times = [t for t, _, _ in res]
    return {
        "n": len(items),
        "accuracy": float(np.mean([ok for _, ok, _ in res])),
        "mean_confidence": float(np.mean([c for _, _, c in res])),
        "median_ms": float(np.median(times) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "wall_ms_per_item": wall / len(items) * 1000,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA, "latency_image.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(DATA, "banking77_test.jsonl"), encoding="utf-8")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]

    groups = list(GROUPS)
    question = {
        "type": "choice",
        "instructions": "Which category best describes the customer's request?",
        "criteria": {g: GROUPS[g][1] for g in groups},
    }
    cfg = jev.Config(llama_url=args.llama_url, model="llama-jev",
                     question_first=True, max_workers=1)

    items = []
    for r in rows:
        state = " ".join(r["text"].split())[:400]
        items.append((state, INTENT_TO_GROUP[r["label_text"]], render_png(state)))

    print(f"image decisions (10 groups, described), n={len(items)}, workers={args.workers}\n")
    results = {}
    for workers in (1, args.workers):
        m = measure(items, question, groups, cfg, workers, True)
        label = f"{workers} worker{'s' if workers > 1 else ''}"
        results[label] = m
        print(f"  {label:<10} median {m['median_ms']:7.1f} ms   p95 {m['p95_ms']:7.1f} ms"
              f"   throughput {m['wall_ms_per_item']:6.1f} ms/item   acc {m['accuracy']:.3f}")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
