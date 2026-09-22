#!/usr/bin/env python3
"""Latency comparison: local wrapper vs Jev and the LLMs on Jevals' board.

Jev/LLM latencies come from the ``seconds`` field of Jevals' per-decision run
logs; local latency is measured live against llama-server.

Run::

    python3 bench/latency.py --n 60
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _bootstrap import DATA, JE

import jev_server as jev


def jevals_latency(bench: str) -> dict[str, tuple[float, float, int]]:
    stats = {}
    for path in glob.glob(os.path.join(JE, "runs", f"*__{bench}__0.1.0.jsonl")):
        name = os.path.basename(path).split("__")[0]
        secs = []
        for line in open(path):
            row = json.loads(line)
            if row.get("type") != "header" and row.get("seconds") is not None:
                secs.append(float(row["seconds"]))
        if secs:
            stats[name] = (float(np.median(secs)), float(np.percentile(secs, 95)), len(secs))
    return stats


def local_latency(items, question, cfg, workers: int) -> tuple[float, float, float]:
    def timed(item):
        state, _ = item
        t0 = time.perf_counter()
        jev.answer_question(state, question, cfg)
        return time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        times = list(pool.map(timed, items))
    wall = time.perf_counter() - t0
    return float(np.median(times)), float(np.percentile(times, 95)), wall / len(items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    print("=== Jev / LLM latency (median / p95 seconds, from Jevals run logs) ===\n")
    for bench in ("pubmedqa", "banking77", "helpsteer2"):
        stats = jevals_latency(bench)
        if not stats:
            continue
        print(f"{bench}:")
        for name, (med, p95, n) in sorted(stats.items(), key=lambda kv: kv[1][0]):
            print(f"  {name:<24} median {med*1000:7.1f} ms   p95 {p95*1000:8.1f} ms   (n={n})")
        print()

    # --- local, measured live ---
    rows = [json.loads(l) for l in open(os.path.join(DATA, "banking77_test.jsonl"), encoding="utf-8")]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    cfg = jev.Config(llama_url=args.llama_url, model="llama-jev", mode="chat",
                     question_first=True, max_workers=1)

    short_q = {"type": "noul", "instructions": "Is this request urgent?",
               "criteria": {"true": "urgent", "false": "not urgent"}}
    short_items = [(" ".join(r["text"].split())[:1500], 0) for r in rows]

    long_q = {"type": "choice",
              "instructions": "Which category best describes the customer's request?",
              "criteria": {f"category_{i}": "A description of what belongs in this category."
                           for i in range(10)}}
    long_items = short_items

    print(f"=== local (Qwen3.5-0.8B via llama-server), n={args.n} ===\n")
    print(f"{'config':<34}{'median':>10}{'p95':>10}{'ms/item wall':>14}")
    for label, q, workers in (
        ("noul, 2 options, 1 worker", short_q, 1),
        ("choice, 10 options, 1 worker", long_q, 1),
        ("choice, 10 options, 4 workers", long_q, 4),
    ):
        med, p95, per = local_latency(long_items, q, cfg, workers)
        print(f"{label:<34}{med*1000:>9.1f}ms{p95*1000:>9.1f}ms{per*1000:>13.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
