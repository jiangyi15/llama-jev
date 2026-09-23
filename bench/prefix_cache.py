#!/usr/bin/env python3
"""KV-cache reuse benchmark: how much does a shared prompt prefix save?

Two batch shapes against one llama-server slot (sequential requests):

  A) same question, many states   (classification workload)
  B) same state,   many questions (one document, several decisions)

Each request is the chat-formatted prompt the wrapper builds, sent to the native
``/completion`` with ``cache_prompt`` on/off. ``tokens_evaluated`` in the response
is the direct measure of cache benefit: with a warm cache, only the new suffix
tokens are processed.

Run::

    python3 bench/prefix_cache.py --llama-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import numpy as np

SYSTEM = "Classify the state. Output exactly one letter (A, B, C, ...). No explanation."


def build_prompt(state: str, question: str, options: list[str], qfirst: bool,
                 system: str = SYSTEM) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options)]
    opts = ", ".join(f"{letters[i]}. {options[i]}" for i in range(len(options)))
    user = (f"{question}\nOptions: {opts}\n\n{state}\n\nAnswer with a single letter."
            if qfirst else
            f"{state}\n\n{question}\nOptions: {opts}\n\nAnswer with a single letter.")
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def complete(url: str, prompt: str, cache: bool) -> tuple[float, int]:
    body = {"prompt": prompt, "n_predict": 1, "temperature": 1.0, "top_k": 0,
            "top_p": 1.0, "min_p": 0.0, "n_probs": 5, "post_sampling_probs": True,
            "cache_prompt": cache}
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        r = json.loads(resp.read())
    return time.perf_counter() - t0, int(r.get("tokens_evaluated") or 0)


def batch(url: str, prompts: list[str], cache: bool) -> dict:
    times, evals = [], []
    for p in prompts:
        dt, ev = complete(url, p, cache)
        times.append(dt * 1000)
        evals.append(ev)
    return {"n": len(prompts), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "mean_prompt_tokens_processed": float(np.mean(evals)),
            "total_s": round(sum(times) / 1000, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--data", default="data/banking77_test.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-questions", type=int, default=6)
    ap.add_argument("--out", default="data/prefix_cache.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    states = [" ".join(r["text"].split())[:400] for r in rows[: args.n]]
    o_class = ["billing — payments, charges, refunds", "technical — bugs, access, outages",
               "account — profile, identity, closure", "other — anything else"]
    q_class = "Which category best describes the customer's request?"
    questions = ["Does this customer want a refund?", "Is this about a card payment?",
                 "Is the customer angry?", "Is this time-sensitive?",
                 "Does this involve fraud or a stolen card?", "Is this about the mobile app?"]
    long_state = ("Context passages from a biomedical abstract follow. " + ("Biomedical text. " * 300))

    url = args.llama_url.rstrip("/")
    results = []

    def record(name: str, prompts: list[str], cache: bool) -> None:
        m = batch(url, prompts, cache)
        results.append({"workload": name, "cache": cache, **m})
        print(f"  {name:<40}{cache!s:>7}  median {m['median_ms']:7.1f} ms"
              f"  prompt-tokens-processed {m['mean_prompt_tokens_processed']:7.1f}")

    print(f"A) same question, {len(states)} states\n")
    record("A question-first, cache ON",
           [build_prompt(s, q_class, o_class, True) for s in states], True)
    record("A question-first, cache OFF",
           [build_prompt(s, q_class, o_class, True) for s in states], False)
    record("A state-first,   cache ON",
           [build_prompt(s, q_class, o_class, False) for s in states], True)

    print(f"\nB) one long state, {args.n_questions} questions\n")
    record("B state-first,   cache ON",
           [build_prompt(long_state, q, ["yes", "no"], False) for q in questions], True)
    record("B state-first,   cache OFF",
           [build_prompt(long_state, q, ["yes", "no"], False) for q in questions], False)
    record("B question-first, cache ON",
           [build_prompt(long_state, q, ["yes", "no"], True) for q in questions], True)
    record("B question-first, cache OFF",
           [build_prompt(long_state, q, ["yes", "no"], True) for q in questions], False)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
