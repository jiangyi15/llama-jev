#!/usr/bin/env python3
"""Reconstruct the exact 300 PubMedQA items Jev ran, from the pinned HF revision.

Jev's suite pins `qiaojin/PubMedQA` config `pqa_labeled` at revision `9001f285…` and
records each item's `row_idx` and `state_sha256`. We fetch that revision's parquet
(via hf-mirror.com, since huggingface.co is blocked here), flatten the suite's
`state_fields` (`["question", "context.contexts"]`) to `{"question": …, "context": …}`,
and verify every item against `state_sha256` — giving an item-exact comparison.

Needs `pyarrow` (not in the default env)::

    conda run -n pytorch python bench/fetch_pubmedqa_exact.py

Writes ``data/pubmedqa_hf/exact_items.jsonl`` (state, target, item_id).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

from _bootstrap import DATA, SUITE

OUT_DIR = os.path.join(DATA, "pubmedqa_hf")
PARQUET = os.path.join(OUT_DIR, "pqa_labeled.parquet")
ITEMS = os.path.join(OUT_DIR, "exact_items.jsonl")


def main() -> int:
    import pyarrow.parquet as pq

    suite = json.load(open(os.path.join(SUITE, "pubmedqa.json")))
    revision = suite["hf_revision"]
    url = (f"https://hf-mirror.com/datasets/qiaojin/PubMedQA/resolve/{revision}"
           "/pqa_labeled/train-00000-of-00001.parquet")

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PARQUET):
        urllib.request.urlretrieve(url, PARQUET)
    rows = pq.read_table(PARQUET).to_pylist()

    def compact(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    items, ok = [], 0
    for item in suite["items"]:
        row = rows[item["row_idx"]]
        state = compact({"question": row["question"], "context": row["context"]["contexts"]})
        assert hashlib.sha256(state.encode()).hexdigest() == item["state_sha256"], item["item_id"]
        ok += 1
        items.append({"item_id": item["item_id"], "target": int(item["target"]), "state": state})

    with open(ITEMS, "w", encoding="utf-8") as handle:
        for entry in items:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"{ok}/{len(suite['items'])} items verified by sha256 -> {ITEMS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
