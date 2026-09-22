# llama-jev

Gives a plain **`llama-server`** (llama.cpp) Jev-like structured-decision abilities: the same
request shape as [Jev](https://docs.typesafe.ai/) (TypeSafe AI's "System One" model) —
`state` + typed `questions` — answered with calibrated probabilities instead of text. It runs
against whatever model your llama-server is serving; no API key, no cloud.

The interesting question is not *whether* the API works (it does), but **how close a local
model gets to Jev**. That's what this README is about.

---

## How it works

1. Build a prompt: `state` + `instructions` + `A. opt1, B. opt2, …`.
2. Ask `llama-server` for **exactly one token**, grammar-constrained to the option labels:
   `grammar: "root ::= [ABC]"`, `n_predict: 1`, `temperature: 1`, `top_k: 0`,
   `top_p: 1`, `min_p: 0`, `post_sampling_probs: true`.
3. Read the **grammar-constrained, normalised** token distribution over the labels.

Grammar forcing guarantees a bare option label (no `" A"` / `"B."` guessing); if the backend
rejects grammars it retries without one and matches surface forms tolerantly.

The core logic is ~45 lines in [`simple_jev.py`](simple_jev.py): it builds a small **ChatML
prompt itself** (no `/apply-template` call) and is competitive with the full wrapper —
choice 0.53 vs 0.53, noul **0.85** vs 0.78 (`bench/minimal_perf.py`). Adjust the control tokens
for other model families.

---

## Performance

Scored with Jevals' own metric (`Decision Score = 100·(1 − L_system/L_prior)`, 0 = base-rate
prior, 100 = perfect), recomputed from [Jevals' published release](https://jevals.com/)
(2026-09-18).

### Model & hardware

| | |
|---|---|
| model | [Qwen3.5-0.8B](https://modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF/) (base `Qwen/Qwen3.5-0.8B`) — 0.8B params, vision-language, hybrid Gated Delta Net + sparse MoE |
| GGUF | unsloth **Unsloth Dynamic 2.0**, quant **`UD-Q4_K_XL`** (~4-bit, extra precision on key layers) — **559 MB** |
| GPU | NVIDIA GeForce RTX 3070 Ti **Laptop**, 8 GB (Ampere, CC 8.6), driver 595.84 |
| CPU | AMD Ryzen 7 6800H (16 threads) |
| runtime | llama.cpp `b8818-b572d1ecd`, 4 slots, `-c 8192` |

Single call per decision, `chat` mode + question-first.

### How close is llama-jev (Qwen3.5-0.8B) to Jev?

| task | Jev | best LLM | **llama-jev + Qwen3.5-0.8B** |
|---|---|---|---|
| `noul` — PubMedQA (yes/no) | 69.0 / 91.3% | Gemini 73.0 / 92.5% | **6.4 / 62.3%** |
| `score` — HelpSteer2 (0–4) | 9.2 / 41.3% | GLM 7.8 / 43.0% | **−2.6 / 31.0%** |
| `choice` — Banking77 (77) | 67.8 / 79.7% | Gemini 74.1 / 84.6% | **−8.1 / 2.5%** |
| `choice` — Banking77, pointwise | — | — | **0.1 / 21.7%** |

*(Decision Score / accuracy. HelpSteer2 uses the exact same 300 items; Banking77 the same
277; PubMedQA same dataset + class balance.)*

**Bottom line:** the mechanics reproduce Jev, the *decisions* do not. A 0.8B model is far
below Jev and every frontier LLM — but it is **~8× faster**.

### Latency

Median / p95 per decision. Jev/LLM figures are from the `seconds` field of Jevals' run logs
(1,500 decisions each); llama-jev is measured live (`bench/latency.py`), 1 worker.

| system | PubMedQA median / p95 | Banking77 median / p95 |
|---|---|---|
| **llama-jev + Qwen3.5-0.8B** | **55 / 67 ms** | **63 / 70 ms** |
| Jev | 438 / 653 ms | 467 / 693 ms |
| Mercury 2.5 (fastest LLM) | 584 / 1037 ms | 639 / 1507 ms |
| DeepSeek V4.1 Flash | 838 / 1123 ms | 999 / 1311 ms |
| Gemini 3.8 Flash (slowest) | 1854 / 5541 ms | 1636 / 3938 ms |

- Local is **~8× faster than Jev** (median) and ~10–30× faster than the LLMs.
- With `JEV_MAX_WORKERS=4` the per-call latency rises (queuing) but **throughput improves to
  ~49 ms/item** (from ~64 ms).

### Calibration — is the probability trustworthy?

Confidence = max probability; ECE = expected calibration error (validated against Jev's
published calibration gaps).

| task | options | n | accuracy | mean conf | ECE |
|---|---|---|---|---|---|
| PubMedQA `noul` | 2 | 500 | 0.520 | 0.646 | 13.7 (over) |
| IMDB `noul` | 2 | 500 | 0.766 | 0.613 | 15.3 (under) |
| HelpSteer2 `score` | 5 | 1038 | 0.309 | 0.358 | 4.9 (over) |
| Banking77 `choice` | 5 | 200 | 0.520 | 0.468 | 20.4 (under) |
| Banking77 `choice` | 10 | 400 | 0.307 | 0.394 | 14.0 (over) |

For **few options the probability tracks accuracy** (ECE ~5 pts for score, on par with Jev);
it degrades for many-option `choice`. A simple **Platt scaling** `q = sigmoid(a·logit(p)+b)`
cuts ECE 2–20× (PubMedQA 13.7 → 5.3, IMDB 15.3 → 3.9, HelpSteer2 4.9 → 0.2) — see
`bench/calibrate_probs.py`.

### What actually moves the needle

| finding | effect |
|---|---|
| **Chat template + question-first** vs raw `state…results:` prompt | noul accuracy 0.50 → **0.83** |
| **Format-guiding system prompt** (`Output exactly one letter, no explanation.`) — now the default for `choice` | choice 0.29 → **0.47** |
| **Regroup** 77 intents → 10 described groups (with the system prompt) | 77-way 0.025 → **0.523** (10-way; chance 0.10) |
| Add per-group **descriptions** (with the system prompt) | 0.240 → **0.523** (McNemar p<0.001) |
| **Pointwise** choice (score each option yes/no) vs one 77-way pick | 0.025 → **0.217** |
| **Two-stage** (group → intent) | 0.025 → **0.164** (stage-1 group acc 0.532 caps it) |

The system prompt helps `choice` (and IMDB `noul`) but **flips PubMedQA's yes/no prior**
(with no prompt: 99% yes-recall / 7% no-recall; with it: 17% / 91%), so it is applied by
default **only to `choice`**. Override with `JEV_SYSTEM` (`""` disables).

> **These prompt findings are model-dependent.** Every number here is measured on
> **Qwen3.5-0.8B Q4** with its own chat template. A different model — size, family, chat
> template, or quant — may prefer a **different** system prompt, option label style, or
> layout (some models may even do worse with the format guide). Treat these as a method and a
> starting point, not universal constants: re-measure on your own model with `bench/`.

Two consistent failure modes on many-option tasks:
- **First-option / label bias** — with 20–77 options the model picks by *position*, not
  content: option #0 chosen 80% (K=20) / 46% (K=77) of the time, and the same *intent* is
  picked across shuffled orders 0% of the time.
- **The model is the ceiling** — layout tricks buy 1.5–3× each, but a 0.8B model lacks the
  intent understanding; Jev's 79.7% on Banking77 is out of reach.

Full detail: `SUMMARY.md`, `bench/`.

---

## Quick start

```bash
# 1. a model on llama-server (4 slots recommended)
llama-server -m model.gguf --port 8080 -np 4 -c 8192

# 2. the wrapper — chat models work best with their own template + question-first
JEV_MODE=chat JEV_QUESTION_FIRST=1 JEV_MAX_WORKERS=4 \
  python3 jev_server.py --llama-url http://127.0.0.1:8080 --port 8000

# 3. ask a decision
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for September and want a refund.",
  "questions": {
    "route": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"}},
    "urgent": {"type": "noul", "instructions": "Answer today?",
               "criteria": {"true": "time sensitive", "false": "can wait"}}
  }
}'
```

The request/response shape is **Jev's** — `state` plus typed `choice` / `score` / `noul`
questions returning probabilities. See the [Jev API docs](https://docs.typesafe.ai/) for the
schema; this wrapper mirrors it. Interactive OpenAPI is at `/docs`.

---

## Tuning

| env var | default | meaning |
|---|---|---|
| `JEV_LLAMA_URL` | `http://127.0.0.1:8080` | llama-server base URL |
| `JEV_MODE` | `raw` | `chat` uses the model's template — **use this** for instruct models |
| `JEV_QUESTION_FIRST` | `0` | `1` puts the question before a long state |
| `JEV_CHOICE_STRATEGY` | `grammar` | `pointwise` scores each option (better for many options) |
| `JEV_LETTERS` | `A..Z` | widen to `A-Za-z0-9` + symbols for up to ~79 single-token options |
| `JEV_MAX_WORKERS` | `1` | parallel calls — keep ≤ llama-server `-np` |
| `JEV_MODEL`, `JEV_HOST`, `JEV_PORT`, `JEV_API_KEY`, `JEV_N_PROBS`, `JEV_TIMEOUT`, `JEV_SYSTEM`, `JEV_THINKING`, `JEV_PROMPT_TEMPLATE`, `JEV_POINTWISE_INSTRUCTIONS` | | |

See the module docstring in `jev_server.py` for the full list.

**System prompt.** In `chat` mode a format-guiding system prompt is applied by default for
`choice` questions (choice 0.29 → 0.47) and left off for `noul`/`score` (it flips PubMedQA's
yes/no prior). Override or extend it:

```bash
JEV_SYSTEM="Classify the state. Output exactly one letter (A, B, C, ...). No explanation."
# or disable:  JEV_SYSTEM=""
```

---

## Layout

| path | purpose |
|---|---|
| `jev_server.py` | the wrapper (engine + FastAPI app) |
| `simple_jev.py` | minimal read-only reference (~40 lines) |
| `tests/test_jev_server.py` | 19 tests with a stub llama-server |
| `bench/` | benchmarks, Jev comparison, calibration (see `bench/README`-style docstrings) |
| `data/` | datasets, `jevals-data`, result JSONs |
| `SUMMARY.md` | design notes + full benchmark/calibration detail |

```bash
python3 -m unittest discover -s tests       # tests
python3 bench/compare_with_jev.py           # vs Jev's board (needs llama-server)
python3 bench/regroup_choice.py             # 77 -> 10 groups, bare vs described
python3 bench/two_stage_choice.py           # group -> intent
python3 bench/check_calibration.py          # reliability / ECE
python3 bench/latency.py                    # latency vs Jev / LLMs
```

---

## Requirements

Python 3.10+, `fastapi` + `uvicorn` + `pydantic` (HTTP layer), a running `llama-server`.
The decision engine itself is standard-library only.

## Honest scope

This reproduces Jev's **interface and mechanics** on top of llama-server, not its accuracy. Use it
for cheap, fast, offline decisions where a small model's judgement is good enough — and add a
confidence gate or recalibrate before acting on the probabilities.

All prompt/calibration results were measured on one model (Qwen3.5-0.8B Q4); **different
models may prefer different prompts, templates, or option layouts**, so re-measure before
relying on any specific setting.
