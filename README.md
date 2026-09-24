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
| model | [Qwen3.5-0.8B](https://modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF/) (base `Qwen/Qwen3.5-0.8B`) — 0.8B params, vision-language, hybrid Gated Delta Net architecture (dense, no MoE at this size) |
| GGUF | unsloth **Unsloth Dynamic 2.0**, quant **`UD-Q4_K_XL`** (~4-bit, extra precision on key layers) — **559 MB** |
| GPU | NVIDIA GeForce RTX 3070 Ti **Laptop**, 8 GB (Ampere, CC 8.6), driver 595.84 |
| CPU | AMD Ryzen 7 6800H (16 threads) |
| runtime | llama.cpp `b8818-b572d1ecd`, 4 slots, `-c 32768` (8192/slot) |

Single call per decision, `chat` mode + question-first.

### How close is llama-jev (Qwen3.5-0.8B) to Jev?

| task | Jev | best LLM | **llama-jev + Qwen3.5-0.8B** |
|---|---|---|---|
| `noul` — PubMedQA (yes/no) | 69.0 / 91.3% | Gemini 73.0 / 92.5% | **4.1 / 64.0%** |
| `score` — HelpSteer2 (0–4) | 9.2 / 41.3% | GLM 7.8 / 43.0% | **−2.6 / 31.0%** |
| `choice` — Banking77 (77) | 67.8 / 79.7% | Gemini 74.1 / 84.6% | **−8.1 / 2.5%** |
| `choice` — Banking77, pointwise | — | — | **0.1 / 21.7%** |

*(Decision Score / accuracy. All three boards are **item-exact**: HelpSteer2 300/300, PubMedQA
300/300 (reconstructed from the HF revision by `state_sha256`), Banking77 277/300 aligned.)*

**Bottom line:** the mechanics reproduce Jev, the *decisions* do not. A 0.8B model is far
below Jev and every frontier LLM — but it is **~7× faster** (61 vs 438 ms median).

### Scaling the model: 0.8B → 2B → 4B → 35B-A3B

Same wrapper, same prompts, different base models. Accuracy on the balanced calibration
sample (n=500) for the 0.8B/2B rows; the 35B rows are on the **item-exact boards** above.

| metric | 0.8B | 2B | VL-4B | **9B (hybrid)** | **35B-A3B (MoE)** |
|---|---|---|---|---|
| choice, 10 groups (described) | 0.523 | 0.724 | **0.766** | **0.757** | **0.807** (chance 0.10) |
| choice, 77-way (Banking77 full board) | 0.025 | not run | **0.527** | not run | **0.682** (chance 0.013) |
| PubMedQA `noul` acc — item-exact 300, paired w/ Jev | 0.640 | 0.680 | **0.733** | **0.787** | **0.787** |
| HelpSteer2 acc — item-exact 300, paired w/ Jev | 0.310 | 0.377 | **0.443** | **0.423** | **0.427** |
| PubMedQA Decision Score | 4.11 | −21.67 | **0.11** | **43.19** | **39.07** |
| HelpSteer2 Decision Score | −2.56 | −33.74 | **−39.61** | **−2.69** | **−1.16** |
| two-stage, end-to-end | 0.164 | 0.461 | not run | not run | not run |

*(Jev on the same boards: PubMedQA 0.913 / 69.06, HelpSteer2 0.410 / 9.28, Banking77 0.797 / 67.79.)*

**What this shows:** a 35B-A3B MoE with zero training already **matches Jev's accuracy on
HelpSteer2** (0.427 vs 0.410) and closes most of PubMedQA (78.7% vs 91.3%) — with our
prompting, no decision training. And the **dense VL-4B beats the same-size hybrid
(Qwen3.5-4B)** on both accuracy (0.766 vs 0.736) and latency (65 vs 131 ms/item) —
on llama.cpp today, dense transformers are the right pick; the hybrid arch costs
speed without buying quality at this scale.

The VL-4B also runs the **full 77-way Banking77 board** via the widened single-token
alphabet (79 labels) — accuracy 0.527, 40x chance, Decision Score +19.7: no regrouping
needed.

**The VL-4B on the item-exact boards**: PubMedQA accuracy 0.733 (best local), HelpSteer2
**0.443 — above Jev's 0.410** — and choice K=5 at **0.920** (ECE 4.4, well-calibrated).
Its raw confidences are heavily overconfident though (mean 0.95+), so its **Decision
Scores collapse** (−39.6 on HelpSteer2) until a single Platt parameter is fitted:
HelpSteer2 ECE 53.8 → 1.4, choice K=10 ECE 29.5 → 5.5 (`bench/calibrate_probs.py`).

**The 35B-A3B on the item-exact boards**: PubMedQA **0.787 / DS 39.07**, HelpSteer2
0.427 / −1.16 — and on the full 77-way Banking77 board (79-label alphabet) it reaches
**0.682 / DS 50.75**, between Mercury 2.5 and Qwen3.8 Flash. Latency: ~1.1–1.3 s per
decision with `--no-mmap` (measured: noul 0.81 s, 10-choice 1.14 s median — `--no-mmap`
is ~10% faster than mmap for a CPU-offloaded MoE, whose sparse expert access otherwise
keeps faulting pages in; `-np` does not affect single-stream latency) on this laptop
(only 10/40 layers fit in the 8 GB GPU; a desktop GPU would change this) — still slower
than Jev's ~440 ms median, because the MoE weights are CPU-offloaded on this 8 GB card.
Pointwise (per-option yes/no) was re-checked here and
**hurts at this scale**: 0.484 acc / DS 15.8 vs 0.682 / 50.75 listwise — its flat
normalised distribution (mean confidence 0.20) wastes the Brier score, and isolated
binary calls rank worse than the joint 77-way pick (sharpening recovers only
DS ≈ 18). Pointwise is a small-model crutch, not a general strategy.

**Scale buys accuracy, not calibration.** The 2B is far more accurate but its raw softmax is
wildly overconfident (mean confidence 0.92 at 0.62 accuracy), so the Decision Score *falls*
(choice K=10 DS −8 → −30). One Platt/temperature parameter (`q = sigmoid(a·logit(p)+b)`, fitted
`a ≈ 0.3`) restores calibration (ECE 29.5 → 5.5). This is the concrete version of the earlier
point: you need **scale + training/calibration**, not scale alone. Same for the VL-4B:
one Platt parameter (a ≈ 0.1–0.5) pulls its ECE back to 1.4–6.9 everywhere.

### Latency

Median / p95 per decision. Jev/LLM figures are from the `seconds` field of Jevals' run logs
(1,500 decisions each); llama-jev is measured live (`bench/latency.py`), 1 worker.

| system | PubMedQA median / p95 | Banking77 median / p95 |
|---|---|---|
| **llama-jev + Qwen3-VL-2B** | **27 / 31 ms** | **27 / 35 ms** |
| **llama-jev + Qwen3-VL-4B** | 76 / 91 ms | 62 / 85 ms |
| **llama-jev + Qwen3.5-4B (hybrid)** | 102 / 113 ms | 131 / 134 ms |
| **llama-jev + Qwen3.5-9B** | 196 / 222 ms | 262 / 274 ms |
| **llama-jev + Qwen3.5-0.8B** | 61 / 74 ms | 65 / 68 ms |
| Jev | 438 / 653 ms | 467 / 693 ms |
| Mercury 2.5 (fastest LLM) | 584 / 1037 ms | 639 / 1507 ms |
| DeepSeek V4.1 Flash | 838 / 1123 ms | 999 / 1311 ms |
| Gemini 3.8 Flash (slowest) | 1854 / 5541 ms | 1636 / 3938 ms |

- Local is **~8–16× faster than Jev** (median) and ~10–30× faster than the LLMs. Surprisingly
  the 2B VL model is ~2× faster than the 0.8B on this llama.cpp build (newer hybrid
  architecture, less optimised kernels).
- With `JEV_MAX_WORKERS=4` the per-call latency rises (queuing) but **throughput improves**:
  ~50 ms/item (0.8B), ~23 ms/item (2B).

### Vision — a capability Jev doesn't have

Jev is **text-only** ("no image or audio input at launch" per its docs). Because
`llama-jev` runs on a vision-language model (Qwen3-VL-2B + its mmproj projector), it can
decide on **images**: screenshots of tickets, photos of receipts or statements.

Verified end-to-end: render a support ticket to a PNG, send it as a multimodal user message
with the 10 category options and the grammar `root ::= [ABCDEFGHIJ]`, read the letter
probabilities. Accuracy on the 10-group choice task: **0.75–0.78** (vs 0.72 text) at
~63 ms median (1 worker) / ~52 ms/item (4 workers) — image encoding included.

The wrapper accepts an optional `image` per question or at the top level
(data URL or file path); it requires a vision model loaded with
`--mmproj`. Example:

```json
{"state": "", "image": "/tmp/ticket.png",
 "questions": {"route": {"type": "choice", "instructions": "Which category?",
                         "criteria": {"billing": "…", "technical": "…"}}}}
```

For text-only decisions on non-vision models this stays inert.

### KV-cache reuse — batching decisions on a shared state

The wrapper sends `cache_prompt: true`, so llama.cpp reuses the KV for the longest shared
prompt prefix. Measured on the 35B-A3B (sequential requests, one slot):

| workload | ordering | cache | median |
|---|---|---|---|
| one state, 6 questions | **state-first** | **ON** | **113 ms** (3.7× vs OFF) |
| one state, 6 questions | state-first | OFF | 417 ms |
| one state, 6 questions | question-first | ON | 599 ms (~1.0× vs OFF) |
| 40 states, same question | question-first | ON | 84.6 ms (overhead > savings) |
| 40 states, same question | question-first | OFF | 69.8 ms |

**Rule: put whatever the requests SHARE at the front** — the cache reuses the longest
common token prefix, so the ordering should match the workload's shared structure:
- **one state, many questions** (Jev's native shape) → state-first: the shared state sits
  at the front and is processed once; each extra question costs only its tail (~3.7× here).
- **one question, many items** (classification) → question-first: the shared question
  prefix is reused; each item costs only its state.
- The win scales with `shared_prefix / total_prompt` — a 1270-token shared state in a
  1330-token prompt → 3.7×; a 115-token shared prefix in a 220-token prompt → ~1.0×.

Same effect on the **classification shape** (one question, 60 different items, 10 described
groups — a ~450-token shared prefix, 35B-A3B):

| ordering | median | wall/item |
|---|---|---|
| **question-first** (shared prefix reused) | **83 ms** | **120 ms** |
| state-first (shared prefix recomputed per item) | 354 ms | 379 ms |

Note: the reported `tokens_evaluated` is the nominal prompt length in both cases — the
latency is the only honest signal of the reuse.

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
| **Pointwise** choice (score each option yes/no) vs one 77-way pick | 0.8B: 0.025 → **0.217**; 35B-A3B: 0.682 → **0.484** (hurts) |
| **Two-stage** (group → intent) | 0.025 → **0.164** (stage-1 group acc 0.532 caps it) |

The system prompt helps `choice` (and IMDB `noul`) but **flips PubMedQA's yes/no prior**
(with no prompt: 99% yes-recall / 7% no-recall; with it: 17% / 91%), so it is applied by
default **only to `choice`**. Override with `JEV_SYSTEM` (`""` disables).

> **These prompt findings are model-dependent — verified by re-measurement.** The big
> levers were found on **Qwen3.5-0.8B**; re-running the same sweeps on **Qwen3-VL-4B**
> showed the format-guiding system prompt and the option-layout effects **disappear**
> (all variants within noise), and the first-option bias is **absent** (option #0 picked
> 7% vs 46-80% on the 0.8B). Every number here is measured on Qwen3.5-0.8B Q4 with its
> own chat template. A different model — size, family, chat template, or quant — may
> prefer a **different** system prompt, option label style, or layout (some models may
> even do worse with the format guide). Treat these as a method and a starting point,
> not universal constants: re-measure on your own model with `bench/`.

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
JEV_QUESTION_FIRST=1 JEV_MAX_WORKERS=4 \
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

Single-question convenience (no `questions` object):
```json
{"state": "…", "question": "Which team?", "options": ["billing", "technical"]}
```

---

## Tuning

| env var | default | meaning |
|---|---|---|
| `JEV_LLAMA_URL` | `http://127.0.0.1:8080` | llama-server base URL |
| `JEV_QUESTION_FIRST` | `0` | `1` puts the question before a long state |
| `JEV_LETTERS` | `A..Z` | widen to `A-Za-z0-9` + symbols for up to ~79 single-token options |
| `JEV_MAX_WORKERS` | `1` | parallel calls — keep ≤ llama-server `-np` |
| `JEV_MODEL`, `JEV_API_KEY`, `JEV_N_PROBS`, `JEV_TIMEOUT`, `JEV_SYSTEM`, `JEV_THINKING` | | |

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
| `tests/test_jev_server.py` | 20 tests with a stub llama-server |
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
