# llama-jev — design notes & benchmarks

Adds Jev-like structured decisions to a plain `llama-server` (llama.cpp). Jev answers typed
questions (`choice` / `score` / `noul`) with calibrated probabilities instead of text; this
project reproduces the interface and mechanics on top of llama.cpp, and measures how close a
small local model gets.

---

## 1. How it works

1. Prompt: `state` + `instructions` + `A. opt1, B. opt2, …`.
2. `llama-server` `/completion`, **one token**, grammar-constrained to the option labels:
   `grammar: "root ::= [ABC]"`, `n_predict: 1`, `temperature: 1`, `top_k: 0`,
   `top_p: 1`, `min_p: 0`, `post_sampling_probs: true`, `n_probs: 100`.
3. Read the grammar-constrained, normalised `top_probs` over the labels.

Grammar forcing guarantees a bare label (no `" A"` / `"B."` guessing); `post_sampling_probs`
makes the distribution reflect the grammar. If the backend rejects grammars, retry without
one and match surface forms tolerantly.

---

## 2. Performance vs Jev

Metric: Jevals' **Decision Score** `= 100·(1 − L_system/L_prior)` (0 = base-rate prior,
100 = perfect), recomputed from the Jevals 2026-09-18 release. Our recomputation reproduces
the published board exactly for every system.

**Local setup:** [Qwen3.5-0.8B](https://modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF/)
(base `Qwen/Qwen3.5-0.8B`, vision-language, hybrid Gated Delta Net + sparse MoE), unsloth
**Unsloth Dynamic 2.0** GGUF at quant **`UD-Q4_K_XL`** (~4-bit, 559 MB); NVIDIA RTX 3070 Ti
Laptop 8 GB (Ampere) + Ryzen 7 6800H; llama.cpp `b8818`, 4 slots, `-c 8192`.
A second model was also measured: [Qwen3-VL-2B-Instruct-1M IQ4_NL](https://modelscope.cn/models/unsloth/Qwen3-VL-2B-GGUF/)
(dense transformer, 1.05 GB) and **Qwen3.6-35B-A3B UD-IQ4_NL** (18 GB MoE, 3B active,
10/40 layers on the 8 GB GPU — ~1.3–1.6 s/decision on this laptop).

### noul — PubMedQA
| system | Decision Score | accuracy |
|---|---|---|
| Gemini 3.8 Flash | 72.98 | 92.5% |
| **Jev** | **69.03** | **91.3%** |
| Qwen3.8 Flash | 62.44 | 89.7% |
| GLM-5.3 | 60.63 | 88.7% |
| Mistral Medium 3.5 | 58.03 | 88.8% |
| Mercury 2.5 | 55.66 | 87.1% |
| DeepSeek V4.1 Flash | 47.50 | 83.7% |
| **llama-jev + Qwen3.5-0.8B** | **4.11** | **64.0%** |
| **llama-jev + Qwen3-VL-4B** | **0.11** | **73.3%** |

### score — HelpSteer2 (exact same 300 items)
| system | Decision Score | accuracy |
|---|---|---|
| **Jev** | **9.21** | **41.3%** |
| GLM-5.3 | 7.78 | 43.0% |
| Gemini 3.8 Flash | 4.59 | 42.4% |
| Qwen3.8 Flash | −1.43 | 36.1% |
| **llama-jev + Qwen3.5-0.8B** | **−2.56** | **31.0%** |
| **llama-jev + Qwen3-VL-4B** | **−39.61** | **44.3%** |
| Mercury 2.5 | −5.55 | 41.7% |
| Mistral Medium 3.5 | −13.73 | 43.9% |
| DeepSeek V4.1 Flash | −19.04 | 34.7% |

### choice — Banking77 (same 277 items)
| system | Decision Score | accuracy |
|---|---|---|
| Gemini 3.8 Flash | 74.11 | 84.6% |
| **Jev** | **67.79** | **79.7%** |
| GLM-5.3 | 66.82 | 78.9% |
| DeepSeek V4.1 Flash | 63.92 | 75.9% |
| Qwen3.8 Flash | 61.78 | 76.9% |
| Mistral Medium 3.5 | 59.45 | 74.5% |
| Mercury 2.5 | 54.01 | 70.3% |
| **llama-jev + Qwen3.5-0.8B (listwise)** | **−8.14** | **2.5%** |
| **llama-jev + Qwen3.5-0.8B (pointwise)** | **0.12** | **21.7%** |
| **llama-jev + Qwen3-VL-4B (77-way, 79-label alphabet)** | **19.65** | **52.7%** |
| **llama-jev + Qwen3.6-35B-A3B (77-way, 79-label alphabet)** | **50.75** | **68.2%** |
| chance | 0 | 1.3% |

The VL-4B's 77-way run uses the widened single-token alphabet (A-Z a-z 0-9 + safe
symbols = 79 labels), so the full board runs without regrouping — 40x chance, and a
positive Decision Score where the 0.8B was below chance.

**Latency:** see the table below.

---

## 3. Latency

Median / p95 per decision. Jev/LLM figures are from the `seconds` field of Jevals' per-decision
run logs (n=1500 each); llama-jev is measured live against llama-server (`bench/latency.py`).

| system | PubMedQA | HelpSteer2 | Banking77 |
|---|---|---|---|
| **llama-jev + Qwen3-VL-2B** | **27 / 31 ms** | ~30 ms | **27 / 35 ms** |
| **llama-jev + Qwen3-VL-4B** | 76 / 91 ms | 62 / 85 ms |
| **llama-jev + Qwen3.5-4B** | 102 / 113 ms | 131 / 134 ms |
| **llama-jev + Qwen3.5-0.8B** | 61 / 74 ms | ~65 ms | 65 / 68 ms |
| Jev | 438 / 653 ms | 479 / 670 ms | 467 / 693 ms |
| Mercury 2.5 | 584 / 1037 ms | 618 / 1165 ms | 639 / 1507 ms |
| Mistral Medium 3.5 | 534 / 917 ms | 706 / 1179 ms | 937 / 1458 ms |
| DeepSeek V4.1 Flash | 838 / 1123 ms | 912 / 1203 ms | 999 / 1311 ms |
| Qwen3.8 Flash | 1088 / 3537 ms | 1422 / 2640 ms | 1713 / 3125 ms |
| Gemini 3.8 Flash | 1854 / 5541 ms | 1888 / 3980 ms | 1636 / 3938 ms |
| GLM-5.3 | 1758 / 2879 ms | 2211 / 3632 ms | 2111 / 3186 ms |

- Local is **~8–16× faster than Jev** (median) and ~10–30× faster than the frontier LLMs.
- Concurrency: `JEV_MAX_WORKERS=4` raises per-call latency (~200 ms) but improves throughput
  to **~50 ms/item (0.8B)** / **~23 ms/item (2B)**.
- The 2B VL model is faster than the 0.8B on this llama.cpp build — the Qwen3.5 hybrid
  (Gated Delta Net + MoE) kernels are newer and less optimised than the dense path.

---

## 4. Calibration

Confidence = max probability; ECE = expected calibration error (10 bins), validated against
Jev's published calibration gaps. HL = Hosmer–Lemeshow p-value.

| task | options | n | accuracy | mean conf | ECE | HL |
|---|---|---|---|---|---|---|
| PubMedQA `noul` | 2 | 500 | 0.520 | 0.646 | 13.7 | 0.0002 (over) |
| IMDB `noul` | 2 | 500 | 0.766 | 0.613 | 15.3 | <1e-4 (under) |
| HelpSteer2 `score` | 5 | 1038 | 0.309 | 0.358 | 4.9 | 0.0002 (over) |
| Banking77 `choice` | 5 | 200 | 0.520 | 0.468 | 20.4 | <1e-4 (under) |
| Banking77 `choice` | 10 | 400 | 0.307 | 0.394 | 14.0 | 0.0006 (over) |

- For **few options** the probability tracks accuracy (ECE ~5–8 pts, on par with Jev).
- Confidence is **compressed** (rarely > 0.8), so the model is often underconfident.
- **Platt scaling** `q = sigmoid(a·logit(p)+b)` (5-fold CV ECE):

| task | raw | calibrated | a | b |
|---|---|---|---|---|
| PubMedQA | 13.72 | 5.33 | 2.17 | −1.24 |
| IMDB | 15.33 | 3.85 | 4.26 | −0.42 |
| HelpSteer2 | 4.90 | 0.23 | 0.59 | −0.46 |
| Banking77 K=5 | 20.35 | 13.96 | 3.42 | 0.82 |
| Banking77 K=10 | 13.96 | 6.15 | 1.94 | −0.03 |

Monotone, so it never changes the argmax — only the confidence.

---

## 5. Findings

> **Model-dependent.** Every finding below is measured on **Qwen3.5-0.8B Q4** with its own
> chat template. Prompt-shape results (system prompt, option label style, layout, descriptions)
> can differ — even invert — for other models, sizes, chat templates, or quants. Re-measure on
> your target model with `bench/` rather than assuming these settings transfer.

1. **A format-guiding system prompt is a big lever — for `choice`.** `"Classify the state. Output exactly one letter (A, B, C, ...). No explanation."` raises choice (10 groups) 0.29 -> **0.47** (n=500, p<0.001) and IMDB noul 0.78 -> **0.84**. It is now the **default for `choice` questions only**: on PubMedQA it merely flips the model's yes/no prior (no prompt: 99% yes-recall / 7% no-recall; with it: 17% / 91%), so it is not applied to `noul`/`score`. `JEV_SYSTEM` overrides ("" disables). Keeping the format rule **only in the system prompt** (not repeated in the user message) matters: the duplicate dropped choice from 0.535 to 0.415, so the wrapper omits the user-message instruction whenever a system prompt is present.

2. **Prompt format dominates.** Controlled sentiment set (n=60):
   | format | accuracy | AUC |
   |---|---|---|
   | raw `state…results:` | 0.500 | 0.642 |
   | chat | 0.683 | 0.838 |
   | **chat, question-first** | **0.833** | **0.938** |
   | chat + system | 0.700 | 0.874 |

3. **First-option / label bias.** Many-option choice picks by *position*: option #0 chosen
   **80%** (K=20) / **46%** (K=77); the same *intent* across shuffled orders **0%**;
   accuracy ≈ chance. Grammar forcing removes `" A"` surface variants but not the token prior.

4. **Option count matters, but isn't the root cause.** top-1 by K: 0.40 (K=5), 0.15 (K=10),
   0.10 (K=20/40), 0.00 (K=77); even K=5 is only 2× chance.

5. **Pointwise beats listwise for many options.** Score each option with a binary yes/no and
   take the max: Banking77 0.025 → **0.217**. Binary beats ordinal levels (top-1 0.35 vs
   0.05–0.15 on a 20-item dev set). Ranking improves; calibration stays poor (near-uniform
   after normalisation).

6. **Regrouping is the biggest structural win.** 77 intents → 10 thematic groups
   (with the choice system prompt): bare names **0.240**, with descriptions **0.523**
   (chance 0.10; McNemar p<0.001 — with the system prompt, descriptions matter a lot).

7. **Two-stage (group → intent)** recovers full 77-label resolution: stage-1 group accuracy
   **0.532**, stage-2 given correct group **0.308**, end-to-end **0.164** (vs flat 77-way
   0.025). Error compounding caps it at stage-1 accuracy.

8. **Prompt-surface tweaks (label style, layout) don't help.** Swept `A.` vs `(A)`, `A)`,
   `[A]`, `Option A:`, `A -`, and inline vs one-option-per-line, on choice and noul (n=400
   each, paired McNemar — `bench/option_layout.py`): **no variant beats the inline
   `A. a, B. b` default on both tasks**. `\n A.` is significantly *worse* (choice −4.2 pts,
   noul −14.5 pts); `(A)`/`A)` hurt noul; `[A]` and `\n Option A:` help one task and hurt the
   other. Only the **system prompt** (choice) and **option descriptions** are consistent
   levers — bracket style and line breaks are not.

9. **Scale buys accuracy, not calibration.** Swapping the 0.8B for `Qwen3-VL-2B-Instruct-1M
   IQ4_NL` (same wrapper/prompts): choice 10-group 0.523 → **0.724**, two-stage 0.164 →
   **0.461**, IMDB noul 0.766 → **0.886**, HelpSteer2 0.309 → 0.361. But the raw softmax is
   badly overconfident (choice K=10 mean conf 0.915 at 0.620 accuracy; ECE 14.0 → 29.5), so
   Decision Score *falls* (−8 → −30). A single Platt/temperature parameter (`a ≈ 0.3`) fixes it
   (ECE 29.5 → 5.5). I.e. scale **plus** calibration/training, not scale alone.
   The **Qwen3.5-4B** hybrid lands between: choice 10-group **0.736** (vs 2B 0.724) at
   ~131 ms/item — same accuracy as the 2B dense, ~5× its latency; the hybrid arch costs
   speed on this llama.cpp build.
   The **dense Qwen3-VL-4B** (Q4_K_M) beats the same-size hybrid (Qwen3.5-4B) on both axes:
   choice 10-group **0.766** vs 0.736 at **65 vs 131 ms/item** — on llama.cpp today the
   dense transformer path is the right pick; the hybrid arch costs speed without buying
   quality at this scale.
   The **35B-A3B MoE** (`Qwen3.6-35B-A3B UD-IQ4_NL`, item-exact boards, paired) closes
   most of the accuracy gap with zero training: PubMedQA **0.787** acc / **DS 39.1**
   (Jev 0.913 / 69.1), HelpSteer2 **0.427** acc (Jev 0.410) at ~1.4 s/decision on this
   laptop — and its Decision Score is far better than the smaller models because its
   confidences are closer to calibrated (0.833 mean vs 0.787 acc).

10. **Vision: a capability Jev doesn't have.** Qwen3-VL-2B + its mmproj projector accepts image input; the same grammar single-token probability readout works on images. Rendering customer messages to PNGs and classifying them into the 10 groups gives accuracy **0.75–0.775** at ~63 ms median (image encoding included) — comparable to text. Jev is text-only ("no image or audio input at launch"), so this is a genuine extension. (`bench/latency_image.py`)

---

## 6. Implementation

Engine split into four stages (`jev_server.py`):

```
resolve_question → _pick_probs (grammar | pointwise) → shape_answer → FastAPI routes
```

- `ResolvedQuestion` dataclass; one `shape_answer` for all typed answers.
- `_pick_probs` isolates the strategy (grammar-constrained single token, or pointwise).
- HTTP layer is **FastAPI** (`create_app(cfg)` factory, `/docs`), with `BadRequest→400`,
  `BackendError→502`. Run via `python3 jev_server.py` or `uvicorn jev_server:app`.
- `_grammar()` validates `JEV_LETTERS` against a safe character set.
- `simple_jev.py` is a self-contained reference that builds a small ChatML prompt itself
  (no `/apply-template`) and performs on par with the wrapper — choice 0.42 / noul 0.87
  (`bench/minimal_perf.py`).

Layout: `jev_server.py` + `simple_jev.py` at the root, tests in `tests/`, analysis in
`bench/` (with `bench/_bootstrap.py` for shared paths).

---

## 7. Limitations

- **Model is the ceiling** — a 0.8B model is far below Jev/frontier; the API mechanics are
  correct, the signal is weak.
- **Many-option choice** needs pointwise/regroup/two-stage, and still calibrates poorly.
- **Item coverage:** HelpSteer2 and PubMedQA are **item-exact** (300/300; PubMedQA
  reconstructed from the HF revision `9001f285` by `state_sha256`, fetched via
  `hf-mirror.com`); Banking77 is paired on the 277/300 items we could align.
- **Pointwise cost** = one call per option (77× for Banking77).
- **Prompt settings are model-specific** — system prompt, label style, layout, and Platt
  parameters were all fitted/measured on Qwen3.5-0.8B Q4; a different model may prefer
  different templates, so re-measure (`bench/system_prompt.py`, `bench/option_layout.py`,
  `bench/calibrate_probs.py`).

## 8. Reproduce

```bash
llama-server -m model.gguf --port 8080 -np 4 -c 8192
python3 -m unittest discover -s tests
python3 bench/compare_with_jev.py --workers 4
python3 bench/regroup_choice.py --n 1000 --workers 4
python3 bench/two_stage_choice.py --n 1000 --workers 4
python3 bench/check_calibration.py && python3 bench/plot_calibration.py
python3 bench/latency.py --n 60
python3 bench/system_prompt.py --n 500 --workers 4
python3 bench/option_layout.py --n 400 --workers 4
python3 bench/minimal_perf.py --n 400 --workers 4
```

Data: ModelScope / GitHub; Jev suites + run logs from `github.com/Jevals/jevals-data`
(CC BY 4.0 — cite "Jevals (jevals.com), release 2026-09-18").
