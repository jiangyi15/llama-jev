#!/usr/bin/env python3
"""llama-jev: Jev-like structured decisions for ``llama-server`` (llama.cpp).

Jev is TypeSafe AI's "System One" model: instead of generating prose it
answers *typed* questions with calibrated probabilities.  This module exposes a
small, Jev-shaped HTTP API and implements it locally by asking a
``llama-server`` to emit exactly ONE token after a prompt of the form::

    <state>
    <question> : A. <option1>, B. <option2>, ...
    results:

The generation is grammar-constrained to the declared option letters (e.g.
``root ::= [ABC]``), so the single generated token is guaranteed to be a bare
option letter.  The probability of each option is then read from the
grammar-constrained, normalised token distribution returned by llama-server's
native ``/completion`` endpoint (``post_sampling_probs``).  No chat endpoint is
required and no text is generated for the caller.

If the backend does not support grammars, the server falls back to a plain
single-token request with tolerant matching of surface variants (``" A"``,
``"B."``, ...).

Endpoints
---------
``GET  /health``               liveness + backend info
``GET  /v1/models``            minimal model listing
``POST /v1/systemone``         Jev-style decisions (alias ``/api/v1/decisions``)

Request body
------------
::

    {
      "model": "llama-jev",               # optional, echoed back
      "state": "the text to decide about",
      "questions": {
        "route":   {"type": "choice", "instructions": "...",
                    "criteria": {"billing": "Payments/refunds", "tech": "Bugs"}},
        "urgency": {"type": "noul",   "instructions": "...",
                    "criteria": {"true": "time-sensitive", "false": "can wait"}},
        "sev":     {"type": "score",  "instructions": "...",
                    "criteria": ["Low", "Medium", "High"]}
      }
    }

Convenience: if ``questions`` is omitted but ``question`` + ``options`` are
present, the body is treated as a single ``choice`` question keyed ``"answer"``.

Response body
-------------
::

    {"model": "...", "answers": {...}, "usage": {"input_tokens": N, "output_tokens": M}}

Environment
-----------
``JEV_LLAMA_URL``       base URL of llama-server        (default http://127.0.0.1:8080)
``JEV_HOST``/``JEV_PORT`` bind address for this server  (default 0.0.0.0:8000)
``JEV_MODEL``           model name echoed in responses  (default llama-jev)
``JEV_API_KEY``         if set, require ``Authorization: Bearer <key>``
``JEV_N_PROBS``         top-N token probs to request    (default 100)
``JEV_TIMEOUT``         seconds per llama.cpp request   (default 120)
``JEV_MAX_WORKERS``     parallel question calls         (default 1)
``JEV_PROMPT_TEMPLATE`` template with {state} {question} {options} {letters}
``JEV_LETTERS``         option labels (default A..Z; widen to ``A-Za-z0-9`` plus
                        safe symbols for up to ~79 single-token options)
``JEV_MODE``            ``raw`` plain completion or ``chat`` template (default raw)
``JEV_QUESTION_FIRST``  ``1`` to put the question before the state (default 0)
``JEV_SYSTEM``          chat system message. Default: a format guide for
                        ``choice`` questions; set to override ("" disables).
``JEV_THINKING``        ``1`` to let reasoning models think first (default 0)
``JEV_CHOICE_STRATEGY`` ``grammar`` one pick, or ``pointwise`` score each option
                        with a yes/no question and pick the highest (default grammar)

The HTTP layer is FastAPI (routes as decorators, OpenAPI at ``/docs``)::

    python3 jev_server.py --llama-url http://127.0.0.1:8080 --port 8000
    # or:  uvicorn jev_server:app --port 8000

Then::

    curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
      "state": "I was billed twice and want a refund.",
      "questions": {"route": {"type": "choice", "instructions": "Which team?",
                    "criteria": {"billing": "payments", "technical": "bugs"}}}
    }'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import string
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

DEFAULT_TEMPLATE = "{state}\n\n{question} : {options}\nresults:"
QUESTION_FIRST_TEMPLATE = "{question} : {options}\n\n{state}\nresults:"
# Default system prompt for chat mode (measured win); JEV_SYSTEM="" disables it.
DEFAULT_CHAT_SYSTEM = "Classify the state. Output exactly one letter (A, B, C, ...). No explanation."
DEFAULT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Characters safe inside a GBNF char class (no ']', '\', '^', '-').
SAFE_LETTERS = string.ascii_letters + string.digits + "!#$%&()*+/:;<=>?@"
# Optional llama.cpp fields that older builds reject; dropped on a 400 retry.
_OPTIONAL_COMPLETION_FIELDS = ("post_sampling_probs", "return_tokens", "cache_prompt")

# Token text is stripped of surrounding whitespace and this punctuation before
# matching it against an option letter (so " A", "A.", "A)" all match "A").
_STRIP_CHARS = " \t\r\n.,:;)]}"


class BackendError(RuntimeError):
    """Raised when llama-server cannot be reached or returns nothing usable."""


class BadRequest(ValueError):
    """Raised on invalid request / question definitions."""


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    llama_url: str = "http://127.0.0.1:8080"
    model: str = "llama-jev"
    api_key: str | None = None
    n_probs: int = 100
    timeout: float = 120.0
    max_workers: int = 1
    prompt_template: str = DEFAULT_TEMPLATE
    letters: str = DEFAULT_LETTERS
    mode: str = "raw"              # "raw" (plain /completion) or "chat"
    question_first: bool = False   # put the question before the (long) state
    system_prompt: str | None = None  # chat system message; None = built-in default
    enable_thinking: bool = False  # disable reasoning for chat models
    choice_strategy: str = "grammar"  # "grammar" (one pick) or "pointwise" (score each)
    pointwise_instructions: str = "Does this option describe what the state is about?"

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        env = os.environ
        base = cls(
            llama_url=env.get("JEV_LLAMA_URL", cls.llama_url),
            model=env.get("JEV_MODEL", cls.model),
            api_key=env.get("JEV_API_KEY") or None,
            n_probs=int(env.get("JEV_N_PROBS", cls.n_probs)),
            timeout=float(env.get("JEV_TIMEOUT", cls.timeout)),
            max_workers=int(env.get("JEV_MAX_WORKERS", cls.max_workers)),
            prompt_template=env.get("JEV_PROMPT_TEMPLATE", cls.prompt_template),
            letters=env.get("JEV_LETTERS", cls.letters),
            mode=env.get("JEV_MODE", cls.mode).lower(),
            question_first=_truthy(env.get("JEV_QUESTION_FIRST", "0")),
            system_prompt=env.get("JEV_SYSTEM"),
            enable_thinking=_truthy(env.get("JEV_THINKING", "0")),
            choice_strategy=env.get("JEV_CHOICE_STRATEGY", cls.choice_strategy).lower(),
            pointwise_instructions=env.get("JEV_POINTWISE_INSTRUCTIONS", cls.pointwise_instructions),
        )
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(base, **clean) if clean else base


# --------------------------------------------------------------------------- #
# prompt building
# --------------------------------------------------------------------------- #
def _render_option(letter: str, text: str) -> str:
    return f"{letter}. {text}" if text else f"{letter}."


def _render_options(letters: str, texts: list[str]) -> str:
    return ", ".join(_render_option(letters[i], texts[i]) for i in range(len(texts)))


def build_prompt(state: str, question: str, options: list[str], cfg: Config) -> str:
    """Render the raw single-token decision prompt for one question."""
    letters = cfg.letters[: len(options)]
    template = (QUESTION_FIRST_TEMPLATE
                if cfg.question_first and cfg.prompt_template == DEFAULT_TEMPLATE
                else cfg.prompt_template)
    return template.format(
        state=state,
        question=question,
        options=_render_options(letters, options),
        letters=", ".join(letters),
    )


def build_chat_content(state: str, question: str, options: list[str], cfg: Config,
                       instruction: str = "Answer with a single letter.") -> str:
    """Render the user message for chat models (question first helps long states)."""
    head = f"{question}\nOptions: {_render_options(cfg.letters[: len(options)], options)}"
    body = f"{head}\n\n{state}" if cfg.question_first else f"{state}\n\n{head}"
    return body + (f"\n\n{instruction}" if instruction else "")


def apply_chat_template(content: str, cfg: Config, system: str | None = None) -> str:
    """Format a user message with the model's own chat template via llama-server.

    ``system`` overrides the config's ``system_prompt``; ``None`` means "use the
    config value" (which may itself be ``None`` for no system message).
    """
    resolved = cfg.system_prompt if system is None else system
    messages: list[dict[str, str]] = []
    if resolved:
        messages.append({"role": "system", "content": resolved})
    messages.append({"role": "user", "content": content})
    payload = {
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
    }
    endpoint = cfg.llama_url.rstrip("/") + "/apply-template"
    try:
        return str(_post_json(endpoint, payload, cfg.timeout)["prompt"])
    except (urllib.error.URLError, KeyError, ValueError) as exc:
        raise BackendError(f"llama-server /apply-template failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# question normalisation -> options
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedQuestion:
    """A question normalised into prompt options and reported labels."""

    qtype: str           # "choice" | "score" | "noul"
    labels: list[str]    # values reported back to the caller
    texts: list[str]     # how each option is written into the prompt
    instructions: str


def _noul_option(description: str, word: str) -> str:
    """Option text for a noul answer.

    Jev's criteria already read ``"Yes: …"`` / ``"No: …"``; use those verbatim.
    For bare descriptions (``"positive"``) prefix the word so the yes/no mapping
    is still explicit.
    """
    desc = description.strip()
    if not desc:
        return word
    return desc if desc.lower().startswith(word) else f"{word}: {desc}"


def resolve_question(question: dict[str, Any]) -> ResolvedQuestion:
    if not isinstance(question, dict):
        raise BadRequest("each question must be an object")
    qtype = str(question.get("type", "")).lower()
    instructions = str(question.get("instructions") or question.get("question") or "").strip()
    criteria = question.get("criteria")
    if not instructions:
        raise BadRequest("question.instructions is required")

    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise BadRequest("choice.criteria must map at least two option -> description")
        labels = [str(k) for k in criteria.keys()]
        texts = [f"{k}: {criteria[k]}" if criteria[k] else str(k) for k in criteria.keys()]
        return ResolvedQuestion(qtype, labels, texts, instructions)

    if qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise BadRequest("score.criteria must be an ordered list of at least two levels")
        labels = [str(level) for level in criteria]
        return ResolvedQuestion(qtype, labels, list(labels), instructions)

    if qtype == "noul":
        true_desc = false_desc = ""
        if isinstance(criteria, dict):
            true_desc = str(criteria.get("true") or criteria.get("yes") or "")
            false_desc = str(criteria.get("false") or criteria.get("no") or "")
        elif isinstance(criteria, list) and len(criteria) == 2:
            true_desc, false_desc = str(criteria[0]), str(criteria[1])
        # Jev's criteria text ("Yes: …" / "No: …") is used verbatim; bare
        # descriptions get a "yes:"/"no:" prefix (see _noul_option).
        texts = [_noul_option(true_desc, "yes"), _noul_option(false_desc, "no")]
        return ResolvedQuestion(qtype, ["yes", "no"], texts, instructions)

    raise BadRequest(f"unknown question type: {qtype!r} (expected choice, score or noul)")


# --------------------------------------------------------------------------- #
# llama-server client
# --------------------------------------------------------------------------- #
def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_llama(
    prompt: str,
    cfg: Config,
    grammar: str | None = None,
    n_probs: int | None = None,
) -> dict[str, Any]:
    """Ask llama-server for exactly one token plus its top-N probabilities.

    When ``grammar`` is given (e.g. ``root ::= [ABC]``) generation is forced to
    a single bare option letter.  ``temperature=1`` plus a disabled ``top_k``
    keep the softmax shape intact, and ``post_sampling_probs`` makes
    llama-server return the *grammar-constrained, normalised* distribution, so
    each option letter's reported probability is directly meaningful.
    """
    endpoint = cfg.llama_url.rstrip("/") + "/completion"
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": 1,               # one token only
        "temperature": 1.0,           # keep the true softmax shape (0 => 1.0 probs)
        "top_k": 0,                   # disabled: never drop a small option
        "top_p": 1.0,
        "min_p": 0.0,
        "n_probs": n_probs or cfg.n_probs,
        "post_sampling_probs": True,  # grammar-constrained, normalised probs
        "return_tokens": True,
        "cache_prompt": True,         # reuse the shared <state> prefix
        "stream": False,
    }
    if grammar:
        payload["grammar"] = grammar

    try:
        return _post_json(endpoint, payload, cfg.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        # Older llama-server builds may not know an optional field; retry minimal.
        if exc.code == 400 and any(
            field in detail for field in ("post_sampling_probs", "return_tokens", "cache_prompt")
        ):
            for field in ("post_sampling_probs", "return_tokens", "cache_prompt"):
                payload.pop(field, None)
            try:
                return _post_json(endpoint, payload, cfg.timeout)
            except urllib.error.URLError as retry_exc:
                raise BackendError(f"llama-server request failed: {retry_exc}") from retry_exc
        raise BackendError(f"llama-server HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BackendError(f"cannot reach llama-server at {cfg.llama_url}: {exc}") from exc


# --------------------------------------------------------------------------- #
# probability extraction
# --------------------------------------------------------------------------- #
def _entry_prob(entry: dict[str, Any]) -> float:
    """Probability of a token entry, accepting either ``prob`` or ``logprob``."""
    if "prob" in entry:
        return max(0.0, float(entry["prob"]))
    logprob = entry.get("logprob")
    if logprob is None:
        return 0.0
    try:
        return math.exp(float(logprob))
    except (OverflowError, ValueError):
        return 0.0


def token_distribution(response: dict[str, Any]) -> list[tuple[str, float]]:
    """Return ``[(token_text, probability), ...]`` for the first generated token."""
    candidates = response.get("completion_probabilities") or []
    if not candidates:
        return []
    first = candidates[0]
    top = first.get("top_logprobs") or first.get("top_probs") or []
    dist = [(str(e.get("token", "")), _entry_prob(e)) for e in top]
    if not dist and first.get("token"):
        dist = [(str(first["token"]), _entry_prob(first))]
    return dist


def _match_letter(token_text: str, letter: str, strict: bool = False) -> bool:
    """Does ``token_text`` represent option ``letter``?

    ``strict`` expects a bare single character — what a grammar-forced answer
    emits.  The tolerant form additionally accepts surface variants such as
    ``" A"`` or ``"B."`` for backends where no grammar could be applied.
    """
    if strict:
        return token_text == letter
    stripped = token_text.strip().strip(_STRIP_CHARS)
    return len(stripped) == 1 and stripped.upper() == letter.upper()


def option_probabilities(
    response: dict[str, Any], n_options: int, letters: str, strict: bool = False
) -> list[float] | None:
    """Renormalised probability per option, or ``None`` if no option matched.

    With a grammar-constrained request (``strict=True``) the distribution only
    contains the allowed letters, so matching is exact and no surface variants
    leak in.
    """
    labels = letters[:n_options]
    totals = [0.0] * n_options
    for text, prob in token_distribution(response):
        for i, letter in enumerate(labels):
            if _match_letter(text, letter, strict):
                totals[i] += prob
                break

    total = sum(totals)
    if total <= 0.0:
        # Fall back to the sampled token if the distribution did not expose it.
        content = str(response.get("content", ""))
        for i, letter in enumerate(labels):
            if _match_letter(content, letter, strict):
                totals = [0.0] * n_options
                totals[i] = 1.0
                total = 1.0
                break
    if total <= 0.0:
        return None
    return [p / total for p in totals]


# --------------------------------------------------------------------------- #
# answering
# --------------------------------------------------------------------------- #
def _map(fn: Callable[[Any], Any], items: list[Any], max_workers: int) -> list[Any]:
    """Run ``fn`` over ``items``, in threads when ``max_workers`` > 1."""
    if not items:
        return []
    workers = max(1, min(max_workers, len(items)))
    if workers == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def _grammar(letters: str) -> str:
    """GBNF that forces a single token from ``letters`` (rejecting unsafe chars)."""
    unsafe = set(letters) - set(SAFE_LETTERS)
    if unsafe:
        raise BadRequest(f"JEV_LETTERS has characters unsafe for a grammar: {sorted(unsafe)}")
    return "root ::= [" + letters + "]"


def _render_prompt(state: str, q: ResolvedQuestion, cfg: Config) -> str:
    if cfg.mode == "chat":
        # A format-guiding system prompt helps choice a lot but shifts the yes/no
        # prior on noul, so default it only for choice; JEV_SYSTEM overrides.
        # When a system prompt carries the format rule, don't repeat it in the
        # user message (the duplicate measurably hurts).
        system = None
        if cfg.system_prompt is None:
            system = DEFAULT_CHAT_SYSTEM if q.qtype == "choice" else ""
        instruction = "" if system else "Answer with a single letter."
        content = build_chat_content(state, q.instructions, q.texts, cfg, instruction)
        return apply_chat_template(content, cfg, system=system)
    return build_prompt(state, q.instructions, q.texts, cfg)


def _pick_probs(state: str, q: ResolvedQuestion, cfg: Config) -> tuple[list[float], int]:
    """Option probabilities for one question, plus prompt tokens used.

    Tries a grammar-constrained single token first; if the backend cannot honour
    the grammar (or returns nothing usable) it retries without it and matches
    token surface forms tolerantly.
    """
    n = len(q.labels)
    if n > len(cfg.letters):
        raise BadRequest(f"too many options ({n}); increase JEV_LETTERS")
    prompt = _render_prompt(state, q, cfg)
    n_probs = max(cfg.n_probs, n + 16)

    probs: list[float] | None = None
    response: dict[str, Any] | None = None
    try:
        response = call_llama(prompt, cfg, grammar=_grammar(cfg.letters[:n]), n_probs=n_probs)
        probs = option_probabilities(response, n, cfg.letters, strict=True)
    except BackendError:
        probs = None
    if probs is None:
        response = call_llama(prompt, cfg, n_probs=n_probs)
        probs = option_probabilities(response, n, cfg.letters, strict=False)
    if probs is None or response is None:
        raise BackendError(
            "llama-server returned no usable token probabilities for the option letters"
        )
    return probs, int(response.get("tokens_evaluated") or 0)


def shape_answer(q: ResolvedQuestion, probs: list[float]) -> dict[str, Any]:
    """Turn a probability vector into the typed Jev answer."""
    labels, best = q.labels, max(probs)
    if q.qtype == "choice":
        return {
            "type": "choice",
            "choice": labels[probs.index(best)],
            "confidence": round(best, 6),
            "probabilities": {labels[i]: round(probs[i], 6) for i in range(len(labels))},
        }
    if q.qtype == "score":
        return {
            "type": "score",
            "score": round(sum(i * p for i, p in enumerate(probs)), 6),
            "confidence": round(best, 6),
            "legend": {str(i): labels[i] for i in range(len(labels))},
            "probabilities": {str(i): round(probs[i], 6) for i in range(len(labels))},
        }
    return {"type": "noul", "noul": round(probs[0], 6)}  # noul: P(yes), no confidence


def _pointwise_score(state: str, instructions: str, option_text: str, cfg: Config) -> tuple[float, int]:
    """P(option matches the state) via a yes/no question (few labels => low bias)."""
    sub = resolve_question({
        "type": "noul",
        "instructions": cfg.pointwise_instructions,
        "criteria": {"true": "the option matches", "false": "the option does not match"},
    })
    sub_state = f"Question: {instructions}\nCandidate option: {option_text}\n\nState:\n{state}"
    probs, tokens = _pick_probs(sub_state, sub, cfg)
    return probs[0], tokens


def answer_choice_pointwise(state: str, q: ResolvedQuestion, cfg: Config) -> tuple[dict[str, Any], int]:
    """Score every option independently, normalise, and pick the highest.

    Avoids the first-option bias of asking a small model to pick among many
    options at once: each call has only two labels (match / no match).
    """
    results = _map(lambda text: _pointwise_score(state, q.instructions, text, cfg),
                   q.texts, cfg.max_workers)
    scores = [score for score, _ in results]
    total = sum(scores)
    probs = [s / total for s in scores] if total > 0 else [1.0 / len(scores)] * len(scores)
    return shape_answer(q, probs), sum(tokens for _, tokens in results)


def answer_question(state: str, question: dict[str, Any], cfg: Config) -> tuple[dict[str, Any], int]:
    """Answer one question. Returns ``(answer, prompt_tokens_used)``."""
    q = resolve_question(question)
    if q.qtype == "choice" and cfg.choice_strategy == "pointwise":
        return answer_choice_pointwise(state, q, cfg)
    probs, tokens = _pick_probs(state, q, cfg)
    return shape_answer(q, probs), tokens


def handle_decisions(payload: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Validate a Jev-style request and answer every question."""
    if not isinstance(payload, dict):
        raise BadRequest("request body must be a JSON object")
    state = payload.get("state")
    if not isinstance(state, str) or not state.strip():
        raise BadRequest("'state' must be a non-empty string")

    questions = payload.get("questions")
    if questions is None:
        # Convenience single-question form.
        if "question" in payload and "options" in payload:
            questions = {"answer": {
                "type": "choice",
                "instructions": payload["question"],
                "criteria": {str(o): "" for o in payload["options"]},
            }}
        else:
            raise BadRequest("'questions' is required (or provide 'question' + 'options')")
    if not isinstance(questions, dict) or not questions:
        raise BadRequest("'questions' must be a non-empty object")

    def run(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any], int]:
        key, question = item
        answer, tokens = answer_question(state, question, cfg)
        return key, answer, tokens

    results = _map(run, list(questions.items()), cfg.max_workers)
    return {
        "model": str(payload.get("model") or cfg.model),
        "answers": {key: answer for key, answer, _ in results},
        "usage": {
            "input_tokens": sum(tokens for _, _, tokens in results),
            # one reported decision per question; pointwise issues N completions
            "output_tokens": len(results),
        },
    }


# --------------------------------------------------------------------------- #
# HTTP layer (FastAPI)
# --------------------------------------------------------------------------- #
def create_app(cfg: Config) -> FastAPI:
    """Build the Jev-shaped FastAPI app for a given config."""
    app = FastAPI(title="llama-jev", version="0.1",
                  description="llama-jev: Jev-like structured decisions for llama-server")

    def require_auth(authorization: str | None) -> None:
        if cfg.api_key and authorization != f"Bearer {cfg.api_key}":
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "invalid request body"})

    @app.exception_handler(BadRequest)
    async def _bad(request: Request, exc: BadRequest) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(BackendError)
    async def _backend(request: Request, exc: BackendError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "backend": cfg.llama_url, "model": cfg.model}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list",
                "data": [{"id": cfg.model, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/systemone")
    @app.post("/api/v1/decisions")
    def decisions(payload: dict[str, Any],
                  authorization: str | None = Header(default=None)) -> dict[str, Any]:
        require_auth(authorization)
        return handle_decisions(payload, cfg)

    return app


app = create_app(Config.from_env())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jev-like decisions API backed by llama-server")
    parser.add_argument("--llama-url", help="llama-server base URL")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model")
    args = parser.parse_args(argv)

    cfg = Config.from_env(llama_url=args.llama_url, model=args.model)
    host = args.host or os.environ.get("JEV_HOST", "0.0.0.0")
    port = args.port if args.port is not None else int(os.environ.get("JEV_PORT", 8000))

    import uvicorn
    print(f"jev-server on http://{host}:{port}  ->  llama-server {cfg.llama_url}", flush=True)
    uvicorn.run(create_app(cfg), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
