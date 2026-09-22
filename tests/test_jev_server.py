#!/usr/bin/env python3
"""Tests for the llama-jev API in ``jev_server.py``.

A stub ``llama-server`` returns a synthetic single-token logprob distribution
so the whole path (prompt building -> /completion call -> probability parsing
-> Jev response) can be exercised without a real model.

Run: ``python3 -m unittest -v test_jev_server``
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jev_server as jev

OPTION_RE = re.compile(r"\b([A-Z])\. ")
GRAMMAR_RE = re.compile(r"\[([A-Z]+)\]")


class StubLlama(BaseHTTPRequestHandler):
    """Emulates llama-server ``/completion`` with and without a letter grammar.

    * With a grammar it returns ``post_sampling_probs`` keyed ``prob``/``top_probs``
      over exactly the allowed letters (mimicking the real post-grammar
      normalised distribution), and a bare-letter ``content``.
    * Without a grammar it returns raw ``logprob``/``top_logprobs`` with
      leading-space token variants, like a real tokenizer.
    """

    protocol_version = "HTTP/1.1"
    # behaviour switches (set per-test)
    reject_grammar = False
    fail_grammar_probs = False
    requests_seen: list[dict] = []
    chat_seen: list[dict] = []
    chat_seen: list[dict] = []

    def log_message(self, format, *args):  # noqa: A002
        pass

    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/").endswith("/v1/chat/completions"):
            StubLlama.chat_seen.append(body)
            m = re.search(r"\[([A-Za-z]+)\]", body.get("grammar") or "")
            letters = list(m.group(1)) if m else ["A", "B"]
            top = [{"token": letter, "logprob": math.log(1.0 / (i + 1))}
                   for i, letter in enumerate(letters)]
            self._send(200, {"choices": [{
                "message": {"content": letters[0]},
                "logprobs": {"content": [{"token": letters[0], "top_logprobs": top}]}}],
                "usage": {"prompt_tokens": 10}})
            return
        if self.path.rstrip("/").endswith("/apply-template"):
            content = body["messages"][-1]["content"]
            self._send(200, {"prompt": f"<chat>{content}</chat>"})
            return
        StubLlama.requests_seen.append(body)
        prompt = body.get("prompt", "")
        grammar = body.get("grammar")

        if grammar and StubLlama.reject_grammar:
            self._send(400, {"error": {"message": "unsupported field: grammar"}})
            return

        if grammar:
            match = GRAMMAR_RE.search(grammar)
            letters = list(match.group(1)) if match else []
        else:
            letters = sorted(set(OPTION_RE.findall(prompt)))

        weights = {letter: 0.6**i for i, letter in enumerate(letters)}
        total = sum(weights.values()) or 1.0

        if grammar and StubLlama.fail_grammar_probs:
            self._send(200, {"content": "", "tokens_evaluated": 7,
                             "completion_probabilities": []})
            return

        if grammar and body.get("post_sampling_probs"):
            top = [{"id": ord(letter), "token": letter, "prob": w / total}
                   for letter, w in weights.items()]
            top.append({"id": 0, "token": "\n", "prob": 0.0})  # grammar-masked junk
            content = letters[0] if letters else ""
            response = {
                "content": content,
                "tokens_evaluated": 42,
                "tokens_predicted": 1,
                "completion_probabilities": [
                    {"id": ord(content or "A"), "token": content, "prob": weights[content] / total,
                     "top_probs": top}
                ] if letters else [],
            }
        else:  # raw, no grammar: leading-space token variants
            top = [{"id": ord(letter), "token": " " + letter, "logprob": math.log(w / total)}
                   for letter, w in weights.items()]
            first = letters[0] if letters else ""
            response = {
                "content": first,
                "tokens_evaluated": 42,
                "tokens_predicted": 1,
                "completion_probabilities": [
                    {"id": ord(first), "token": first, "logprob": math.log(weights[first] / total),
                     "top_logprobs": top}
                ] if letters else [],
            }
        self._send(200, response)


def start_server(handler_cls, **attrs):
    for key, value in attrs.items():
        setattr(handler_cls, key, value)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


class JevUnitTests(unittest.TestCase):
    def setUp(self):
        StubLlama.requests_seen = []
        StubLlama.reject_grammar = False
        StubLlama.fail_grammar_probs = False
        StubLlama.chat_seen = []
        self.llama, _ = start_server(StubLlama)
        self.addCleanup(self.llama.shutdown)
        self.cfg = jev.Config(
            llama_url=f"http://127.0.0.1:{self.llama.server_port}",
            model="test-jev",
        )

    # -- prompt building --------------------------------------------------- #
    def test_prompt_matches_requested_format(self):
        prompt = jev.build_prompt("hello world", "Which team?", ["billing: pay", "tech: bug"], self.cfg)
        self.assertIn("hello world", prompt)
        self.assertIn("Which team? : A. billing: pay, B. tech: bug", prompt)
        self.assertTrue(prompt.rstrip().endswith("results:"))

    # -- choice ------------------------------------------------------------ #
    def test_choice_renormalises_and_picks_argmax(self):
        result = jev.handle_decisions(
            {
                "state": "charged twice",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"billing": "payments", "technical": "bugs", "sales": "new"},
                    }
                },
            },
            self.cfg,
        )
        answer = result["answers"]["route"]
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(answer["choice"], "billing")
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=5)
        self.assertAlmostEqual(answer["probabilities"]["billing"], 1 / 1.96, places=5)
        self.assertAlmostEqual(answer["confidence"], 1 / 1.96, places=5)
        self.assertEqual(result["usage"]["output_tokens"], 1)

    # -- noul -------------------------------------------------------------- #
    def test_noul_returns_probability_of_yes(self):
        result = jev.handle_decisions(
            {
                "state": "text",
                "questions": {
                    "on_topic": {
                        "type": "noul",
                        "instructions": "Is this about our product?",
                        "criteria": {"true": "yes", "false": "no"},
                    }
                },
            },
            self.cfg,
        )
        answer = result["answers"]["on_topic"]
        self.assertEqual(answer, {"type": "noul", "noul": 0.625})

    # -- score ------------------------------------------------------------- #
    def test_score_is_weighted_mean(self):
        result = jev.handle_decisions(
            {
                "state": "text",
                "questions": {
                    "severity": {
                        "type": "score",
                        "instructions": "How severe?",
                        "criteria": ["Low", "Medium", "High"],
                    }
                },
            },
            self.cfg,
        )
        answer = result["answers"]["severity"]
        self.assertEqual(answer["legend"], {"0": "Low", "1": "Medium", "2": "High"})
        self.assertListEqual(sorted(answer["probabilities"]), ["0", "1", "2"])
        # weights 1, 0.6, 0.36 -> (0*1 + 1*0.6 + 2*0.36) / 1.96
        self.assertAlmostEqual(answer["score"], 1.32 / 1.96, places=5)

    # -- grammar forcing / fallback --------------------------------------- #
    def test_forces_single_letter_grammar(self):
        jev.handle_decisions(
            {
                "state": "text",
                "questions": {"q": {"type": "choice", "instructions": "Pick",
                                    "criteria": {"one": "", "two": "", "three": ""}}},
            },
            self.cfg,
        )
        self.assertTrue(StubLlama.requests_seen)
        for req in StubLlama.requests_seen:
            self.assertEqual(req.get("grammar"), "root ::= [ABC]")
            self.assertEqual(req.get("n_predict"), 1)
            self.assertTrue(req.get("post_sampling_probs"))
            self.assertEqual(req.get("temperature"), 1.0)
            self.assertEqual(req.get("top_k"), 0)

    def test_falls_back_when_grammar_rejected(self):
        StubLlama.reject_grammar = True
        result = jev.handle_decisions(
            {
                "state": "text",
                "questions": {"q": {"type": "choice", "instructions": "Pick",
                                    "criteria": {"one": "", "two": ""}}},
            },
            self.cfg,
        )
        self.assertEqual(result["answers"]["q"]["choice"], "one")
        self.assertIn("grammar", StubLlama.requests_seen[0])
        self.assertNotIn("grammar", StubLlama.requests_seen[-1])

    def test_falls_back_when_grammar_probs_unusable(self):
        StubLlama.fail_grammar_probs = True
        result = jev.handle_decisions(
            {
                "state": "text",
                "questions": {"q": {"type": "choice", "instructions": "Pick",
                                    "criteria": {"one": "", "two": ""}}},
            },
            self.cfg,
        )
        self.assertEqual(result["answers"]["q"]["choice"], "one")
        self.assertNotIn("grammar", StubLlama.requests_seen[-1])

    def test_question_first_template_puts_question_first(self):
        cfg = jev.Config(question_first=True)
        prompt = jev.build_prompt("STATE", "Q?", ["a", "b"], cfg)
        self.assertTrue(prompt.startswith("Q? : A. a, B. b"))
        self.assertIn("STATE", prompt)

    def test_chat_mode_uses_model_template(self):
        cfg = jev.Config(
            llama_url=self.cfg.llama_url, model="test-jev",
            mode="chat", question_first=True,
        )
        result = jev.handle_decisions(
            {
                "state": "state text",
                "questions": {"q": {"type": "choice", "instructions": "Pick",
                                    "criteria": {"one": "", "two": ""}}},
            },
            cfg,
        )
        self.assertEqual(result["answers"]["q"]["choice"], "one")
        # the /completion prompt is the model-templated content, grammar still forced
        self.assertIn("<chat>", StubLlama.requests_seen[0]["prompt"])
        self.assertEqual(StubLlama.requests_seen[0]["grammar"], "root ::= [AB]")

    def test_image_requires_chat_mode(self):
        cfg = jev.Config(llama_url=self.cfg.llama_url, model="test-jev", mode="raw")
        with self.assertRaises(jev.BadRequest):
            jev.handle_decisions(
                {"state": "x", "image": "/tmp/ticket.png",
                 "questions": {"q": {"type": "noul", "instructions": "urgent?",
                                     "criteria": {"true": "", "false": ""}}}},
                cfg)

    def test_image_propagates_to_messages(self):
        cfg = jev.Config(llama_url=self.cfg.llama_url, model="test-jev",
                         mode="chat", question_first=True)
        result = jev.handle_decisions(
            {"state": "state text", "image": "data:image/png;base64,AAAA",
             "questions": {"q": {"type": "noul", "instructions": "Is it urgent?",
                                 "criteria": {"true": "urgent", "false": "not"}}}},
            cfg,
        )
        self.assertAlmostEqual(result["answers"]["q"]["noul"], 2 / 3, places=5)
        # the multimodal call went to the chat endpoint with image + text parts
        chat_calls = [r for r in StubLlama.requests_seen
                      if isinstance(r.get("messages"), list)]
        self.assertEqual(len(StubLlama.chat_seen), 1)
        parts = StubLlama.chat_seen[0]["messages"][-1]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image_url"])

    def test_image_data_url_from_file(self):
        import tempfile
        path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        url = jev._image_data_url(path)
        self.assertTrue(url.startswith("data:image/png;base64,"))
        os.remove(path)

    def test_pointwise_choice_scores_each_option(self):
        cfg = jev.Config(llama_url=self.cfg.llama_url, model="test-jev",
                         choice_strategy="pointwise", max_workers=1)
        result = jev.handle_decisions(
            {
                "state": "text",
                "questions": {"q": {"type": "choice", "instructions": "Pick",
                                    "criteria": {"one": "", "two": "", "three": ""}}},
            },
            cfg,
        )
        answer = result["answers"]["q"]
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(set(answer["probabilities"]), {"one", "two", "three"})
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=5)
        # exactly one yes/no completion per option
        self.assertEqual(len(StubLlama.requests_seen), 3)
        for req in StubLlama.requests_seen:
            self.assertEqual(req.get("grammar"), "root ::= [AB]")

    # -- validation -------------------------------------------------------- #
    def test_bad_requests(self):
        with self.assertRaises(jev.BadRequest):
            jev.handle_decisions({"questions": {}}, self.cfg)  # no state
        with self.assertRaises(jev.BadRequest):
            jev.handle_decisions({"state": "x", "questions": {}}, self.cfg)  # empty questions
        with self.assertRaises(jev.BadRequest):
            jev.handle_decisions(
                {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?",
                                                   "criteria": {"only_one": ""}}}},
                self.cfg,
            )
        with self.assertRaises(jev.BadRequest):
            jev.handle_decisions(
                {"state": "x", "questions": {"q": {"type": "wat", "instructions": "?"}}},
                self.cfg,
            )

    # -- parsing helpers --------------------------------------------------- #
    def test_strict_matching_ignores_surface_variants(self):
        resp = {"completion_probabilities": [{"token": "A", "top_probs": [
            {"token": "A", "prob": 0.4},
            {"token": " A", "prob": 0.5},   # grammar-masked variant, must be ignored
            {"token": "B", "prob": 0.1},
        ]}]}
        probs = jev.option_probabilities(resp, 2, "AB", strict=True)
        self.assertIsNotNone(probs)
        assert probs is not None
        self.assertAlmostEqual(probs[0], 0.8, places=6)
        self.assertAlmostEqual(probs[1], 0.2, places=6)

    def test_parses_prob_and_logprob_and_bytes(self):
        resp = {
            "completion_probabilities": [
                {"token": "A", "prob": 0.9, "top_probs": [
                    {"token": "A", "prob": 0.9},
                    {"token": "B", "prob": 0.1},
                ]}
            ]
        }
        probs = jev.option_probabilities(resp, 2, "AB")
        self.assertIsNotNone(probs)
        assert probs is not None
        self.assertAlmostEqual(probs[0], 0.9, places=6)
        self.assertAlmostEqual(probs[1], 0.1, places=6)

        resp_log = {
            "completion_probabilities": [
                {"token": "A", "logprob": math.log(0.3), "top_logprobs": [
                    {"token": " A", "logprob": math.log(0.3)},
                    {"token": "B.", "logprob": math.log(0.7)},
                ]}
            ]
        }
        probs = jev.option_probabilities(resp_log, 2, "AB")
        self.assertIsNotNone(probs)
        assert probs is not None
        self.assertAlmostEqual(probs[0], 0.3, places=6)
        self.assertAlmostEqual(probs[1], 0.7, places=6)

    def test_unmatched_distribution_falls_back_to_content(self):
        resp = {"content": "B", "completion_probabilities": [
            {"token": "Z", "logprob": math.log(0.5), "top_logprobs": [
                {"token": "Z", "logprob": math.log(0.5)},
            ]}
        ]}
        probs = jev.option_probabilities(resp, 2, "AB")
        self.assertIsNotNone(probs)
        assert probs is not None
        self.assertAlmostEqual(probs[0], 0.0, places=6)
        self.assertAlmostEqual(probs[1], 1.0, places=6)


class JevHttpTests(unittest.TestCase):
    def setUp(self):
        StubLlama.requests_seen = []
        StubLlama.reject_grammar = False
        StubLlama.fail_grammar_probs = False
        StubLlama.chat_seen = []
        self.llama, _ = start_server(StubLlama)
        self.addCleanup(self.llama.shutdown)
        self.cfg = jev.Config(
            llama_url=f"http://127.0.0.1:{self.llama.server_port}",
            model="test-jev",
            api_key="secret",
        )
        self.client = TestClient(jev.create_app(self.cfg))

    def _post(self, path, body, key="secret"):
        resp = self.client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
        return resp.status_code, resp.json()

    def test_health_and_models(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")
        resp = self.client.get("/v1/models")
        self.assertEqual(resp.json()["data"][0]["id"], "test-jev")

    def test_auth_is_enforced(self):
        status, _ = self._post("/v1/systemone", {"state": "x", "questions": {}}, key="wrong")
        self.assertEqual(status, 401)

    def test_end_to_end_choice(self):
        status, body = self._post(
            "/v1/systemone",
            {
                "state": "I was charged twice.",
                "questions": {
                    "route": {"type": "choice", "instructions": "Which team?",
                              "criteria": {"billing": "payments", "technical": "bugs"}},
                },
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "test-jev")
        self.assertEqual(body["answers"]["route"]["choice"], "billing")

    def test_single_question_convenience_form(self):
        status, body = self._post(
            "/api/v1/decisions",
            {"state": "x", "question": "Pick one", "options": ["alpha", "beta"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["answers"]["answer"]["type"], "choice")

    def test_invalid_json_is_400(self):
        resp = self.client.post(
            "/v1/systemone",
            content=b"{not json",
            headers={"Content-Type": "application/json", "Authorization": "Bearer secret"},
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
