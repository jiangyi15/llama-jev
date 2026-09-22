#!/usr/bin/env python3
# Minimal local "Jev": ask llama-server for ONE grammar-constrained token and
# read its probability distribution over the options.
#
# It builds a small ChatML prompt (Qwen-style) itself, so no /apply-template call
# is needed. This reaches ~0.42 choice / ~0.81 noul on the benchmarks — close to
# the full wrapper (~0.47 / ~0.78), which uses the model's own template.
#
#   llama-server -m model.gguf --port 8080
#   python3 simple_jev.py

import json
import urllib.request

LLAMA = "http://127.0.0.1:8080"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = "Classify the state. Output exactly one letter (A, B, C, ...). No explanation."


def build_prompt(user):
    # ChatML control tokens (Qwen). Other model families use different tokens:
    # adjust here, or call llama-server /apply-template to use the model's own.
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def decide(state, question, options):
    letters = LETTERS[:len(options)]
    choices = ", ".join(f"{letters[i]}. {options[i]}" for i in range(len(options)))
    user = (f"{question}\nOptions: {choices}\n\n{state}")  # format rule lives in SYSTEM

    body = {
        "prompt": build_prompt(user),
        "n_predict": 1,               # exactly one token
        "temperature": 1.0,           # keep the true softmax shape
        "top_k": 0,                   # don't drop small options
        "top_p": 1.0,
        "min_p": 0.0,
        "n_probs": 100,               # expose the token distribution
        "post_sampling_probs": True,  # probabilities after the grammar mask
        "grammar": f"root ::= [{letters}]",  # force exactly one option letter
    }
    request = urllib.request.Request(
        LLAMA + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())

    # top_probs = [{token, prob}, ...]; keep the option letters (default 0)
    raw = {letter: 0.0 for letter in letters}
    for entry in result["completion_probabilities"][0]["top_probs"]:
        if entry["token"] in raw:
            raw[entry["token"]] = entry["prob"]
    total = sum(raw.values()) or 1.0
    return {options[letters.index(tok)]: round(prob / total, 4)
            for tok, prob in raw.items()}


if __name__ == "__main__":
    state = "I was charged twice for September and want a refund."

    probs = decide(state, "Which team should handle this?", ["billing", "technical", "sales"])
    print("choice:", max(probs, key=lambda name: probs[name]), probs)

    print("noul:", decide(state, "Does this need an answer today?", ["yes", "no"])["yes"])

    levels = ["not severe", "mild", "serious", "critical"]
    probs = decide(state, "How severe is this?", levels)
    print("score:", round(sum(i * probs[name] for i, name in enumerate(levels)), 3), probs)
