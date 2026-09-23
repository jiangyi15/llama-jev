#!/usr/bin/env python3
"""Does regrouping Banking77's 77 intents (and describing each group) help?

The 77 fine-grained intents are mapped to 10 thematic groups, each with a short
"when to pick" description. We then score the same test items twice:

  bare        options are the bare group names
  described   options are "group: description"

and report top-1 accuracy + confidence. Run::

    python3 bench/regroup_choice.py --n 300 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import stats

from _bootstrap import DATA

import jev_server as jev

# intent -> group, with a description per group (the "criteria" text)
GROUPS: dict[str, tuple[list[str], str]] = {
    "cards": ([
        "activate_my_card", "card_about_to_expire", "card_acceptance", "card_arrival",
        "card_delivery_estimate", "card_linking", "card_not_working", "card_swallowed",
        "compromised_card", "contactless_not_working", "get_disposable_virtual_card",
        "get_physical_card", "getting_spare_card", "getting_virtual_card",
        "lost_or_stolen_card", "order_physical_card", "virtual_card_not_working",
        "disposable_card_limits",
    ], "Issues with a card itself: ordering, activating, delivery, lost/stolen, "
       "blocked, not working, contactless, or card limits."),
    "card_payments": ([
        "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "declined_card_payment", "pending_card_payment",
        "reverted_card_payment?", "apple_pay_or_google_pay", "supported_cards_and_currencies",
        "visa_or_mastercard", "direct_debit_payment_not_recognised",
    ], "A specific card payment: declined, pending, not recognised, reverted, fees on a "
       "card payment, or wallet/scheme questions."),
    "charges_fees": ([
        "extra_charge_on_statement", "transaction_charged_twice", "transfer_fee_charged",
        "cash_withdrawal_charge", "exchange_charge", "top_up_by_card_charge",
        "top_up_by_bank_transfer_charge",
    ], "An unexpected charge or fee on the statement: double charge, extra charge, or "
       "transfer/cash/exchange/top-up fees."),
    "refunds": ([
        "Refund_not_showing_up", "request_refund",
    ], "Refunds: asking for one, or a refund that has not arrived."),
    "cash_atm": ([
        "atm_support", "cash_withdrawal_not_recognised", "declined_cash_withdrawal",
        "pending_cash_withdrawal", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ], "Cash withdrawals and ATMs: declined, pending, wrong amount, or ATM support."),
    "transfers": ([
        "balance_not_updated_after_bank_transfer", "beneficiary_not_allowed",
        "cancel_transfer", "declined_transfer", "failed_transfer", "pending_transfer",
        "receiving_money", "transfer_into_account", "transfer_not_received_by_recipient",
        "transfer_timing",
    ], "Sending or receiving money: pending, failed, declined, cancelled, beneficiary, "
       "or transfer timing."),
    "top_up": ([
        "automatic_top_up", "pending_top_up", "top_up_by_cash_or_cheque", "top_up_failed",
        "top_up_limits", "top_up_reverted", "topping_up_by_card", "verify_top_up",
        "balance_not_updated_after_cheque_or_cash_deposit",
    ], "Adding money to the account: failed, pending, reverted, limits, or top-up fees."),
    "currency_exchange": ([
        "exchange_rate", "exchange_via_app", "fiat_currency_support", "country_support",
    ], "Currency and exchange: exchange rates, exchanging via the app, or supported "
       "currencies/countries."),
    "identity_verification": ([
        "edit_personal_details", "terminate_account", "unable_to_verify_identity",
        "verify_my_identity", "verify_source_of_funds", "why_verify_identity", "age_limit",
    ], "Identity and account: verifying identity, source of funds, personal details, "
       "age limit, or closing the account."),
    "security_pin": ([
        "change_pin", "passcode_forgotten", "pin_blocked", "lost_or_stolen_phone",
    ], "PIN, passcode, or a lost/stolen phone."),
}

INTENT_TO_GROUP = {intent: group for group, (intents, _) in GROUPS.items() for intent in intents}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    options = json.load(open(os.path.join(DATA, "jevals-data", "suites", "0.1.0",
                                          "banking77.json")))["options"]
    missing = [o for o in options if o not in INTENT_TO_GROUP]
    if missing:
        raise SystemExit(f"unmapped intents: {missing}")
    print(f"77 intents -> {len(GROUPS)} groups")
    print("billing/charges group 'charges_fees':",
          ", ".join(GROUPS["charges_fees"][0]), "\n")

    rows = [json.loads(l) for l in open(os.path.join(DATA, "banking77_test.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["label_text"] in INTENT_TO_GROUP]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    groups = list(GROUPS)
    pos = {g: i for i, g in enumerate(groups)}
    items = [(" ".join(r["text"].split())[:1500], pos[INTENT_TO_GROUP[r["label_text"]]]) for r in rows]

    cfg = jev.Config(llama_url=args.llama_url, model="llama-jev",
                     question_first=True, max_workers=1)

    def evaluate(described: bool) -> tuple[np.ndarray, float]:
        criteria = {g: (GROUPS[g][1] if described else "") for g in groups}
        q = {"type": "choice",
             "instructions": "Which category best describes the customer's request?",
             "criteria": criteria}

        def work(item):
            state, gold = item
            answer, _ = jev.answer_question(state, q, cfg)
            probs = np.array([answer["probabilities"].get(g, 0.0) for g in groups])
            return int(np.argmax(probs)) == gold, float(probs.max())

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            res = list(pool.map(work, items))
        return np.array([ok for ok, _ in res], bool), float(np.mean([c for _, c in res]))

    def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        p = k / n
        d = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / d
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return centre - half, centre + half

    print(f"n={len(items)}  chance={1/len(groups):.3f}\n")
    print(f"{'config':<12}{'accuracy':>10}{'95% CI':>16}{'mean conf':>11}")
    results = {}
    for described in (False, True):
        name = "described" if described else "bare"
        ok, conf = evaluate(described)
        results[name] = ok
        lo, hi = wilson(int(ok.sum()), len(ok))
        print(f"{name:<12}{ok.mean():>10.3f}{f'[{lo:.3f},{hi:.3f}]':>16}{conf:>11.3f}")

    bare, desc = results["bare"], results["described"]
    b = int(np.sum(bare & ~desc))   # bare right, described wrong
    c = int(np.sum(~bare & desc))   # bare wrong, described right
    p = stats.binomtest(c, b + c, 0.5).pvalue if (b + c) else 1.0
    print(f"\npaired (McNemar): bare-only right={b}, described-only right={c}, p={p:.3f}"
          f"  -> descriptions {'help significantly' if p < 0.05 and c > b else 'no significant gain'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
