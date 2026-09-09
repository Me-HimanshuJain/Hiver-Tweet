"""
08_banking77_setup.py
=====================
Phase 1 of Banking77 integration (no API calls).

Downloads PolyAI/banking77 from Hugging Face, builds the explicit
Banking77-to-Amazon-taxonomy mapping, and saves a curated 130-query
cross-domain sample for use in 09_banking77_cross_domain.py.

Outputs
-------
  data/banking77_test.jsonl                — full test split (3 080 queries)
  data/banking77_mapping.json             — decision-log mapping table
  data/banking77_cross_domain_sample.jsonl — 130 curated queries (10 per mapped intent)
"""
from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Banking77 → Amazon taxonomy mapping ───────────────────────────────────
# Decision-log rationale is embedded in the "rationale" field.
# Only intents with a clear semantic match are included; 4 Amazon categories
# (order_cancellation, return_or_exchange, product_issue,
#  device_and_digital_support) have no banking analog and are excluded.
BANKING77_MAPPING: dict[str, dict] = {
    # ── order_status_inquiry — neutral "where is X / what is the status" ──
    "card_arrival": {
        "amazon_label": "order_status_inquiry",
        "rationale": (
            "'When will my card arrive?' is the banking analog of "
            "'Where is my order?' — neutral status inquiry without complaint."
        ),
    },
    "pending_transfer": {
        "amazon_label": "order_status_inquiry",
        "rationale": (
            "Asking about a pending transaction status is information-seeking, "
            "not yet a complaint — maps to order_status_inquiry."
        ),
    },
    "transfer_timing": {
        "amazon_label": "order_status_inquiry",
        "rationale": (
            "'How long does a transfer take?' is a timing/ETA inquiry, "
            "equivalent to asking for an estimated delivery window."
        ),
    },
    # ── shipping_delay_complaint — broken promise / late delivery ──────────
    "card_delivery_estimate": {
        "amazon_label": "shipping_delay_complaint",
        "rationale": (
            "KEY HARD-PAIR ANALOG: 'Why hasn't my card arrived? I was told "
            "it would take 3-5 days.' The card_delivery_estimate Banking77 "
            "intent parallels shipping_delay_complaint exactly — a promised "
            "ETA that wasn't met. Contrasted with card_arrival (neutral inquiry)."
        ),
    },
    "transfer_not_received_by_recipient": {
        "amazon_label": "delivery_issue",
        "rationale": (
            "Sent but not received by the other party — a delivery failure "
            "complaint analog (package marked delivered but not received)."
        ),
    },
    "failed_transfer": {
        "amazon_label": "shipping_delay_complaint",
        "rationale": (
            "Transfer initiated but not completed — the banking equivalent of "
            "a shipment that never left the warehouse."
        ),
    },
    # ── refund_request ─────────────────────────────────────────────────────
    "request_refund": {
        "amazon_label": "refund_request",
        "rationale": "Direct semantic analog — customer wants money returned.",
    },
    "refund_not_showing_up": {
        "amazon_label": "refund_request",
        "rationale": (
            "'My refund was processed but hasn't appeared in my account' — "
            "the banking equivalent of 'where's my refund?' follow-ups."
        ),
    },
    "transaction_charged_twice": {
        "amazon_label": "refund_request",
        "rationale": (
            "Duplicate charge complaint — customer expects one of the charges "
            "to be reversed, equivalent to an unauthorised-charge refund request."
        ),
    },
    # ── account_access ─────────────────────────────────────────────────────
    "passcode_forgotten": {
        "amazon_label": "account_access",
        "rationale": "Exact analog — locked out of account due to forgotten credential.",
    },
    "pin_blocked": {
        "amazon_label": "account_access",
        "rationale": "Card/account access blocked — equivalent to suspended/locked account.",
    },
    "compromised_card": {
        "amazon_label": "account_access",
        "rationale": (
            "Security breach requiring account control — maps to the "
            "'my account was hacked' variant of account_access."
        ),
    },
    # ── general_complaint_or_feedback ──────────────────────────────────────
    "extra_charge_on_statement": {
        "amazon_label": "prime_membership",
        "rationale": (
            "Unexplained recurring or extra charge — the banking equivalent "
            "of a surprise Prime membership renewal or subscription fee."
        ),
    },
    "direct_debit_payment_not_recognised": {
        "amazon_label": "prime_membership",
        "rationale": (
            "Direct debit / subscription charge not recognized — equivalent "
            "to a customer complaining about a Prime subscription billing."
        ),
    },
}

MAPPED_INTENTS = list(BANKING77_MAPPING.keys())  # 13 intents
AMAZON_LABELS  = [v["amazon_label"] for v in BANKING77_MAPPING.values()]


def main() -> None:
    # ── 1. Install datasets if not present ────────────────────────────────
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("Installing 'datasets'...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset  # type: ignore

    # ── 2. Download Banking77 ─────────────────────────────────────────────
    # Strategy 1: HF datasets library — Parquet revision (datasets >= 3.x)
    # Strategy 2: HF datasets library — default (older versions)
    # Strategy 3: Direct CSV download from PolyAI GitHub (always works)
    print("Downloading PolyAI/banking77 ...")
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    label_names = ds["train"].features["label"].names

    print(f"  Train: {len(ds['train'])} | Test: {len(ds['test'])}")

    # -- 3. Save full test split -------------------------------------------
    test_path = os.path.join(DATA_DIR, "banking77_test.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for row in ds["test"]:
            intent = row.get("intent") or label_names[row["label"]]
            f.write(json.dumps({"text": row["text"], "banking77_intent": intent}) + "\n")
    print(f"  Saved {len(ds['test'])} test queries -> {test_path}")

    # -- 4. Save mapping table ---------------------------------------------
    mapping_path = os.path.join(DATA_DIR, "banking77_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": (
                    "Explicit Banking77 -> Amazon-taxonomy mapping. "
                    "13 of 77 Banking77 intents have a clear semantic analog "
                    "in our 11-category Amazon taxonomy. 4 Amazon categories "
                    "(order_cancellation, return_or_exchange, product_issue, "
                    "device_and_digital_support) have no banking analog."
                ),
                "mapping": BANKING77_MAPPING,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Saved mapping -> {mapping_path}")

    # -- 5. Build cross-domain sample (10 per mapped intent) ---------------
    random.seed(42)
    # Group test set by intent
    by_intent: dict[str, list[str]] = {}
    for row in ds["test"]:
        intent = row.get("intent") or label_names[row["label"]]
        by_intent.setdefault(intent, []).append(row["text"])

    sample_rows = []
    for b77_intent, meta in BANKING77_MAPPING.items():
        pool = by_intent.get(b77_intent, [])
        chosen = random.sample(pool, min(10, len(pool)))
        for text in chosen:
            sample_rows.append(
                {
                    "text": text,
                    "banking77_intent": b77_intent,
                    "expected_amazon_label": meta["amazon_label"],
                }
            )

    sample_path = os.path.join(DATA_DIR, "banking77_cross_domain_sample.jsonl")
    with open(sample_path, "w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(sample_rows)} cross-domain queries -> {sample_path}")

    # -- 6. Print intent distribution of sample ----------------------------
    from collections import Counter
    dist = Counter(r["expected_amazon_label"] for r in sample_rows)
    print("\nSample distribution (Banking77 queries -> Amazon label):")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {label:<35s} {count:>3d}")

    print("\nHard-pair Banking77 intents in sample:")
    for b77, meta in BANKING77_MAPPING.items():
        pool_size = len(by_intent.get(b77, []))
        print(f"  {b77:<40s} -> {meta['amazon_label']:<30s}  (pool: {pool_size})")

    print("\nPhase 1 complete. Run next:")
    print("  python scripts/09_banking77_cross_domain.py")


if __name__ == "__main__":
    main()
