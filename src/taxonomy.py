"""
taxonomy.py — Shared intent definitions, labels, and few-shot examples.

This is the single source of truth for the intent taxonomy.  Both the
classifier and the retrieval layer import from here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# ── Canonical intent labels ───────────────────────────────────────────────
INTENTS: List[str] = [
    "order_status_inquiry",
    "delivery_issue",
    "shipping_delay_complaint",
    "refund_request",
    "order_cancellation",
    "return_or_exchange",
    "product_issue",
    "account_access",
    "prime_membership",
    "device_and_digital_support",
    "general_complaint_or_feedback",
]

# ── Definitions (used verbatim in the system prompt) ─────────────────────
DEFINITIONS: dict[str, str] = {
    "order_status_inquiry": (
        "Customer asks where their order is, requests tracking information, "
        "or asks for an estimated delivery time — without yet asserting that "
        "something has gone wrong."
    ),
    "delivery_issue": (
        "Customer reports a concrete delivery failure: package marked "
        "'delivered' but not received, delivered to the wrong address or "
        "neighbour, left in an unsafe location, or the delivery agent "
        "behaved badly."
    ),
    "shipping_delay_complaint": (
        "Customer complains that a promised shipping speed (e.g. Prime "
        "2-day, next-day) was not met, or that their shipment is running "
        "late relative to the stated estimate."
    ),
    "refund_request": (
        "Customer asks for a refund, reports an unauthorised charge, "
        "questions a payment, or asks why a promised refund has not arrived."
    ),
    "order_cancellation": (
        "Customer wants to cancel an existing order, asks whether a "
        "cancellation is still possible, or reports difficulty cancelling."
    ),
    "return_or_exchange": (
        "Customer wants to return an item, exchange it for a different "
        "size/colour/product, or asks about the return window or process."
    ),
    "product_issue": (
        "Customer received a defective, damaged, counterfeit, wrong, or "
        "incomplete item, or reports that a product stopped working."
    ),
    "account_access": (
        "Customer cannot sign in — account locked, suspended, hacked, or "
        "they have forgotten their password or no longer control the "
        "associated email address."
    ),
    "prime_membership": (
        "Customer has a question or complaint specifically about Amazon "
        "Prime membership: billing, cancellation, renewal, trial period, "
        "or whether a Prime benefit applies."
    ),
    "device_and_digital_support": (
        "Customer needs technical help with an Amazon device (Kindle, "
        "Fire tablet, Fire Stick, Echo/Alexa) or a digital service "
        "(Prime Video, Amazon Music, app errors, digital content access)."
    ),
    "general_complaint_or_feedback": (
        "Customer expresses general dissatisfaction or satisfaction, "
        "vents about the brand, or gives feedback without a specific "
        "actionable request that fits any category above.  This is the "
        "catch-all for residual / ambiguous messages."
    ),
}

# ── Few-shot examples (2 per intent) — real tweets from our sample ────────
# Format: (tweet_text, intent_label)
FEW_SHOT_EXAMPLES: List[tuple[str, str]] = [
    # order_status_inquiry
    (
        "@AmazonHelp just wondering when my package will be delivered? "
        "The order is #111-4488671-1254655. It's been a while.",
        "order_status_inquiry",
    ),
    (
        "@AmazonHelp where is my order? Tracking hasn't updated in 3 days.",
        "order_status_inquiry",
    ),
    # delivery_issue
    (
        "@AmazonHelp Found my package but in someone else's house. "
        "Amazon delivered it to some other person and updated as handed "
        "over to customer directly.",
        "delivery_issue",
    ),
    (
        "@AmazonHelp just wanted to let y'all know you delivered my packages "
        "to the wrong house... luckily they were nice people & brought me "
        "my stuff",
        "delivery_issue",
    ),
    # shipping_delay_complaint
    (
        "@AmazonHelp October 31st? Um that's not 2 day shipping. That's "
        "4 day shipping. I bought Prime to avoid this...what the hell",
        "shipping_delay_complaint",
    ),
    (
        "@AmazonHelp I'm a prime member and I paid extra for next day "
        "shipping but it still hasn't shipped after two days.",
        "shipping_delay_complaint",
    ),
    # refund_request
    (
        "@AmazonHelp I need to get a refund for order# 114-2485779-5009866 "
        "I received a pair of shoes that were the wrong size.",
        "refund_request",
    ),
    (
        "@AmazonHelp Hi, I haven't received my refund from 3 weeks ago. "
        "I was told it would take 3-5 business days. Could you look into it?",
        "refund_request",
    ),
    # order_cancellation
    (
        "@AmazonHelp if I cancel an order today before it is dispatched "
        "do I still get charged?",
        "order_cancellation",
    ),
    (
        "@AmazonHelp please help. I need to cancel or re-route an order "
        "that was ordered by mistake! It just was ordered yesterday.",
        "order_cancellation",
    ),
    # return_or_exchange
    (
        "@AmazonHelp hi I ordered some shoes and they came with a defect. "
        "I need to exchange them or return. Thanks.",
        "return_or_exchange",
    ),
    (
        "@AmazonHelp How can I get a pre-signed return label printed? "
        "Thank you!",
        "return_or_exchange",
    ),
    # product_issue
    (
        "@AmazonHelp I ordered cable modem that costed over $200 and "
        "what shows up is butter chips and diapers.",
        "product_issue",
    ),
    (
        "@AmazonHelp This is not the appropriate way to ship a laptop, "
        "there is no padding whatsoever.",
        "product_issue",
    ),
    # account_access
    (
        "@AmazonHelp My account has been suspended for 5 weeks. My appeal "
        "said it would be looked at by October 4th. Please advise.",
        "account_access",
    ),
    (
        "@AmazonHelp I am having problems logging into my account and "
        "resetting my password.",
        "account_access",
    ),
    # prime_membership
    (
        "@AmazonHelp hello! my prime membership renewed automatically but "
        "it charged the wrong payment method. Is it possible to get a refund?",
        "prime_membership",
    ),
    (
        "Hey, @AmazonHelp. I cancelled my Prime. Can you quit charging "
        "my card now? Thanks.",
        "prime_membership",
    ),
    # device_and_digital_support
    (
        "@AmazonHelp I have a Kindle Fire HD and it can't recognize when "
        "it's being charged. I have to restart the device every time.",
        "device_and_digital_support",
    ),
    (
        "@AmazonHelp hey I was wondering if you could help me fix an error "
        "on my prime video app?",
        "device_and_digital_support",
    ),
    # general_complaint_or_feedback
    (
        "@AmazonHelp has the WORST customer service period",
        "general_complaint_or_feedback",
    ),
    (
        "Shoutout to @AmazonHelp customer service phone team - Phone call "
        "was only 2 minutes & 48 seconds and the issue was resolved!! FAST!!",
        "general_complaint_or_feedback",
    ),

    # ── Banking77 contrastive examples (real dataset lookups) ─────────────
    # These are populated dynamically in get_few_shot_block() to avoid
    # hardcoding synthetic strings.
]


@dataclass
class Intent:
    """Convenience wrapper — not required at runtime, useful for IDE tooling."""
    label: str
    definition: str
    examples: List[str] = field(default_factory=list)


def get_intent_definitions_block() -> str:
    """Return a formatted string of all intents for use in system prompts."""
    lines = []
    for label in INTENTS:
        lines.append(f"- **{label}**: {DEFINITIONS[label]}")
    return "\n".join(lines)


def get_few_shot_block() -> str:
    """Return formatted few-shot examples for use in prompts."""
    lines = []
    for tweet, label in FEW_SHOT_EXAMPLES:
        lines.append(f'Tweet: "{tweet}"\nIntent: {label}\n')

    # Dynamically inject real Banking77 contrastive examples
    import os, json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(root, "data", "banking77_cross_domain_sample.jsonl")
    
    if os.path.exists(sample_path):
        target_intents = {
            "order_status_inquiry": 3,
            "shipping_delay_complaint": 3,
            "prime_membership": 3,
            "delivery_issue": 3
        }
        found: dict[str, list[str]] = {k: [] for k in target_intents}
        
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                label = row.get("expected_amazon_label")
                if label in target_intents and len(found[label]) < target_intents[label]:
                    found[label].append(row["text"])
                    
        for label, texts in found.items():
            for t in texts:
                lines.append(f'Tweet: "{t}"\nIntent: {label}\n')

    return "\n".join(lines)
