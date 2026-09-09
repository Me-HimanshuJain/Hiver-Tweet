"""
05_label_sample.py — Create a small, hand-stratified labelled evaluation set
for the classifier.

Strategy: pull ~10 tweets per intent from the clustered sample, clean them,
and write them with SUGGESTED labels (from keyword heuristics).

The resulting labelled_eval.jsonl is the ground-truth test set for 06_evaluate.py.

IMPORTANT: In a real project you would review and correct these labels manually
before running the evaluator.  For reproducibility the labels here are derived
from the same keyword heuristics used in step 4, applied on a HELD-OUT set
(tweets NOT in the training few-shots).

Run:
    python scripts/05_label_sample.py

Output:
    data/labelled_eval.jsonl   — {tweet, label} per line
"""
import os, sys, json, re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from taxonomy import INTENTS

DATA_DIR = os.path.join(ROOT, "data")
SAMPLE_CSV = os.path.join(DATA_DIR, "amazon_help_sample.csv")
OUT_PATH   = os.path.join(DATA_DIR, "labelled_eval.jsonl")

# ── Intent patterns (same as 04_build_index.py) ────────────────────────
PATTERN_MAP = [
    ("order_status_inquiry",        r"track|where is my order|where.s my package|shipping status|how long|when will|estimated delivery|hasn.t arrived|still waiting"),
    ("delivery_issue",              r"delivered to wrong|wrong address|someone else.s|wrong house|left.*unsafe|not received.*marked|says delivered|delivered incorrectly|driver"),
    ("shipping_delay_complaint",    r"2.day shipping|next.day shipping|prime shipping|paid.*shipping|shipping.*late|shipping.*slow|delayed.*shipping"),
    ("refund_request",              r"refund|money back|reimburse|unauthori[sz]ed charge|missing refund|get my money"),
    ("order_cancellation",          r"cancel (my )?order|how (do i|to) cancel|need to cancel|cancel.*before|stop.*order"),
    ("return_or_exchange",          r"\breturn\b|\bexchange\b|send (it )?back|swap|return label|return (window|policy|process)|how do i return"),
    ("product_issue",               r"defective|damaged|broken|doesn.t work|not working|wrong (item|product)|counterfeit|fake|empty box|missing (item|part)|wrong size"),
    ("account_access",              r"account.*lock|account.*block|account.*suspend|can.t (log|sign).?in|forgot.*password|reset.*password|locked out|hacked|unauthori[sz]ed access"),
    ("prime_membership",            r"prime (member|subscription|trial|free|benefit)|membership (fee|charge|cancel|renew|cost)|cancel.*prime|prime.*price"),
    ("device_and_digital_support",  r"kindle|fire (tablet|stick|tv|hd)|echo|alexa|prime video|streaming|app.*crash|device.*not work|digital content"),
    ("general_complaint_or_feedback", None),   # catch-all
]

def keyword_label(text: str) -> str | None:
    t = text.lower()
    for intent, pattern in PATTERN_MAP:
        if pattern is None:
            return intent
        if re.search(pattern, t):
            return intent
    return "general_complaint_or_feedback"


# ── Load and clean sample ─────────────────────────────────────────────────
df = pd.read_csv(SAMPLE_CSV)
df = df.dropna(subset=["text"]).copy()

# Remove the few-shot tweets used in the system prompt (exact dedup)
from taxonomy import FEW_SHOT_EXAMPLES
few_shot_texts = {t.lower().strip() for t, _ in FEW_SHOT_EXAMPLES}
df = df[~df["text"].str.lower().str.strip().isin(few_shot_texts)]

# Apply keyword labels
df["label"] = df["text"].apply(keyword_label)

# ── Stratified sample: 10 per intent ─────────────────────────────────────
SAMPLES_PER_INTENT = 10
records = []
for intent in INTENTS:
    subset = df[df["label"] == intent]
    n = min(SAMPLES_PER_INTENT, len(subset))
    if n == 0:
        print(f"  WARNING: no examples for {intent}")
        continue
    sampled = subset.sample(n=n, random_state=42)
    for _, row in sampled.iterrows():
        records.append({
            "tweet": str(row["text"]),
            "label": intent,
        })
    print(f"  {intent:40s} {n:2d} examples")

print(f"\nTotal eval records: {len(records)}")

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Saved → {OUT_PATH}")
