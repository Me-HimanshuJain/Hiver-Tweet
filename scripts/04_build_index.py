"""
04_build_index.py — Build the FAISS retrieval index from resolved
AmazonHelp conversation threads.

A "resolved thread" = a customer tweet that AmazonHelp replied to,
paired with the brand's reply.  We extract these from the full dataset
(which we already have cached locally from step 1).

Run:
    python scripts/04_build_index.py

Outputs:
    data/resolved_threads.jsonl
    data/faiss_index.bin
    data/faiss_meta.jsonl
"""
import os
import sys
import json
import re
import kagglehub
import pandas as pd

# ── Make src/ importable ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from retrieval import build_index
from taxonomy import INTENTS

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "data")
THREADS_PATH = os.path.join(DATA_DIR, "resolved_threads.jsonl")
INDEX_PATH   = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH    = os.path.join(DATA_DIR, "faiss_meta.jsonl")

os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Load the full dataset (already cached from step 1) ─────────────────
print("Loading dataset from Kaggle cache...")
path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
csv_file = os.path.join(path, "twcs", "twcs.csv")
if not os.path.exists(csv_file):
    csv_file = os.path.join(path, "twcs.csv")

df = pd.read_csv(csv_file)
print(f"Loaded {len(df):,} rows")

# ── 2. Extract resolved AmazonHelp threads ────────────────────────────────
# A resolved thread = an AmazonHelp outbound reply + the customer tweet it
# replies to.  We want pairs: (customer_tweet, brand_reply).

amazon_out = df[(df["author_id"] == "AmazonHelp") & (df["inbound"] == False)].copy()
amazon_out = amazon_out.dropna(subset=["in_response_to_tweet_id"])
amazon_out["resp_id"] = amazon_out["in_response_to_tweet_id"].astype(int)
print(f"AmazonHelp outbound replies: {len(amazon_out):,}")

# Build lookup: tweet_id → text for inbound tweets
inbound = df[df["inbound"] == True][["tweet_id", "text"]].set_index("tweet_id")

# Join
merged = amazon_out.set_index("resp_id").join(
    inbound.rename(columns={"text": "customer_tweet"}),
    how="inner"
)
merged = merged.rename(columns={"text": "brand_reply"})
merged = merged.dropna(subset=["customer_tweet", "brand_reply"])

print(f"Resolved thread pairs found: {len(merged):,}")

# ── 3. Filter: only threads where customer tweet mentions @AmazonHelp ─────
merged = merged[
    merged["customer_tweet"].str.contains("@AmazonHelp", case=False, na=False)
]
print(f"Filtered to @AmazonHelp mentions: {len(merged):,}")

# ── 4. Sample — keep a manageable index size (≤ 8,000 for speed) ─────────
MAX_INDEX = 8000
if len(merged) > MAX_INDEX:
    merged = merged.sample(n=MAX_INDEX, random_state=42)
    print(f"Sampled to {MAX_INDEX} for index build")

# ── 5. Assign placeholder intents using keyword heuristics ───────────────
# We don't have ground-truth labels for the index threads, so we apply a
# simple keyword classifier as a bootstrap.  This will be refined once the
# classifier is available.
def keyword_intent(text: str) -> str:
    t = text.lower()
    if re.search(r"track|where is my order|where.s my package|shipping status", t):
        return "order_status_inquiry"
    if re.search(r"delivered to wrong|wrong address|someone else.s house|delivered to the wrong", t):
        return "delivery_issue"
    if re.search(r"cancel|cancell", t):
        return "order_cancellation"
    if re.search(r"return|exchange|send back|swap", t):
        return "return_or_exchange"
    if re.search(r"refund|money back|reimburse", t):
        return "refund_request"
    if re.search(r"defective|damaged|broken|doesn.t work|not working|wrong (item|product)|counterfeit|fake", t):
        return "product_issue"
    if re.search(r"account.*lock|account.*block|account.*suspend|password|sign.?in|log.?in|locked out|hacked", t):
        return "account_access"
    if re.search(r"prime (member|subscription|trial|free)|membership|renew", t):
        return "prime_membership"
    if re.search(r"kindle|fire (tablet|stick|tv)|echo|alexa|prime video|streaming|app.*crash|device", t):
        return "device_and_digital_support"
    if re.search(r"late|delay|2.day|next.day|shipping.*slow", t):
        return "shipping_delay_complaint"
    return "general_complaint_or_feedback"

merged["intent"] = merged["customer_tweet"].apply(keyword_intent)

intent_dist = merged["intent"].value_counts()
print(f"\nBootstrap intent distribution:")
for intent, n in intent_dist.items():
    print(f"  {intent:40s} {n:4d}")

# ── 6. Build thread records ───────────────────────────────────────────────
threads = []
for i, (idx, row) in enumerate(merged.iterrows()):
    threads.append({
        "thread_id":      str(idx),
        "customer_tweet": str(row["customer_tweet"])[:280],
        "brand_reply":    str(row["brand_reply"])[:280],
        "intent":         row["intent"],
    })

# Save JSONL
with open(THREADS_PATH, "w", encoding="utf-8") as f:
    for t in threads:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(f"\nSaved {len(threads)} threads → {THREADS_PATH}")

# ── 7. Build FAISS index ──────────────────────────────────────────────────
print("\nBuilding FAISS index...")
build_index(threads, INDEX_PATH, META_PATH)

print("\n✓ Index build complete.")
print(f"  Threads  : {THREADS_PATH}")
print(f"  Index    : {INDEX_PATH}")
print(f"  Metadata : {META_PATH}")
