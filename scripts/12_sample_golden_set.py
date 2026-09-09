"""
12_sample_golden_set.py — Generate stratified, diversity-sampled candidate tweets
for a human-labeled golden evaluation set.

The user will hand-label the output sheet.  This script does NOT auto-label.

Sampling method (documented in results/golden_set_sampling_notes.txt):
    1.  Load data/amazon_help_sample.csv  (3,000 inbound AmazonHelp tweets).
    2.  Apply keyword heuristics (same PATTERN_MAP as 05_label_sample.py) to
        assign a PREDICTED intent to each tweet.  This is used for stratification
        only — not as a ground-truth label.
    3.  Per-intent stratum:
        a.  Embed clean text with sentence-transformers all-MiniLM-L6-v2.
        b.  k-means (k=5) on the embeddings gives 5 diversity buckets.
        c.  Sample ceil(target / 5) tweets from each bucket, shuffled.
    4.  Target: 18–20 per intent × 11 intents ≈ 198–220 total candidates.
    5.  Exclude tweets already in data/labelled_eval.jsonl (avoid contamination).

Outputs:
    data/golden_candidates.csv       — full candidate list with metadata
    data/golden_label_sheet.csv      — same + blank human_label + human_escalate columns
    results/golden_set_sampling_notes.txt  — this method, reproducibly documented

Run:
    python scripts/12_sample_golden_set.py

No API keys required.  Runtime: ~60 s (embedding step).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import textwrap

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from taxonomy import INTENTS

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLE_CSV  = os.path.join(DATA_DIR, "amazon_help_sample.csv")
EVAL_JSONL  = os.path.join(DATA_DIR, "labelled_eval.jsonl")
OUT_CAND    = os.path.join(DATA_DIR, "golden_candidates.csv")
OUT_SHEET   = os.path.join(DATA_DIR, "golden_label_sheet.csv")
OUT_NOTES   = os.path.join(RESULTS_DIR, "golden_set_sampling_notes.txt")

TARGET_PER_INTENT   = 19   # 19 × 11 = 209 ≈ midpoint of 150-250 range
DIVERSITY_BUCKETS   = 5
RANDOM_SEED         = 42

# ── Intent keyword patterns (same as 05_label_sample.py) ──────────────────
PATTERN_MAP = [
    ("order_status_inquiry",
     r"track|where is my order|where.s my package|shipping status|how long|"
     r"when will|estimated delivery|hasn.t arrived|still waiting"),
    ("delivery_issue",
     r"delivered to wrong|wrong address|someone else.s|wrong house|left.*unsafe|"
     r"not received.*marked|says delivered|delivered incorrectly|driver"),
    ("shipping_delay_complaint",
     r"2.day shipping|next.day shipping|prime shipping|paid.*shipping|"
     r"shipping.*late|shipping.*slow|delayed.*shipping"),
    ("refund_request",
     r"refund|money back|reimburse|unauthori[sz]ed charge|missing refund|get my money"),
    ("order_cancellation",
     r"cancel (my )?order|how (do i|to) cancel|need to cancel|cancel.*before|stop.*order"),
    ("return_or_exchange",
     r"\breturn\b|\bexchange\b|send (it )?back|swap|return label|"
     r"return (window|policy|process)|how do i return"),
    ("product_issue",
     r"defective|damaged|broken|doesn.t work|not working|wrong (item|product)|"
     r"counterfeit|fake|empty box|missing (item|part)|wrong size"),
    ("account_access",
     r"account.*lock|account.*block|account.*suspend|can.t (log|sign).?in|"
     r"forgot.*password|reset.*password|locked out|hacked|unauthori[sz]ed access"),
    ("prime_membership",
     r"prime (member|subscription|trial|free|benefit)|"
     r"membership (fee|charge|cancel|renew|cost)|cancel.*prime|prime.*price"),
    ("device_and_digital_support",
     r"kindle|fire (tablet|stick|tv|hd)|echo|alexa|prime video|streaming|"
     r"app.*crash|device.*not work|digital content"),
    ("general_complaint_or_feedback", None),  # catch-all
]


def keyword_label(text: str) -> str:
    t = text.lower()
    for intent, pattern in PATTERN_MAP:
        if pattern is None:
            return intent
        if re.search(pattern, t):
            return intent
    return "general_complaint_or_feedback"


def clean_text(text: str) -> str:
    """Strip @mentions and URLs; collapse whitespace."""
    text = re.sub(r"@\w+", "@mention", str(text))
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    np.random.seed(RANDOM_SEED)

    # ── 1. Load pool ───────────────────────────────────────────────────────
    print("Loading pool…")
    df = pd.read_csv(SAMPLE_CSV)
    # Inbound-only (customer tweets)
    if "inbound" in df.columns:
        df = df[df["inbound"] == True].copy()
    df["clean"] = df["text"].apply(clean_text)
    # Drop very short tweets (< 30 chars after cleaning)
    df = df[df["clean"].str.len() >= 30].reset_index(drop=True)
    print(f"  Pool after cleaning: {len(df)} tweets")

    # ── 2. Exclude existing labelled_eval.jsonl tweets ────────────────────
    existing: set[str] = set()
    if os.path.exists(EVAL_JSONL):
        with open(EVAL_JSONL, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                existing.add(obj["tweet"][:80])
    df = df[~df["clean"].str[:80].isin(existing)].reset_index(drop=True)
    print(f"  After removing eval contamination: {len(df)} tweets")

    # ── 3. Predict intent via keywords ────────────────────────────────────
    print("Predicting intents via keywords…")
    df["predicted_intent"] = df["clean"].apply(keyword_label)
    dist = df["predicted_intent"].value_counts()
    print("  Predicted intent distribution:")
    for k, v in dist.items():
        print(f"    {k:40s} {v}")

    # ── 4. Embed for diversity bucketing ──────────────────────────────────
    print("Loading embedding model for diversity bucketing…")
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Encoding…", flush=True)
    embeddings = model.encode(df["clean"].tolist(), batch_size=256,
                              show_progress_bar=True, normalize_embeddings=True)
    df["_emb_idx"] = range(len(df))
    emb_matrix = np.array(embeddings)

    # ── 5. Stratified + diversity sample ──────────────────────────────────
    print(f"\nSampling {TARGET_PER_INTENT} per intent (diversity k={DIVERSITY_BUCKETS})…")
    all_candidates: list[dict] = []
    intent_stats: dict[str, dict] = {}

    for intent in INTENTS:
        mask = df["predicted_intent"] == intent
        stratum = df[mask].copy()
        n_avail = len(stratum)

        if n_avail == 0:
            print(f"  ⚠  {intent}: 0 examples — skipping")
            intent_stats[intent] = {"available": 0, "sampled": 0}
            continue

        # Diversity buckets via k-means on embeddings
        k = min(DIVERSITY_BUCKETS, n_avail)
        emb_sub = emb_matrix[stratum["_emb_idx"].values]
        if k > 1:
            km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=3)
            stratum = stratum.copy()
            stratum["diversity_bucket"] = km.fit_predict(emb_sub)
        else:
            stratum = stratum.copy()
            stratum["diversity_bucket"] = 0

        # Sample evenly across buckets
        target = min(TARGET_PER_INTENT, n_avail)
        per_bucket = math.ceil(target / k)
        sampled_parts = []
        for b in range(k):
            bucket_rows = stratum[stratum["diversity_bucket"] == b]
            take = min(per_bucket, len(bucket_rows))
            sampled_parts.append(
                bucket_rows.sample(n=take, random_state=RANDOM_SEED)
            )
        sampled = pd.concat(sampled_parts, ignore_index=True)
        # May overshoot slightly — trim to target
        if len(sampled) > TARGET_PER_INTENT:
            sampled = sampled.sample(n=TARGET_PER_INTENT, random_state=RANDOM_SEED)

        intent_stats[intent] = {"available": n_avail, "sampled": len(sampled)}
        for _, row in sampled.iterrows():
            all_candidates.append({
                "id":                f"{intent[:4]}_{row.get('tweet_id', row.name)}",
                "tweet":             row["text"],
                "clean_tweet":       row["clean"],
                "predicted_intent":  intent,
                "diversity_bucket":  int(row["diversity_bucket"]),
            })

    # ── 6. Shuffle and write ───────────────────────────────────────────────
    np.random.shuffle(all_candidates)
    for i, cand in enumerate(all_candidates):
        cand["sample_id"] = i + 1

    cand_df = pd.DataFrame(all_candidates, columns=[
        "sample_id", "id", "tweet", "clean_tweet",
        "predicted_intent", "diversity_bucket",
    ])

    # Candidate list (with metadata)
    cand_df.to_csv(OUT_CAND, index=False)

    # Label sheet (user fills in human_label + human_escalate)
    sheet_df = cand_df[["sample_id", "id", "tweet",
                         "predicted_intent", "diversity_bucket"]].copy()
    sheet_df["human_label"]    = ""   # user fills: one of 11 intent names
    sheet_df["human_escalate"] = ""   # user fills: Y or N
    sheet_df["notes"]          = ""   # optional
    sheet_df.to_csv(OUT_SHEET, index=False)

    # ── 7. Print summary ──────────────────────────────────────────────────
    n_total = len(all_candidates)
    print(f"\n{'─'*60}")
    print(f"Generated {n_total} candidates  →  {OUT_CAND}")
    print(f"Label sheet                    →  {OUT_SHEET}")
    print(f"\nPer-intent breakdown:")
    for intent in INTENTS:
        s = intent_stats.get(intent, {"available": 0, "sampled": 0})
        print(f"  {intent:40s}  avail={s['available']:4d}  sampled={s['sampled']:3d}")

    # ── 8. Write sampling notes ───────────────────────────────────────────
    notes = textwrap.dedent(f"""\
        Golden Set Sampling Notes
        ========================
        Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        Script   : scripts/12_sample_golden_set.py
        Pool     : data/amazon_help_sample.csv  ({len(df)+len(existing)} tweets before dedup)
        Exclusion: data/labelled_eval.jsonl tweets removed to avoid contamination
        Pool after cleaning & dedup: {len(df)} tweets
        Target per intent: {TARGET_PER_INTENT}
        Total candidates : {n_total}
        Random seed      : {RANDOM_SEED}

        Sampling Method
        ---------------
        1. Keyword heuristics (PATTERN_MAP, same as 05_label_sample.py) assign
           a PREDICTED intent to each tweet.  This is used for stratification only
           — NOT as a ground-truth label.  The user must assign human_label.

        2. Per-intent stratum:
           a. Embed tweet text using sentence-transformers/all-MiniLM-L6-v2
              (384-dim, L2-normalised).
           b. MiniBatchKMeans (k={DIVERSITY_BUCKETS}) gives {DIVERSITY_BUCKETS} diversity buckets.
           c. Sample ceil({TARGET_PER_INTENT}/{DIVERSITY_BUCKETS}) tweets from each bucket,
              then trim to {TARGET_PER_INTENT} if oversampled.

        3. All candidates are shuffled (seed={RANDOM_SEED}) before writing so
           the label sheet has no visible intent ordering.

        Diversity rationale
        -------------------
        Without diversity sampling, each stratum would oversample the modal
        cluster (e.g., refund_request tweets tend to cluster around "where is
        my refund" phrasing).  Bucketing ensures the 19 examples per intent
        cover different phrasings, tones, and sub-topics.

        Label instructions for the human annotator
        ------------------------------------------
        Open data/golden_label_sheet.csv and fill in two columns:

        human_label    — one of the 11 taxonomy intents, or SKIP if ambiguous:
          account_access, delivery_issue, device_and_digital_support,
          general_complaint_or_feedback, order_cancellation, order_status_inquiry,
          prime_membership, product_issue, refund_request, return_or_exchange,
          shipping_delay_complaint

        human_escalate — Y if you would escalate to a human agent, N if you
          would auto-handle.  Use Y for: legal/media threats, security breaches,
          repeated unresolved contacts (>2), extreme negative language.

        notes  — optional free text

        Per-intent availability and sample counts
        -----------------------------------------
    """)
    for intent in INTENTS:
        s = intent_stats.get(intent, {"available": 0, "sampled": 0})
        notes += f"  {intent:42s}  avail={s['available']:4d}  sampled={s['sampled']}\n"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_NOTES, "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"\nSampling notes                 →  {OUT_NOTES}")
    print("\nNext step: open data/golden_label_sheet.csv and fill in")
    print("  human_label and human_escalate, then run 13_automated_metrics.py")


if __name__ == "__main__":
    main()
