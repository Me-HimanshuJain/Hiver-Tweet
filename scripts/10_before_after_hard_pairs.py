"""
10_before_after_hard_pairs.py
==============================
Phase 3: before/after F1 on the two hardest confusion pairs.

"Before"  = Run 4 results already saved in results/eval_report_corrected.json
            (no API calls needed for before-side).

"After"   = Re-classify the 30 tweets from labelled_eval.jsonl that belong
            to the three hard-pair intents (order_status_inquiry,
            shipping_delay_complaint, prime_membership) using the updated
            taxonomy.py (which now includes Banking77 contrastive few-shots).

Usage
-----
    $env:NVIDIA_API_KEY = 'nvapi-...'
    python scripts/10_before_after_hard_pairs.py

Outputs
-------
    results/before_after_hard_pairs.json
    results/before_after_hard_pairs.txt
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from classifier import classify_batch, MODEL_ID, MIN_INTERVAL

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
EVAL_PATH   = os.path.join(DATA_DIR, "labelled_eval.jsonl")
BEFORE_JSON = os.path.join(RESULTS_DIR, "eval_report_corrected.json")
REPORT_JSON = os.path.join(RESULTS_DIR, "before_after_hard_pairs.json")
REPORT_TXT  = os.path.join(RESULTS_DIR, "before_after_hard_pairs.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)

HARD_PAIR_INTENTS = [
    "order_status_inquiry",
    "shipping_delay_complaint",
    "prime_membership",
]

# ── Load "before" metrics from Run 4 ──────────────────────────────────────
if not os.path.exists(BEFORE_JSON):
    sys.exit("ERROR: Run scripts/07_retry_failed.py first to produce the baseline.")

with open(BEFORE_JSON, encoding="utf-8") as f:
    before_report = json.load(f)

before_metrics = {
    intent: before_report["per_class_metrics"].get(intent, {})
    for intent in HARD_PAIR_INTENTS
}

print("BEFORE metrics (Run 4 — with original taxonomy):")
print(f"  {'Intent':<35s} {'F1':>6} {'P':>6} {'R':>6}")
for intent in HARD_PAIR_INTENTS:
    m = before_metrics[intent]
    print(f"  {intent:<35s} {m.get('f1',0):6.3f} {m.get('precision',0):6.3f} {m.get('recall',0):6.3f}")
print()

# ── Load the 30 hard-pair tweets from the eval set ─────────────────────────
all_records = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8") if l.strip()]
hard_records = [r for r in all_records if r["label"] in HARD_PAIR_INTENTS]
tweets      = [r["tweet"] for r in hard_records]
true_labels = [r["label"] for r in hard_records]

print(f"Hard-pair eval set: {len(hard_records)} tweets")
from collections import Counter
print("  Distribution:", dict(Counter(true_labels)))
print(f"\nModel    : {MODEL_ID}")
print(f"Throttle : {MIN_INTERVAL}s per call, max_workers=2")
print(f"Taxonomy : NOW includes Banking77 contrastive few-shots for hard pairs")
print()

# ── Classify ───────────────────────────────────────────────────────────────
t0 = time.perf_counter()
raw_results = classify_batch(tweets, max_workers=2, progress=True)
elapsed = time.perf_counter() - t0
print(f"\nDone in {elapsed:.1f}s")

pred_labels = []
for r in raw_results:
    if r.error or r.intent == "uncertain":
        pred_labels.append("general_complaint_or_feedback")
    else:
        pred_labels.append(r.intent)

n_errors = sum(1 for r in raw_results if r.error)

# ── Compute after metrics ─────────────────────────────────────────────────
report_dict = classification_report(
    true_labels, pred_labels,
    labels=HARD_PAIR_INTENTS,
    output_dict=True,
    zero_division=0,
)
report_str = classification_report(
    true_labels, pred_labels,
    labels=HARD_PAIR_INTENTS,
    zero_division=0,
)

after_metrics = {
    intent: {
        "precision": round(report_dict[intent]["precision"], 3),
        "recall":    round(report_dict[intent]["recall"],    3),
        "f1":        round(report_dict[intent]["f1-score"],  3),
        "support":   int(report_dict[intent]["support"]),
    }
    for intent in HARD_PAIR_INTENTS
}

# ── Delta ─────────────────────────────────────────────────────────────────
deltas = {
    intent: {
        "f1_before":    round(before_metrics[intent].get("f1", 0), 3),
        "f1_after":     after_metrics[intent]["f1"],
        "delta_f1":     round(after_metrics[intent]["f1"] - before_metrics[intent].get("f1", 0), 3),
        "p_before":     round(before_metrics[intent].get("precision", 0), 3),
        "p_after":      after_metrics[intent]["precision"],
        "r_before":     round(before_metrics[intent].get("recall", 0), 3),
        "r_after":      after_metrics[intent]["recall"],
    }
    for intent in HARD_PAIR_INTENTS
}

# ── Per-tweet detail ───────────────────────────────────────────────────────
per_tweet = []
for rec, pred, res in zip(hard_records, pred_labels, raw_results):
    per_tweet.append({
        "tweet": rec["tweet"][:120],
        "true":  rec["label"],
        "pred":  pred,
        "conf":  res.confidence,
        "correct": rec["label"] == pred,
        "error": res.error,
    })

# ── Save report ───────────────────────────────────────────────────────────
report = {
    "model": MODEL_ID,
    "n_hard_pair_tweets": len(hard_records),
    "n_errors": n_errors,
    "elapsed_seconds": round(elapsed, 2),
    "note": "Banking77 contrastive few-shots added to taxonomy.py before this run.",
    "before_metrics": before_metrics,
    "after_metrics":  after_metrics,
    "deltas":         deltas,
    "per_tweet":      per_tweet,
    "full_classification_report": report_str,
}

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ── Human-readable ────────────────────────────────────────────────────────
W = 72
lines = [
    "=" * W,
    "BEFORE / AFTER — HARD-PAIR F1 IMPROVEMENT",
    "=" * W,
    f"Model   : {MODEL_ID}",
    f"Change  : Banking77 contrastive few-shots injected into taxonomy.py",
    f"Tweets  : {len(hard_records)} (10 per intent × 3 hard-pair intents)",
    f"Errors  : {n_errors}",
    "",
    "─" * W,
    f"{'Intent':<35s} {'F1-Before':>10} {'F1-After':>10} {'ΔF1':>8}",
    "─" * W,
]
for intent, d in deltas.items():
    arrow = "▲" if d["delta_f1"] > 0 else ("▼" if d["delta_f1"] < 0 else "─")
    lines.append(
        f"{intent:<35s} {d['f1_before']:>10.3f} {d['f1_after']:>10.3f} "
        f"{arrow}{abs(d['delta_f1']):>6.3f}"
    )

lines += [
    "",
    "─" * W,
    "DETAILED BEFORE / AFTER (Precision / Recall / F1)",
    "─" * W,
    f"{'Intent':<35s} {'Metric':>8} {'Before':>8} {'After':>8} {'Delta':>8}",
]
for intent, d in deltas.items():
    for metric, key_b, key_a in [("Precision","p_before","p_after"),
                                   ("Recall",   "r_before","r_after"),
                                   ("F1",       "f1_before","f1_after")]:
        val_b = d[key_b]
        val_a = d[key_a]
        delta = round(val_a - val_b, 3)
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        lines.append(
            f"{intent:<35s} {metric:>8} {val_b:>8.3f} {val_a:>8.3f} "
            f"{arrow}{abs(delta):>6.3f}"
        )
    lines.append("")

lines += [
    "─" * W,
    "PER-TWEET RESULTS (hard-pair tweets)",
    "─" * W,
]
for t in per_tweet:
    status = "✓" if t["correct"] else "✗"
    lines.append(
        f"  {status} [{t['true']:30s}→{t['pred']:30s}] conf={t['conf']:.2f}"
    )
    lines.append(f"    {t['tweet'][:90]}")

lines.append("=" * W)
summary = "\n".join(lines)
print("\n" + summary)

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\nSaved → {REPORT_JSON}")
print(f"Saved → {REPORT_TXT}")
