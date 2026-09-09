"""
07_retry_failed.py — Re-classify only the tweets that errored in the first
evaluation run, then merge with the existing good results to produce a
corrected eval_report.

Usage:
    $env:NVIDIA_API_KEY = 'nvapi-...'
    python scripts/07_retry_failed.py

Reads:
    data/labelled_eval.jsonl          — ground-truth labels
    results/eval_report.json          — prior run (needs per-tweet detail)

Writes:
    results/eval_report_corrected.json
    results/eval_report_corrected.txt

NOTE: The prior eval_report.json does not store per-tweet predictions, so
this script re-runs the full 110 tweets against the fixed classifier.
The improved retry (5 attempts, 503-specific backoff) should eliminate the
27-error problem.  Expected runtime: ~6 min with max_workers=2.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from classifier import classify_batch, CONFIDENCE_THRESHOLD, MODEL_ID, MIN_INTERVAL
from taxonomy import INTENTS

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
EVAL_PATH   = os.path.join(DATA_DIR, "labelled_eval.jsonl")
REPORT_JSON = os.path.join(RESULTS_DIR, "eval_report_corrected.json")
REPORT_TXT  = os.path.join(RESULTS_DIR, "eval_report_corrected.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load eval set ──────────────────────────────────────────────────────────
records = [json.loads(l) for l in open(EVAL_PATH, encoding='utf-8') if l.strip()]
tweets      = [r["tweet"] for r in records]
true_labels = [r["label"] for r in records]

print(f"Loaded {len(records)} labelled tweets")
print(f"Model      : {MODEL_ID}")
print(f"Throttle   : {MIN_INTERVAL}s between calls, max_workers=2")
print(f"Retries    : 5 attempts, 503-specific backoff (10s/20s/40s/80s/160s)")
print()

# ── Classify ───────────────────────────────────────────────────────────────
t0 = time.perf_counter()
raw_results = classify_batch(tweets, max_workers=2, progress=True)
elapsed = time.perf_counter() - t0
print(f"\nDone in {elapsed:.1f}s  ({elapsed / len(tweets) * 1000:.0f}ms/tweet avg)")

errors  = [r for r in raw_results if r.error]
n_err   = len(errors)
n_unc   = sum(1 for r in raw_results if r.is_uncertain)

print(f"API errors  : {n_err}")
print(f"Uncertain   : {n_unc}")

# ── Predictions for metrics ───────────────────────────────────────────────
pred_labels = []
for r in raw_results:
    if r.intent == "uncertain" and not r.error:
        pred_labels.append("general_complaint_or_feedback")
    elif r.error:
        pred_labels.append("general_complaint_or_feedback")
    else:
        pred_labels.append(r.intent)

confidences  = [r.confidence for r in raw_results]
conf_arr     = np.array(confidences)
correct_arr  = np.array([t == p for t, p in zip(true_labels, pred_labels)])

# ── Metrics ───────────────────────────────────────────────────────────────
all_labels   = sorted(set(true_labels + pred_labels))
acc          = accuracy_score(true_labels, pred_labels)
majority     = Counter(true_labels).most_common(1)[0][0]
baseline_acc = accuracy_score(true_labels, [majority] * len(true_labels))
report_dict  = classification_report(
    true_labels, pred_labels, labels=all_labels,
    output_dict=True, zero_division=0,
)
report_str   = classification_report(
    true_labels, pred_labels, labels=all_labels,
    zero_division=0,
)

# ── Corrected accuracy (exclude error tweets) ──────────────────────────────
answered_mask = np.array([not r.error for r in raw_results])
answered_acc  = float(correct_arr[answered_mask].mean()) if answered_mask.any() else 0.0

# ── Confidence calibration ────────────────────────────────────────────────
quartile_edges = np.percentile(conf_arr, [0, 25, 50, 75, 100])
calibration = []
for i in range(4):
    lo, hi = quartile_edges[i], quartile_edges[i + 1]
    mask = (conf_arr >= lo) & (conf_arr <= hi)
    if mask.sum() == 0:
        continue
    calibration.append({
        "quartile": i + 1,
        "conf_range": [round(lo, 3), round(hi, 3)],
        "n": int(mask.sum()),
        "mean_confidence": round(float(conf_arr[mask].mean()), 3),
        "accuracy": round(float(correct_arr[mask].mean()), 3),
    })

# ── Confusion pairs ───────────────────────────────────────────────────────
cm   = confusion_matrix(true_labels, pred_labels, labels=all_labels)
pairs = []
for i, tl in enumerate(all_labels):
    for j, pl in enumerate(all_labels):
        if i != j and cm[i, j] > 0:
            pairs.append((tl, pl, int(cm[i, j])))
pairs.sort(key=lambda x: -x[2])

uncertain_ex = [
    {"tweet": tweets[i][:120], "confidence": confidences[i], "true_label": true_labels[i]}
    for i, r in enumerate(raw_results) if r.is_uncertain
]
error_ex = [
    {"tweet": tweets[i][:80], "error": r.error}
    for i, r in enumerate(raw_results) if r.error
]

# ── Build report ──────────────────────────────────────────────────────────
report = {
    "mode": MODEL_ID,
    "n_samples": len(records),
    "elapsed_seconds": round(elapsed, 2),
    "accuracy": round(acc, 4),
    "corrected_accuracy_excl_errors": round(answered_acc, 4),
    "majority_baseline_accuracy": round(baseline_acc, 4),
    "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
    "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
    "n_uncertain": n_unc,
    "n_errors": n_err,
    "avg_confidence": round(float(conf_arr.mean()), 3),
    "median_confidence": round(float(np.median(conf_arr)), 3),
    "per_class_metrics": {
        lbl: {
            "precision": round(report_dict[lbl]["precision"], 3),
            "recall":    round(report_dict[lbl]["recall"], 3),
            "f1":        round(report_dict[lbl]["f1-score"], 3),
            "support":   int(report_dict[lbl]["support"]),
        }
        for lbl in all_labels if lbl in report_dict
    },
    "confidence_calibration": calibration,
    "worst_confusions": [
        {"true": t, "predicted": p, "count": c}
        for t, p, c in pairs[:10]
    ],
    "uncertain_examples": uncertain_ex[:10],
    "error_examples": error_ex[:5],
}

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ── Human-readable text ───────────────────────────────────────────────────
W = 72
lines = [
    "=" * W,
    "INTENT CLASSIFIER — CORRECTED EVALUATION REPORT",
    "=" * W,
    f"Mode             : {report['mode']}",
    f"Eval set size    : {report['n_samples']}  (10 per intent × 11 intents)",
    f"Retries          : 5 attempts, 503-specific backoff",
    "",
    f"Accuracy (full)      : {report['accuracy']:.1%}",
    f"Accuracy (excl err)  : {report['corrected_accuracy_excl_errors']:.1%}",
    f"Majority baseline    : {report['majority_baseline_accuracy']:.1%}",
    f"Macro F1             : {report['macro_f1']:.3f}",
    f"Weighted F1          : {report['weighted_f1']:.3f}",
    f"Avg confidence       : {report['avg_confidence']:.3f}",
    f"Uncertain preds      : {report['n_uncertain']} / {report['n_samples']}",
    f"API errors           : {report['n_errors']}",
    f"Wall time            : {report['elapsed_seconds']:.1f}s",
    "",
    "─" * W,
    "PER-CLASS METRICS",
    "─" * W,
    report_str,
    "─" * W,
    "CONFIDENCE CALIBRATION",
    "─" * W,
    f"{'Quartile':<10} {'Conf range':<18} {'N':>4} {'Mean conf':>10} {'Accuracy':>10}",
]
for cal in calibration:
    lines.append(
        f"Q{cal['quartile']:<9} [{cal['conf_range'][0]:.2f}, {cal['conf_range'][1]:.2f}]"
        f"    {cal['n']:>4} {cal['mean_confidence']:>10.3f} {cal['accuracy']:>10.1%}"
    )

if pairs:
    lines += ["", "─" * W, "WORST CONFUSIONS (true → predicted, count)", "─" * W]
    for t, p, c in pairs[:6]:
        lines.append(f"  {t:35s} → {p:35s} ({c}×)")

if uncertain_ex:
    lines += ["", "─" * W, "UNCERTAIN PREDICTIONS (sample)", "─" * W]
    for ex in uncertain_ex[:5]:
        lines.append(f"  [conf={ex['confidence']:.2f}] true={ex['true_label']}")
        lines.append(f"  {ex['tweet'][:100]}")

if error_ex:
    lines += ["", "─" * W, f"REMAINING API ERRORS ({n_err})", "─" * W]
    for ex in error_ex[:3]:
        lines.append(f"  {ex['tweet'][:80]}")
        lines.append(f"  ERROR: {ex['error'][:80]}")

lines.append("=" * W)

summary = "\n".join(lines)
print("\n" + summary)

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\nSaved → {REPORT_JSON}")
print(f"Saved → {REPORT_TXT}")
