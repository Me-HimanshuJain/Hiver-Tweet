"""
06_evaluate.py — Run the intent classifier on the labelled eval set and
produce an accuracy / F1 report.

Usage
-----
Real Claude Haiku API (costs ~$0.05 for 110 tweets):
    $env:ANTHROPIC_API_KEY = 'sk-ant-...'   # PowerShell
    python scripts/06_evaluate.py

Dry-run / mock mode (no API key needed — uses keyword heuristics as predictions):
    python scripts/06_evaluate.py --mock

Output:
    results/eval_report.json   — full metrics
    results/eval_report.txt    — human-readable summary (also printed to stdout)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from taxonomy import INTENTS

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
EVAL_PATH   = os.path.join(DATA_DIR, "labelled_eval.jsonl")
REPORT_JSON = os.path.join(RESULTS_DIR, "eval_report.json")
REPORT_TXT  = os.path.join(RESULTS_DIR, "eval_report.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mock", action="store_true",
    help="Use keyword heuristic predictions (no API calls needed)."
)
args = parser.parse_args()

# ── Load eval set ─────────────────────────────────────────────────────────
if not os.path.exists(EVAL_PATH):
    print(f"Eval set not found: {EVAL_PATH}")
    print("Run scripts/05_label_sample.py first.")
    sys.exit(1)

records = []
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

tweets     = [r["tweet"]  for r in records]
true_labels = [r["label"] for r in records]

print(f"Loaded {len(records)} labelled tweets")
print(f"Intent distribution:")
for intent, n in sorted(Counter(true_labels).items(), key=lambda x: -x[1]):
    print(f"  {intent:40s} {n}")

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# PREDICTION BLOCK
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

@dataclass
class _Result:
    intent: str
    confidence: float
    reasoning: str
    error: Optional[str] = None

    @property
    def is_uncertain(self) -> bool:
        return self.confidence < 0.50 or self.intent == "uncertain"


if args.mock:
    # ── MOCK MODE ─────────────────────────────────────────────────────────
    print("\n[MOCK MODE] Using keyword heuristic predictions (no API calls)")

    PATTERN_MAP = [
        ("order_status_inquiry",
         r"track|where is my order|where.s my (package|parcel)|how long|when will|estimated delivery"),
        ("delivery_issue",
         r"delivered to wrong|wrong address|someone else.s|wrong house|left.*unsafe|says delivered|delivered incorrectly"),
        ("shipping_delay_complaint",
         r"2.day shipping|next.day shipping|prime shipping|paid.*shipping|shipping.*late|delayed.*shipping"),
        ("refund_request",
         r"refund|money back|reimburse|unauthori[sz]ed charge|missing refund|get my money"),
        ("order_cancellation",
         r"cancel (my )?order|how (do i|to) cancel|need to cancel|cancel.*before|stop.*order"),
        ("return_or_exchange",
         r"\breturn\b|\bexchange\b|send (it )?back|swap|return label|return (window|policy|process)|how do i return"),
        ("product_issue",
         r"defective|damaged|broken|doesn.t work|not working|wrong (item|product)|counterfeit|fake|empty box|missing (item|part)"),
        ("account_access",
         r"account.*lock|account.*block|account.*suspend|can.t (log|sign).?in|forgot.*password|reset.*password|locked out|hacked"),
        ("prime_membership",
         r"prime (member|subscription|trial|free|benefit)|membership (fee|charge|cancel|renew|cost)|cancel.*prime|prime.*price"),
        ("device_and_digital_support",
         r"kindle|fire (tablet|stick|tv|hd)|echo|alexa|prime video|streaming|app.*crash|device.*not work|digital content"),
        ("general_complaint_or_feedback", None),
    ]

    def _kw_predict(text: str) -> _Result:
        t = text.lower()
        for intent, pattern in PATTERN_MAP:
            if pattern is None:
                return _Result(intent=intent, confidence=0.60,
                               reasoning="Keyword catch-all")
            if re.search(pattern, t):
                return _Result(intent=intent, confidence=0.78,
                               reasoning=f"Keyword match: {pattern[:40]}")
        return _Result(intent="general_complaint_or_feedback", confidence=0.60,
                       reasoning="No pattern matched")

    t0 = time.perf_counter()
    results = [_kw_predict(tw) for tw in tweets]
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s (keyword heuristic, no API)\n")
    mode_label = "keyword-heuristic (mock)"

else:
    # ── REAL NIM API MODE ────────────────────────────────────────────
    from classifier import classify_batch, CONFIDENCE_THRESHOLD, MODEL_ID, MIN_INTERVAL
    print(f"\nRunning NIM classifier ({len(tweets)} API calls)...")
    print(f"Model      : {MODEL_ID}")
    print(f"Throttle   : {MIN_INTERVAL}s between calls, max_workers=2")
    t0 = time.perf_counter()
    raw_results = classify_batch(tweets, max_workers=2, progress=True)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed/len(tweets)*1000:.0f}ms/tweet avg)")

    results = [
        _Result(
            intent=r.intent,
            confidence=r.confidence,
            reasoning=r.reasoning,
            error=r.error,
        )
        for r in raw_results
    ]
    mode_label = MODEL_ID

# ── Extract predictions & confidences ────────────────────────────────────
pred_labels  = [r.intent for r in results]
confidences  = [r.confidence for r in results]
errors       = [r for r in results if r.error]

# For metric purposes: uncertain → treat as general_complaint_or_feedback
pred_for_metrics = [
    p if p != "uncertain" else "general_complaint_or_feedback"
    for p in pred_labels
]

# ── Accuracy & baselines ───────────────────────────────────────────────────
acc = accuracy_score(true_labels, pred_for_metrics)
majority = Counter(true_labels).most_common(1)[0][0]
baseline_acc = accuracy_score(true_labels, [majority] * len(true_labels))

# ── Per-class report ──────────────────────────────────────────────────────
all_labels = sorted(set(true_labels + [p for p in pred_for_metrics]))
report_dict = classification_report(
    true_labels, pred_for_metrics, labels=all_labels,
    output_dict=True, zero_division=0,
)
report_str = classification_report(
    true_labels, pred_for_metrics, labels=all_labels,
    zero_division=0,
)

# ── Confidence calibration ─────────────────────────────────────────────────
conf_arr    = np.array(confidences)
correct_arr = np.array([t == p for t, p in zip(true_labels, pred_for_metrics)])

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

# ── Confusion worst pairs ─────────────────────────────────────────────────
cm = confusion_matrix(true_labels, pred_for_metrics, labels=all_labels)
worst_pairs = []
for i, tl in enumerate(all_labels):
    for j, pl in enumerate(all_labels):
        if i != j and cm[i, j] > 0:
            worst_pairs.append((tl, pl, int(cm[i, j])))
worst_pairs.sort(key=lambda x: -x[2])

# ── Uncertain predictions ─────────────────────────────────────────────────
uncertain_ex = [
    {"tweet": tweets[i][:120], "confidence": confidences[i],
     "true_label": true_labels[i]}
    for i, r in enumerate(results) if r.is_uncertain
]

# ── Assemble report ────────────────────────────────────────────────────────
report = {
    "mode": mode_label,
    "n_samples": len(records),
    "elapsed_seconds": round(elapsed, 2),
    "accuracy": round(acc, 4),
    "majority_baseline_accuracy": round(baseline_acc, 4),
    "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
    "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
    "n_uncertain": len(uncertain_ex),
    "n_errors": len(errors),
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
        for t, p, c in worst_pairs[:10]
    ],
    "uncertain_examples": uncertain_ex[:10],
}

# ── Save JSON ─────────────────────────────────────────────────────────────
with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ── Human-readable summary ─────────────────────────────────────────────────
W = 72
lines = [
    "=" * W,
    "INTENT CLASSIFIER — EVALUATION REPORT",
    "=" * W,
    f"Mode             : {mode_label}",
    f"Eval set size    : {report['n_samples']}  (10 per intent × 11 intents)",
    f"",
    f"Accuracy         : {report['accuracy']:.1%}",
    f"Majority baseline: {report['majority_baseline_accuracy']:.1%}",
    f"Macro F1         : {report['macro_f1']:.3f}",
    f"Weighted F1      : {report['weighted_f1']:.3f}",
    f"Avg confidence   : {report['avg_confidence']:.3f}",
    f"Uncertain preds  : {report['n_uncertain']} / {report['n_samples']}",
    f"API errors       : {report['n_errors']}",
    f"Wall time        : {report['elapsed_seconds']:.2f}s",
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

if worst_pairs:
    lines += ["", "─" * W, "WORST CONFUSIONS (true → predicted, count)", "─" * W]
    for t, p, c in worst_pairs[:6]:
        lines.append(f"  {t:35s} → {p:35s} ({c}×)")

if uncertain_ex:
    lines += ["", "─" * W, "UNCERTAIN PREDICTIONS (sample)", "─" * W]
    for ex in uncertain_ex[:5]:
        lines.append(f"  [conf={ex['confidence']:.2f}] true={ex['true_label']}")
        lines.append(f"  {ex['tweet'][:100]}")

lines.append("=" * W)

summary = "\n".join(lines)
print("\n" + summary)

with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\nSaved → {REPORT_JSON}")
print(f"Saved → {REPORT_TXT}")
