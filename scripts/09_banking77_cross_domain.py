"""
09_banking77_cross_domain.py
============================
Phase 2: cross-domain generalization check.

Runs the unchanged Amazon-Twitter classifier against the 130-query
Banking77 sample from 08_banking77_setup.py (10 queries × 13 mapped
Banking77 intents). Reports how well a Twitter-trained taxonomy
generalizes to clean, differently-worded banking queries.

Key spotlight: card_arrival (→ order_status_inquiry) vs
card_delivery_estimate (→ shipping_delay_complaint) — the exact same
hard disambiguation pair as in Phase 2-Revised, expressed in banking
language.

Usage
-----
    $env:NVIDIA_API_KEY = 'nvapi-...'
    python scripts/09_banking77_cross_domain.py

Outputs
-------
    results/banking77_cross_domain_report.json
    results/banking77_cross_domain_report.txt
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from classifier import classify_batch, MODEL_ID, MIN_INTERVAL, CONFIDENCE_THRESHOLD
from taxonomy import INTENTS

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLE_PATH = os.path.join(DATA_DIR, "banking77_cross_domain_sample.jsonl")
MAPPING_PATH = os.path.join(DATA_DIR, "banking77_mapping.json")
REPORT_JSON = os.path.join(RESULTS_DIR, "banking77_cross_domain_report.json")
REPORT_TXT  = os.path.join(RESULTS_DIR, "banking77_cross_domain_report.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load sample ────────────────────────────────────────────────────────────
if not os.path.exists(SAMPLE_PATH):
    sys.exit("ERROR: Run scripts/08_banking77_setup.py first to build the sample.")

rows = [json.loads(l) for l in open(SAMPLE_PATH, encoding="utf-8") if l.strip()]
texts          = [r["text"] for r in rows]
b77_intents    = [r["banking77_intent"] for r in rows]
expected_labels = [r["expected_amazon_label"] for r in rows]

print(f"Loaded {len(rows)} Banking77 cross-domain queries")
print(f"Model     : {MODEL_ID}")
print(f"Throttle  : {MIN_INTERVAL}s between calls, max_workers=2")
print()

# ── Classify ───────────────────────────────────────────────────────────────
t0 = time.perf_counter()
raw_results = classify_batch(texts, max_workers=2, progress=True)
elapsed = time.perf_counter() - t0
print(f"\nDone in {elapsed:.1f}s  ({elapsed / len(texts) * 1000:.0f}ms/query avg)")

pred_labels  = []
confidences  = []
for r in raw_results:
    if r.error or r.intent == "uncertain":
        pred_labels.append("general_complaint_or_feedback")
    else:
        pred_labels.append(r.intent)
    confidences.append(r.confidence)

n_errors   = sum(1 for r in raw_results if r.error)
n_uncertain = sum(1 for r in raw_results if r.is_uncertain)
conf_arr   = np.array(confidences)
correct_arr = np.array([e == p for e, p in zip(expected_labels, pred_labels)])

# ── Global metrics ─────────────────────────────────────────────────────────
all_labels = sorted(set(expected_labels + pred_labels))
acc        = accuracy_score(expected_labels, pred_labels)
report_dict = classification_report(
    expected_labels, pred_labels, labels=all_labels,
    output_dict=True, zero_division=0,
)
report_str = classification_report(
    expected_labels, pred_labels, labels=all_labels, zero_division=0,
)

# ── Hard-pair spotlight ────────────────────────────────────────────────────
# order_status_inquiry queries (card_arrival, pending_transfer, transfer_timing)
# vs shipping_delay_complaint queries (card_delivery_estimate, transfer_not_received, failed_transfer)
osi_mask  = np.array([e == "order_status_inquiry" for e in expected_labels])
sdc_mask  = np.array([e == "shipping_delay_complaint" for e in expected_labels])

hard_pair_rows = []
for i, (row, pred, exp, conf) in enumerate(zip(rows, pred_labels, expected_labels, confidences)):
    if exp in ("order_status_inquiry", "shipping_delay_complaint"):
        hard_pair_rows.append({
            "text": row["text"][:120],
            "banking77_intent": row["banking77_intent"],
            "expected": exp,
            "predicted": pred,
            "confidence": conf,
            "correct": exp == pred,
        })

hard_pair_acc = float(correct_arr[osi_mask | sdc_mask].mean()) if (osi_mask | sdc_mask).any() else 0.0

# ── Worst confusions ───────────────────────────────────────────────────────
cm = confusion_matrix(expected_labels, pred_labels, labels=all_labels)
confusions = []
for i, tl in enumerate(all_labels):
    for j, pl in enumerate(all_labels):
        if i != j and cm[i, j] > 0:
            confusions.append({"true": tl, "predicted": pl, "count": int(cm[i, j])})
confusions.sort(key=lambda x: -x["count"])

# ── In-domain comparison numbers (from Run 4) ─────────────────────────────
IN_DOMAIN = {
    "order_status_inquiry":    {"f1": 0.27, "precision": 0.40, "recall": 0.20},
    "shipping_delay_complaint":{"f1": 0.60, "precision": 0.45, "recall": 0.90},
    "account_access":          {"f1": 0.84, "precision": 0.89, "recall": 0.80},
    "refund_request":          {"f1": 0.59, "precision": 0.71, "recall": 0.50},
}

# ── Build report ──────────────────────────────────────────────────────────
report = {
    "description": (
        "Cross-domain generalization: Amazon-Twitter classifier on Banking77 queries. "
        "13 Banking77 intents mapped to 7 Amazon taxonomy labels. "
        "10 queries per Banking77 intent = 130 total queries."
    ),
    "model": MODEL_ID,
    "n_queries": len(rows),
    "n_errors": n_errors,
    "n_uncertain": n_uncertain,
    "elapsed_seconds": round(elapsed, 2),
    "overall_accuracy": round(acc, 4),
    "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
    "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
    "avg_confidence": round(float(conf_arr.mean()), 3),
    "hard_pair_accuracy": round(hard_pair_acc, 4),
    "per_class_metrics": {
        lbl: {
            "precision": round(report_dict[lbl]["precision"], 3),
            "recall":    round(report_dict[lbl]["recall"], 3),
            "f1":        round(report_dict[lbl]["f1-score"], 3),
            "support":   int(report_dict[lbl]["support"]),
            "in_domain_f1": IN_DOMAIN.get(lbl, {}).get("f1"),
            "f1_gap":       round(
                report_dict[lbl]["f1-score"] - IN_DOMAIN.get(lbl, {}).get("f1", report_dict[lbl]["f1-score"]),
                3
            ) if lbl in IN_DOMAIN else None,
        }
        for lbl in all_labels if lbl in report_dict
    },
    "worst_confusions": confusions[:8],
    "hard_pair_detail": hard_pair_rows,
}

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ── Human-readable report ─────────────────────────────────────────────────
W = 72
lines = [
    "=" * W,
    "BANKING77 CROSS-DOMAIN GENERALIZATION REPORT",
    "=" * W,
    f"Model           : {report['model']}",
    f"Queries         : {report['n_queries']}  "
    f"(10 per Banking77 intent × 13 mapped intents)",
    f"API errors      : {report['n_errors']}",
    "",
    f"Overall Accuracy: {report['overall_accuracy']:.1%}",
    f"Macro F1        : {report['macro_f1']:.3f}",
    f"Avg Confidence  : {report['avg_confidence']:.3f}",
    f"Hard-pair Acc   : {report['hard_pair_accuracy']:.1%}  "
    "(order_status_inquiry + shipping_delay_complaint queries)",
    "",
    "─" * W,
    "PER-CLASS — cross-domain vs in-domain (Amazon Twitter) F1",
    "─" * W,
    f"{'Amazon Label':<35s} {'XD-F1':>6} {'ID-F1':>6} {'Gap':>6} {'Support':>8}",
]
for lbl, m in report["per_class_metrics"].items():
    id_f1 = f"{m['in_domain_f1']:.2f}" if m["in_domain_f1"] is not None else "  n/a"
    gap   = f"{m['f1_gap']:+.2f}"      if m["f1_gap"]       is not None else "  n/a"
    lines.append(
        f"{lbl:<35s} {m['f1']:>6.2f} {id_f1:>6} {gap:>6} {m['support']:>8d}"
    )

lines += [
    "",
    "─" * W,
    "HARD-PAIR SPOTLIGHT  (order_status_inquiry vs shipping_delay_complaint)",
    "─" * W,
    "Banking77 hard-pair analogs:",
    "  card_arrival / pending_transfer / transfer_timing",
    "    → expected: order_status_inquiry",
    "  card_delivery_estimate / transfer_not_received_by_recipient / failed_transfer",
    "    → expected: shipping_delay_complaint",
    "",
]
hp_correct = [r for r in hard_pair_rows if r["correct"]]
hp_wrong   = [r for r in hard_pair_rows if not r["correct"]]
lines.append(f"Hard-pair accuracy: {len(hp_correct)}/{len(hard_pair_rows)} = {hard_pair_acc:.1%}")
lines.append(f"Misclassified     : {len(hp_wrong)}")
if hp_wrong:
    lines.append("")
    lines.append("Sample misclassifications:")
    for r in hp_wrong[:4]:
        lines.append(f"  [{r['banking77_intent']}]")
        lines.append(f"  Text     : {r['text'][:90]}")
        lines.append(f"  Expected : {r['expected']}")
        lines.append(f"  Got      : {r['predicted']}  (conf={r['confidence']:.2f})")
        lines.append("")

lines += [
    "─" * W,
    "WORST CONFUSIONS (true → predicted, count)",
    "─" * W,
]
for c in confusions[:6]:
    lines.append(f"  {c['true']:35s} → {c['predicted']:35s} ({c['count']}×)")

lines.append("")
lines.append("─" * W)
lines.append("INTERPRETATION")
lines.append("─" * W)
lines.append(
    "Cross-domain accuracy measures how well a Twitter-trained Amazon taxonomy\n"
    "generalises to Banking77's clean, differently-worded queries.\n"
    "Expected gap: 10-20pp below in-domain (different register + domain).\n"
    "The hard pair (order_status vs shipping_delay) is the most informative\n"
    "sub-test: Banking77 provides domain-agnostic evidence for this confusion."
)
lines.append("=" * W)

summary = "\n".join(lines)
print("\n" + summary)
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\nSaved → {REPORT_JSON}")
print(f"Saved → {REPORT_TXT}")
print("\nRun next:")
print("  python scripts/10_before_after_hard_pairs.py")
