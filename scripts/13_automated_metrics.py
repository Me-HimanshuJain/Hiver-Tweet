"""
13_automated_metrics.py — Compute automated evaluation metrics.

Metrics:
    1. Intent accuracy / macro-F1 / weighted-F1 / per-class P-R-F1
    2. Retrieval hit-rate @ 1, 3, 5  (FAISS index, intent-filtered)
    3. Escalation precision / recall / F1  (rule engine vs human_escalate)
    4. Banking77 cross-domain generalization (supplementary, from existing report)

Input priority:
    Primary  : data/golden_labeled.csv   (user fills in golden_label_sheet.csv)
    Fallback : data/labelled_eval.jsonl  (existing 110-example auto-labeled set)

Run:
    python scripts/13_automated_metrics.py [--golden]  # force golden set
    python scripts/13_automated_metrics.py [--fallback] # force fallback set

Output:
    results/automated_metrics.json
    results/automated_metrics.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
GOLDEN_CSV  = os.path.join(DATA_DIR, "golden_labeled.csv")
FALLBACK    = os.path.join(DATA_DIR, "labelled_eval.jsonl")
B77_REPORT  = os.path.join(RESULTS_DIR, "banking77_cross_domain_report.json")
OUT_JSON    = os.path.join(RESULTS_DIR, "automated_metrics.json")
OUT_TXT     = os.path.join(RESULTS_DIR, "automated_metrics.txt")

INTENTS_LIST = [
    "account_access", "delivery_issue", "device_and_digital_support",
    "general_complaint_or_feedback", "order_cancellation", "order_status_inquiry",
    "prime_membership", "product_issue", "refund_request", "return_or_exchange",
    "shipping_delay_complaint",
]


# ── Data loading ──────────────────────────────────────────────────────────

def load_data(force_golden: bool = False, force_fallback: bool = False):
    """Returns (tweets, labels, escalate_labels, source_name)."""
    import pandas as pd

    use_golden = (os.path.exists(GOLDEN_CSV) and not force_fallback) or force_golden

    if use_golden and os.path.exists(GOLDEN_CSV):
        df = pd.read_csv(GOLDEN_CSV)
        required = {"tweet", "human_label"}
        if not required.issubset(df.columns):
            raise ValueError(f"golden_labeled.csv must have columns {required}, got {list(df.columns)}")
        df = df[df["human_label"].notna() & (df["human_label"].str.strip() != "")]
        tweets    = df["tweet"].tolist()
        labels    = df["human_label"].str.strip().tolist()
        escalates = df["human_escalate"].tolist() if "human_escalate" in df.columns else None
        return tweets, labels, escalates, f"golden_labeled.csv ({len(tweets)} examples)"

    # Fallback
    if not os.path.exists(FALLBACK):
        raise FileNotFoundError(f"No labeled data found: neither {GOLDEN_CSV} nor {FALLBACK}")
    rows = []
    with open(FALLBACK, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    tweets    = [r["tweet"] for r in rows]
    labels    = [r["label"] for r in rows]
    return tweets, labels, None, f"labelled_eval.jsonl ({len(tweets)} examples, auto-labeled fallback)"


# ── Metric helpers ────────────────────────────────────────────────────────

def compute_classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]):
    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, f1_score)
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report    = classification_report(y_true, y_pred, labels=labels,
                                      output_dict=True, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, macro_f1, weighted, report, cm, labels


def retrieval_hit_rate(tweets: list[str], labels: list[str], top_ks=(1, 3, 5)) -> dict:
    """
    For each tweet, query the FAISS index and check whether any of the
    top-K retrieved results has the same intent as the ground-truth label.
    Returns hit-rate @K for K in top_ks.
    """
    print("  Loading retrieval index…")
    from retrieval import RetrievalIndex
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
    META_PATH  = os.path.join(DATA_DIR, "faiss_meta.jsonl")
    idx = RetrievalIndex.load(INDEX_PATH, META_PATH)

    hits = {k: 0 for k in top_ks}
    max_k = max(top_ks)

    print(f"  Running retrieval for {len(tweets)} tweets (top_{max_k})…")
    for tweet, label in zip(tweets, labels):
        results = idx.query(tweet, top_k=max_k)
        retrieved_intents = [r.intent for r in results]
        for k in top_ks:
            if label in retrieved_intents[:k]:
                hits[k] += 1

    n = len(tweets)
    return {f"hit@{k}": round(hits[k] / n, 4) for k in top_ks}


def escalation_metrics(tweets: list[str], labels: list[str],
                        escalate_gt: Optional[list]) -> Optional[dict]:
    """
    Run escalation engine; compare to human_escalate ground truth.
    Returns None if ground truth is not available.
    """
    if escalate_gt is None:
        return None

    # Parse ground truth: Y → 1, N → 0; skip rows without value
    from escalation import decide, EscalationDecision
    from classifier import ClassificationResult

    valid_idx = []
    gt_binary  = []
    for i, val in enumerate(escalate_gt):
        if isinstance(val, str) and val.strip().upper() in ("Y", "N"):
            valid_idx.append(i)
            gt_binary.append(1 if val.strip().upper() == "Y" else 0)

    if not valid_idx:
        return None

    print(f"  Running escalation engine on {len(valid_idx)} labelled examples…")
    pred_binary = []
    for i in valid_idx:
        # Build a minimal ClassificationResult for the escalation engine
        clf = ClassificationResult(
            tweet=tweets[i], intent=labels[i], confidence=0.85,
            reasoning="", alternatives=[], raw_response="", latency_ms=0
        )
        decision = decide(tweets[i], clf)
        pred_binary.append(1 if decision.decision == "escalate" else 0)

    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(gt_binary, pred_binary, zero_division=0)
    r = recall_score(gt_binary, pred_binary, zero_division=0)
    f = f1_score(gt_binary, pred_binary, zero_division=0)
    tp = sum(g == 1 and p_ == 1 for g, p_ in zip(gt_binary, pred_binary))
    fp = sum(g == 0 and p_ == 1 for g, p_ in zip(gt_binary, pred_binary))
    fn = sum(g == 1 and p_ == 0 for g, p_ in zip(gt_binary, pred_binary))
    tn = sum(g == 0 and p_ == 0 for g, p_ in zip(gt_binary, pred_binary))
    return {
        "n_evaluated": len(valid_idx),
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def banking77_summary() -> Optional[dict]:
    """Load the Banking77 cross-domain report for the supplementary table."""
    if not os.path.exists(B77_REPORT):
        return None
    with open(B77_REPORT, encoding="utf-8") as f:
        report = json.load(f)
    # Extract key numbers
    summary = {
        "description":            report.get("description", ""),
        "n_queries":              report.get("n_queries"),
        "overall_accuracy":       report.get("overall_accuracy"),
        "macro_f1":               report.get("macro_f1"),
        "weighted_f1":            report.get("weighted_f1"),
        "avg_confidence":         report.get("avg_confidence"),
        "hard_pair_accuracy":     report.get("hard_pair_accuracy"),
        "per_class_f1_gaps": {
            intent: m.get("f1_gap")
            for intent, m in report.get("per_class_metrics", {}).items()
            if m.get("f1_gap") is not None
        },
    }
    return summary


# ── Run classification metrics using existing labelled_eval predictions ───

def run_intent_metrics(tweets: list[str], labels: list[str], source: str):
    """
    Uses keyword heuristics to predict on the fallback set (matching how the
    original labelled_eval was created) OR re-runs the classifier if golden set
    is loaded.  To keep this script fast and API-free, keyword prediction is
    always used for the baseline comparison; the full classifier numbers come
    from the existing results/eval_report.json.
    """
    import re as _re
    PATTERN_MAP = [
        ("order_status_inquiry",
         r"track|where is my order|where.s my package|shipping status|how long|"
         r"when will|estimated delivery|hasn.t arrived|still waiting"),
        ("delivery_issue",
         r"delivered to wrong|wrong address|someone else.s|wrong house|"
         r"left.*unsafe|not received.*marked|says delivered|delivered incorrectly|driver"),
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
        ("general_complaint_or_feedback", None),
    ]

    def kw_predict(text):
        t = text.lower()
        for intent, pattern in PATTERN_MAP:
            if pattern is None:
                return intent
            if _re.search(pattern, t):
                return intent
        return "general_complaint_or_feedback"

    y_pred = [kw_predict(t) for t in tweets]
    return compute_classification_metrics(labels, y_pred, INTENTS_LIST), y_pred


# ── Formatting ────────────────────────────────────────────────────────────

def format_report(results: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("AUTOMATED EVALUATION METRICS")
    lines.append("=" * 72)
    lines.append(f"Eval set : {results['source']}")
    lines.append(f"Run time : {results['timestamp']}")
    lines.append("")

    # Intent
    ci = results["intent"]
    lines.append("─" * 72)
    lines.append("INTENT CLASSIFICATION (keyword baseline — see also results/eval_report.json)")
    lines.append("─" * 72)
    lines.append(f"  Accuracy      : {ci['accuracy']:.3f}")
    lines.append(f"  Macro F1      : {ci['macro_f1']:.3f}")
    lines.append(f"  Weighted F1   : {ci['weighted_f1']:.3f}")
    lines.append("")
    lines.append(f"  {'Intent':<42s}  {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>5}")
    lines.append(f"  {'─'*42}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}")
    for intent, m in sorted(ci["per_class"].items()):
        lines.append(
            f"  {intent:<42s}  {m['precision']:6.3f}  {m['recall']:6.3f}"
            f"  {m['f1']:6.3f}  {m['support']:5d}"
        )
    lines.append("")

    # Retrieval
    r = results["retrieval"]
    lines.append("─" * 72)
    lines.append("RETRIEVAL HIT-RATE  (same-intent match in top-K results)")
    lines.append("─" * 72)
    for k, v in sorted(r.items()):
        lines.append(f"  {k:<10s}: {v:.3f}")
    lines.append("")

    # Escalation
    esc = results.get("escalation")
    lines.append("─" * 72)
    lines.append("ESCALATION PRECISION / RECALL / F1")
    lines.append("─" * 72)
    if esc is None:
        lines.append("  Not available — human_escalate column not found in eval set.")
        lines.append("  Fill in golden_label_sheet.csv and re-run.")
    else:
        lines.append(f"  Evaluated on   : {esc['n_evaluated']} examples")
        lines.append(f"  Precision      : {esc['precision']:.3f}")
        lines.append(f"  Recall         : {esc['recall']:.3f}")
        lines.append(f"  F1             : {esc['f1']:.3f}")
        lines.append(f"  TP/FP/FN/TN    : {esc['tp']}/{esc['fp']}/{esc['fn']}/{esc['tn']}")
    lines.append("")

    # Banking77
    b77 = results.get("banking77_cross_domain")
    lines.append("─" * 72)
    lines.append("BANKING77 CROSS-DOMAIN GENERALIZATION  (supplementary)")
    lines.append("─" * 72)
    if b77 is None:
        lines.append("  results/banking77_cross_domain_report.json not found.")
    else:
        lines.append(f"  Queries        : {b77['n_queries']}")
        lines.append(f"  Accuracy       : {b77['overall_accuracy']:.3f}"
                     f"  (vs {results['intent']['accuracy']:.3f} in-domain)")
        lines.append(f"  Macro F1       : {b77['macro_f1']:.3f}")
        lines.append(f"  Avg confidence : {b77['avg_confidence']:.3f}")
        lines.append(f"  Hard-pair acc  : {b77['hard_pair_accuracy']:.3f}")
        lines.append("")
        lines.append("  Per-class F1 gap (in-domain minus cross-domain):")
        for intent, gap in sorted(b77["per_class_f1_gaps"].items()):
            bar = "█" * max(0, int(abs(gap) * 20))
            lines.append(f"    {intent:<42s}  Δ={gap:+.3f}  {bar}")
        lines.append("")
        lines.append("  Interpretation: shipping_delay_complaint collapses cross-domain")
        lines.append("  (keyword reliance), while account_access is genuinely semantic.")
    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden",   action="store_true")
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    # Load data
    print("Loading evaluation data…")
    tweets, labels, escalate_gt, source = load_data(args.golden, args.fallback)
    print(f"  Source: {source}")

    # Intent metrics (keyword baseline — fast, no API)
    print("Computing intent metrics (keyword predictor)…")
    (acc, macro_f1, weighted, report, cm, label_list), y_pred = \
        run_intent_metrics(tweets, labels, source)

    per_class = {}
    for intent in INTENTS_LIST:
        m = report.get(intent, {})
        per_class[intent] = {
            "precision": round(m.get("precision", 0), 4),
            "recall":    round(m.get("recall", 0),    4),
            "f1":        round(m.get("f1-score", 0),  4),
            "support":   int(m.get("support", 0)),
        }

    intent_results = {
        "accuracy":    round(acc, 4),
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted, 4),
        "per_class":   per_class,
        "n_samples":   len(tweets),
    }

    # Retrieval hit-rate
    print("Computing retrieval hit-rate…")
    retrieval = retrieval_hit_rate(tweets, labels)

    # Escalation (only when ground truth available)
    print("Computing escalation metrics…")
    escalation = escalation_metrics(tweets, labels, escalate_gt)

    # Banking77 supplement
    print("Loading Banking77 cross-domain numbers…")
    b77 = banking77_summary()

    # Assemble result
    from datetime import datetime
    results = {
        "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source":                  source,
        "intent":                  intent_results,
        "retrieval":               retrieval,
        "escalation":              escalation,
        "banking77_cross_domain":  b77,
    }

    # Write outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    report_txt = format_report(results)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print()
    print(report_txt)
    print(f"Saved → {OUT_JSON}")
    print(f"Saved → {OUT_TXT}")


if __name__ == "__main__":
    main()
