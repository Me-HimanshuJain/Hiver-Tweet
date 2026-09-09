"""
15_baselines.py — Implement and evaluate two baselines against the full system.

Baseline 1 — Keyword-Match + Canned Reply  (trivial, zero API calls):
    Intent : keyword PATTERN_MAP (same as 05_label_sample.py)
    Reply  : fixed canned reply string per intent (12 variants)
    Escalation: always auto_handle (intentionally naive)

Baseline 2 — Zero-Shot NIM, No Retrieval  (simple, ~N API calls):
    Intent : same Nemotron model, but ZERO few-shot examples in the prompt
    Reply  : same Nemotron model, but NO grounding examples
    Escalation: same rule engine as full system (fair comparison — only
                isolates classifier and retrieval quality)

Full system results are loaded from existing result files; this script does
NOT re-run the full system (it's already been evaluated in scripts 06/13).

Run:
    $env:NVIDIA_API_KEY='nvapi-...'
    python scripts/15_baselines.py              # both baselines
    python scripts/15_baselines.py --b1-only    # baseline 1 only (no API)
    python scripts/15_baselines.py --n 30       # limit to N examples

Output:
    results/baseline_comparison.json
    results/baseline_comparison.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
GOLDEN_CSV  = os.path.join(DATA_DIR, "golden_labeled.csv")
FALLBACK    = os.path.join(DATA_DIR, "labelled_eval.jsonl")
EVAL_JSON   = os.path.join(RESULTS_DIR, "eval_report.json")
JUDGE_JSON  = os.path.join(RESULTS_DIR, "judge_scores.json")
OUT_JSON    = os.path.join(RESULTS_DIR, "baseline_comparison.json")
OUT_TXT     = os.path.join(RESULTS_DIR, "baseline_comparison.txt")

NIM_BASE_URL  = "https://integrate.api.nvidia.com/v1"
MODEL_ID      = "nvidia/nemotron-3-super-120b-a12b"
MAX_EVAL      = 110  # cap to match existing eval set size

# ── Keyword patterns (same as 05_label_sample.py / 12_sample_golden_set.py) ─

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


def keyword_predict(text: str) -> str:
    t = text.lower()
    for intent, pattern in PATTERN_MAP:
        if pattern is None:
            return intent
        if re.search(pattern, t):
            return intent
    return "general_complaint_or_feedback"


# ── Canned replies per intent (Baseline 1) ─────────────────────────────────

CANNED_REPLIES: dict[str, str] = {
    "order_status_inquiry":
        "Hi! We'd love to help track your order. Please DM us your order number "
        "and we'll look into the status right away. ^AH",
    "delivery_issue":
        "We're sorry your delivery didn't go as expected! Please DM us your "
        "order number so we can investigate and arrange a resolution. ^AH",
    "shipping_delay_complaint":
        "We apologise for the shipping delay. Please DM us your order number "
        "and we'll review what happened and make it right. ^AH",
    "refund_request":
        "We're sorry to hear that! Please DM us your order number and we'll "
        "look into your refund as quickly as possible. ^AH",
    "order_cancellation":
        "We can help with your cancellation request. Please DM us your order "
        "number and the reason and we'll take care of it. ^AH",
    "return_or_exchange":
        "We'd be happy to help with your return or exchange. Please DM us your "
        "order number and we'll send you the return instructions. ^AH",
    "product_issue":
        "We're sorry you received a damaged or incorrect item! Please DM us "
        "your order number and a photo if possible, and we'll resolve this. ^AH",
    "account_access":
        "We're sorry you're having trouble accessing your account. Please DM "
        "us your registered email (not your password) and we'll help. ^AH",
    "prime_membership":
        "We can help with your Prime membership query. Please DM us your "
        "account email and the details, and we'll sort this out. ^AH",
    "device_and_digital_support":
        "We're sorry your device or digital service isn't working. Please DM "
        "us your device model and order number and we'll troubleshoot. ^AH",
    "general_complaint_or_feedback":
        "We're really sorry to hear about your experience. Please DM us the "
        "details and we'll investigate and get back to you. ^AH",
}


# ── Zero-shot NIM classifier (Baseline 2) ─────────────────────────────────

_INTENTS_BLOCK = "\n".join(
    f"  {i+1:2d}. {name}"
    for i, name in enumerate([p[0] for p in PATTERN_MAP])
)

_ZS_SYSTEM = (
    "You are a customer service intent classifier for Amazon.\n"
    "Classify the tweet into exactly one of these 11 intents:\n"
    f"{_INTENTS_BLOCK}\n\n"
    'Output JSON: {"intent": "<label>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}'
)


def zero_shot_classify(tweet: str, client) -> dict:
    from openai import RateLimitError, APIStatusError
    from classifier import _throttle

    for attempt in range(5):
        _throttle()
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL_ID,
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _ZS_SYSTEM},
                    {"role": "user",   "content": tweet},
                ],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = (resp.choices[0].message.content or "").strip()
            obj = json.loads(raw)
            return {
                "intent":     obj.get("intent", "general_complaint_or_feedback"),
                "confidence": float(obj.get("confidence", 0.5)),
                "latency_ms": latency_ms,
            }
        except (RateLimitError, APIStatusError) as e:
            time.sleep(8 * (attempt + 1))
        except Exception as e:
            return {"intent": "general_complaint_or_feedback",
                    "confidence": 0.0, "error": str(e)[:80]}
    return {"intent": "general_complaint_or_feedback", "confidence": 0.0,
            "error": "max retries"}


_ZS_DRAFT_SYSTEM = (
    "You are an Amazon customer support agent. Write an empathetic, specific "
    "reply with a concrete next step. Do not copy examples verbatim. "
    'Output JSON: {"reply": "<reply text>", "tone": "empathetic|informational|apologetic|positive", '
    '"action_taken": "<one phrase>"}'
)


def zero_shot_draft(tweet: str, intent: str, client) -> str:
    from openai import RateLimitError, APIStatusError
    from classifier import _throttle
    from drafter import _parse_plain_reply

    user_msg = f"Customer intent: {intent}\nCustomer tweet: {tweet}"
    for attempt in range(4):
        _throttle()
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                temperature=0.3,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _ZS_DRAFT_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            try:
                parsed = _parse_plain_reply(raw)
                return parsed["reply"]
            except ValueError:
                pass
        except Exception:
            time.sleep(5 * (attempt + 1))
    return CANNED_REPLIES.get(intent, CANNED_REPLIES["general_complaint_or_feedback"])


# ── Metric helpers ────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    return acc, macro_f1, weighted, report


def reply_length_stats(replies):
    import numpy as np
    lens = [len(r) for r in replies]
    return {
        "mean": round(float(np.mean(lens)), 1),
        "median": round(float(np.median(lens)), 1),
        "max": max(lens),
        "over_280": sum(l > 280 for l in lens),
    }


# ── Load eval data ─────────────────────────────────────────────────────────

def load_eval_data(n: int):
    if os.path.exists(GOLDEN_CSV):
        import pandas as pd
        df = pd.read_csv(GOLDEN_CSV)
        df = df[df["human_label"].notna() & (df["human_label"].str.strip() != "")]
        tweets = df["tweet"].tolist()[:n]
        labels = df["human_label"].str.strip().tolist()[:n]
        return tweets, labels, "golden_labeled.csv"
    rows = [json.loads(l) for l in open(FALLBACK, encoding="utf-8")][:n]
    return [r["tweet"] for r in rows], [r["label"] for r in rows], "labelled_eval.jsonl"


# ── Full system numbers (from existing result files) ───────────────────────

def load_full_system_numbers() -> dict:
    result = {
        "intent_accuracy": None, "intent_macro_f1": None,
        "intent_weighted_f1": None, "note": "",
    }
    if os.path.exists(EVAL_JSON):
        with open(EVAL_JSON, encoding="utf-8") as f:
            ev = json.load(f)
        result.update({
            "intent_accuracy":    ev.get("accuracy"),
            "intent_macro_f1":    ev.get("macro_f1"),
            "intent_weighted_f1": ev.get("weighted_f1"),
            "n_samples":          ev.get("n_samples", ev.get("n_samples")),
            "note":               "from results/eval_report.json (Phase 2 eval)",
        })
    # Judge scores
    if os.path.exists(JUDGE_JSON):
        with open(JUDGE_JSON, encoding="utf-8") as f:
            jd = json.load(f)
        scores = jd.get("judge_scores", [])
        overalls = [s.get("overall") for s in scores
                    if s.get("overall") is not None and not s.get("judge_error")]
        if overalls:
            import numpy as np
            result["judge_overall_mean"] = round(float(np.mean(overalls)), 2)
    return result


# ── Formatting ────────────────────────────────────────────────────────────

def format_comparison(b1: dict, b2: Optional[dict], full: dict, source: str) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("BASELINE COMPARISON REPORT")
    lines.append("=" * 72)
    lines.append(f"Eval set  : {source}")
    lines.append("")

    def fmt(v, pct=False) -> str:
        if v is None:
            return "   N/A"
        if pct:
            return f"{v*100:5.1f}%"
        return f"{v:.3f}"

    lines.append(f"{'Metric':<30s}  {'Baseline 1':>12}  {'Baseline 2':>12}  {'Full System':>12}")
    lines.append(f"{'─'*30}  {'─'*12}  {'─'*12}  {'─'*12}")
    lines.append(f"{'Method':<30s}  {'KW+Canned':>12}  {'ZeroShot':>12}  {'Few-shot+RAG':>12}")
    lines.append(f"{'─'*30}  {'─'*12}  {'─'*12}  {'─'*12}")

    b2_acc  = b2.get("intent_accuracy")  if b2 else None
    b2_mf1  = b2.get("intent_macro_f1") if b2 else None

    rows = [
        ("Intent accuracy",        b1["intent_accuracy"],  b2_acc,  full.get("intent_accuracy")),
        ("Intent macro F1",        b1["intent_macro_f1"],  b2_mf1,  full.get("intent_macro_f1")),
        ("Intent weighted F1",     b1["intent_weighted_f1"], b2.get("intent_weighted_f1") if b2 else None,
                                   full.get("intent_weighted_f1")),
        ("Retrieval hit@3",        "N/A",  "N/A",  b1.get("retrieval_hit3", "see metrics")),
        ("Escalation F1",          "N/A",  full.get("escalation_f1","—"),   full.get("escalation_f1","—")),
        ("Reply: mean chars",      b1.get("reply_mean_chars"),
                                   b2.get("reply_mean_chars") if b2 else None,  None),
        ("Reply: judge overall/5", b1.get("judge_overall"),
                                   b2.get("judge_overall") if b2 else None,
                                   full.get("judge_overall_mean")),
    ]

    for label, v1, v2, vf in rows:
        def cell(v):
            if v is None:
                return "   N/A"
            if isinstance(v, str):
                return f"{v:>12}"
            return f"{v:>12.3f}" if isinstance(v, float) else f"{v!s:>12}"
        lines.append(f"  {label:<28s}  {cell(v1)}  {cell(v2)}  {cell(vf)}")

    lines.append("")
    lines.append("Notes:")
    lines.append("  Baseline 1 uses keyword heuristics — no API, deterministic.")
    lines.append("  Baseline 2 uses zero-shot NIM (same model, no few-shot examples, no retrieval).")
    lines.append("  Full system uses few-shot NIM + FAISS retrieval + 8-rule escalation.")
    lines.append(f"  Full system numbers: {full.get('note','')}")
    lines.append("")
    lines.append("Key finding: few-shot + retrieval gap over zero-shot demonstrates")
    lines.append("  the contribution of each component independently.")
    lines.append("=" * 72)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=MAX_EVAL)
    parser.add_argument("--b1-only", action="store_true",
                        help="Run only Baseline 1 (no API calls)")
    args = parser.parse_args()

    n = min(args.n, MAX_EVAL)

    tweets, labels, source = load_eval_data(n)
    print(f"Eval set: {source}  ({len(tweets)} examples)")

    # ── INTENTS ───────────────────────────────────────────────────────────
    label_set = sorted(set(labels))

    # ── BASELINE 1: keyword + canned ─────────────────────────────────────
    print("\nBaseline 1: keyword-match + canned reply…")
    b1_preds   = [keyword_predict(t) for t in tweets]
    b1_replies = [CANNED_REPLIES.get(p, CANNED_REPLIES["general_complaint_or_feedback"])
                  for p in b1_preds]
    acc1, mf1, wf1, rep1 = classification_metrics(labels, b1_preds, label_set)
    b1 = {
        "intent_accuracy":    round(acc1, 4),
        "intent_macro_f1":    round(mf1, 4),
        "intent_weighted_f1": round(wf1, 4),
        "reply_mean_chars":   round(reply_length_stats(b1_replies)["mean"], 1),
        "judge_overall":      None,  # filled after judge runs (optional)
        "per_class":          {
            intent: {
                "precision": round(rep1.get(intent, {}).get("precision", 0), 4),
                "recall":    round(rep1.get(intent, {}).get("recall",    0), 4),
                "f1":        round(rep1.get(intent, {}).get("f1-score",  0), 4),
            } for intent in label_set
        },
    }
    print(f"  Accuracy: {acc1:.3f}   Macro F1: {mf1:.3f}")

    # ── BASELINE 2: zero-shot NIM, no retrieval ───────────────────────────
    b2 = None
    if not args.b1_only:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        key = os.environ.get("NVIDIA_API_KEY", "")
        if not key:
            print("\nBaseline 2: NVIDIA_API_KEY not set — skipping")
        else:
            from openai import OpenAI
            client = OpenAI(base_url=NIM_BASE_URL, api_key=key)

            print(f"\nBaseline 2: zero-shot NIM for {len(tweets)} tweets…")
            b2_preds   = []
            b2_confs   = []
            b2_replies = []
            for i, (tweet, label) in enumerate(zip(tweets, labels)):
                print(f"  [{i+1}/{len(tweets)}] classifying…", end="", flush=True)
                clf_result = zero_shot_classify(tweet, client)
                pred = clf_result["intent"]
                b2_preds.append(pred)
                b2_confs.append(clf_result.get("confidence", 0.5))
                print(f" → {pred} ({clf_result.get('confidence', 0.5):.0%})", end="")

                reply = zero_shot_draft(tweet, pred, client)
                b2_replies.append(reply)
                print(f"  reply={len(reply)}ch")

            acc2, mf2, wf2, rep2 = classification_metrics(labels, b2_preds, label_set)
            import numpy as np
            b2 = {
                "intent_accuracy":    round(acc2, 4),
                "intent_macro_f1":    round(mf2, 4),
                "intent_weighted_f1": round(wf2, 4),
                "avg_confidence":     round(float(np.mean(b2_confs)), 3),
                "reply_mean_chars":   round(reply_length_stats(b2_replies)["mean"], 1),
                "judge_overall":      None,
                "per_class":          {
                    intent: {
                        "precision": round(rep2.get(intent, {}).get("precision", 0), 4),
                        "recall":    round(rep2.get(intent, {}).get("recall",    0), 4),
                        "f1":        round(rep2.get(intent, {}).get("f1-score",  0), 4),
                    } for intent in label_set
                },
            }
            print(f"  Accuracy: {acc2:.3f}   Macro F1: {mf2:.3f}")

    # ── Full system numbers (from existing files) ─────────────────────────
    full = load_full_system_numbers()

    # ── Save + print ──────────────────────────────────────────────────────
    out = {
        "source":      source,
        "n_samples":   len(tweets),
        "baseline_1":  b1,
        "baseline_2":  b2,
        "full_system": full,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    report_txt = format_comparison(b1, b2, full, source)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print()
    print(report_txt)
    print(f"Saved → {OUT_JSON}")
    print(f"Saved → {OUT_TXT}")


if __name__ == "__main__":
    main()
