"""
11_pipeline_demo.py
===================
End-to-end pipeline demo:
  tweet → classify → retrieve → draft_reply → escalate

Runs 8 handpicked tweets that collectively exercise:
  • 5 different intents
  • All 8 escalation rules (legal threat, security, repeated contact,
    low confidence, high negativity, intent-specific, default auto-handle)
  • Both auto_handle and escalate paths

Usage
-----
    $env:NVIDIA_API_KEY = 'nvapi-...'
    python scripts/11_pipeline_demo.py

Output
------
    results/pipeline_demo.json   — full structured output (all fields)
    results/pipeline_demo.txt    — human-readable report
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH  = os.path.join(DATA_DIR, "faiss_meta.jsonl")

# ── Check index exists ─────────────────────────────────────────────────────
if not os.path.exists(INDEX_PATH):
    sys.exit(
        "ERROR: FAISS index not found. Run scripts/04_build_index.py first.\n"
        f"Expected: {INDEX_PATH}"
    )

# ── Imports ────────────────────────────────────────────────────────────────
print("Loading modules…")
from classifier import classify, MODEL_ID, MIN_INTERVAL
from retrieval  import RetrievalIndex
from drafter    import draft
from escalation import decide

# ── Load retrieval index ───────────────────────────────────────────────────
print("Loading retrieval index…")
idx = RetrievalIndex.load(INDEX_PATH, META_PATH)
print(f"  Index: {idx.size} threads\n")

# ── Demo tweets ────────────────────────────────────────────────────────────
# Chosen to exercise every escalation rule path and multiple intents.
DEMO_TWEETS = [
    # 1. Clean status inquiry → auto_handle
    "@AmazonHelp Can you check the status of my order? Tracking says "
    "'out for delivery' but it's been that way for 3 days.",

    # 2. Shipping delay frustration → auto_handle (medium negativity, not enough to escalate)
    "@AmazonHelp I paid for Prime 2-day shipping and my package arrived 6 days "
    "late. This is the second time this month.",

    # 3. Security breach + account_access → escalate (Rule 4)
    "@AmazonHelp someone hacked my account and made purchases I didn't authorise. "
    "Please help immediately, this is urgent.",

    # 4. Legal threat → escalate (Rule 3)
    "@AmazonHelp I've been waiting 3 weeks for my refund. If I don't hear back "
    "today I'm contacting my attorney and filing a chargeback.",

    # 5. Simple refund → auto_handle
    "@AmazonHelp I returned my item 10 days ago and the refund hasn't appeared "
    "yet. Order #302-9384-1234. Can you check?",

    # 6. Repeated contact frustration → escalate (Rule 5)
    "@AmazonHelp I've called customer service 4 times and still no resolution. "
    "Still waiting for someone to actually fix my account issue.",

    # 7. High negativity density → escalate (Rule 6)
    "@AmazonHelp This is absolutely disgusting. Worst customer service ever. "
    "Incompetent, pathetic, appalling service. I am furious and done with Amazon.",

    # 8. Device issue → auto_handle (clear, low urgency)
    "@AmazonHelp my Kindle Paperwhite won't turn on after the latest software "
    "update. Already tried the restart button for 40 seconds.",
]

# ── Run pipeline ───────────────────────────────────────────────────────────
print(f"Running pipeline for {len(DEMO_TWEETS)} tweets…")
print(f"Model: {MODEL_ID}")
print(f"Rate:  {MIN_INTERVAL}s between API calls\n")

pipeline_results = []
t_pipeline_start = time.perf_counter()

for i, tweet in enumerate(DEMO_TWEETS, 1):
    print(f"[{i}/{len(DEMO_TWEETS)}] Classifying…", end=" ", flush=True)
    t0 = time.perf_counter()

    # Step 1: Classify
    clf = classify(tweet)
    print(f"→ {clf.intent} ({clf.confidence:.0%})", end="  ", flush=True)

    # Step 2: Retrieve (use classified intent as filter)
    intent_for_retrieval = clf.intent if clf.intent != "uncertain" else None
    retrieved = idx.query(tweet, intent=intent_for_retrieval, top_k=3)
    print(f"Retrieved {len(retrieved)} examples", end="  ", flush=True)

    # Step 3: Draft reply (only if classifier didn't error)
    print("Drafting…", end=" ", flush=True)
    drafted = draft(tweet, clf, retrieved)
    print(f"✓", end="  ", flush=True)

    # Step 4: Escalation decision (synchronous, no API call)
    esc = decide(tweet, clf)
    elapsed = time.perf_counter() - t0
    print(f"→ {esc.decision.upper()}  ({elapsed:.1f}s)")

    pipeline_results.append({
        "index":     i,
        "tweet":     tweet,
        "classification": {
            "intent":     clf.intent,
            "confidence": clf.confidence,
            "reasoning":  clf.reasoning,
            "alternatives": [
                {"intent": a.intent, "confidence": a.confidence}
                for a in (clf.alternatives or [])
            ],
            "error": clf.error,
        },
        "retrieval": {
            "n_results": len(retrieved),
            "examples": [
                {
                    "thread_id":     r.thread_id,
                    "customer_tweet": r.customer_tweet[:200],
                    "brand_reply":    r.brand_reply[:200],
                    "intent":         r.intent,
                    "similarity":     round(r.similarity, 3),
                }
                for r in retrieved
            ],
        },
        "draft": {
            "reply":        drafted.reply,
            "reply_length": drafted.reply_length,
            "tone":         drafted.tone,
            "action_taken": drafted.action_taken,
            "n_grounding":  drafted.n_grounding_examples,
            "error":        drafted.error,
        },
        "escalation": {
            "decision":        esc.decision,
            "reason":          esc.reason,
            "triggered_rule":  esc.triggered_rule,
            "urgency_level":   esc.urgency_level,
            "urgency_signals": esc.urgency_signals,
            "confidence":      esc.confidence,
        },
    })

total_elapsed = time.perf_counter() - t_pipeline_start
print(f"\nPipeline complete: {len(DEMO_TWEETS)} tweets in {total_elapsed:.1f}s\n")

# ── Save JSON ──────────────────────────────────────────────────────────────
json_path = os.path.join(RESULTS_DIR, "pipeline_demo.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"model": MODEL_ID, "n_tweets": len(DEMO_TWEETS),
               "results": pipeline_results}, f, indent=2, ensure_ascii=False)

# ── Human-readable report ──────────────────────────────────────────────────
W = 78
DIVIDER  = "═" * W
DIVIDER2 = "─" * W

lines = [
    DIVIDER,
    "PIPELINE DEMO — CLASSIFY → RETRIEVE → DRAFT → ESCALATE",
    DIVIDER,
    f"Model : {MODEL_ID}",
    f"Tweets: {len(DEMO_TWEETS)}",
    f"Index : {idx.size} resolved threads",
    "",
]

INTENT_EMOJI = {
    "order_status_inquiry":     "📦",
    "shipping_delay_complaint": "🚚",
    "refund_request":           "💰",
    "account_access":           "🔑",
    "delivery_issue":           "📬",
    "return_or_exchange":       "🔄",
    "product_issue":            "⚠️",
    "prime_membership":         "⭐",
    "order_cancellation":       "❌",
    "device_and_digital_support":"💻",
    "general_complaint_or_feedback":"💬",
    "uncertain":                "❓",
}

DECISION_EMOJI = {"auto_handle": "✅ AUTO-HANDLE", "escalate": "🚨 ESCALATE"}

for r in pipeline_results:
    intent   = r["classification"]["intent"]
    conf     = r["classification"]["confidence"]
    decision = r["escalation"]["decision"]
    emoji_i  = INTENT_EMOJI.get(intent, "❓")
    emoji_d  = DECISION_EMOJI.get(decision, decision)

    lines += [
        DIVIDER2,
        f"Tweet #{r['index']}  │  {emoji_i} {intent}  ({conf:.0%})  │  {emoji_d}",
        DIVIDER2,
        "",
        "📝 ORIGINAL TWEET",
        f'   "{r["tweet"]}"',
        "",
    ]

    # Grounding examples
    examples = r["retrieval"]["examples"]
    if examples:
        lines.append(f"🔍 RETRIEVED GROUNDING ({r['retrieval']['n_results']} results, top {min(3, len(examples))} shown)")
        for j, ex in enumerate(examples[:3], 1):
            lines.append(
                f"   [{j}] sim={ex['similarity']:.2f}  intent={ex['intent']}"
            )
            lines.append(f"       Customer : {ex['customer_tweet'][:90]}")
            lines.append(f"       Resolution: {ex['brand_reply'][:90]}")
        lines.append("")

    # Classifier reasoning
    lines.append("🧠 CLASSIFIER REASONING")
    lines.append(f"   {r['classification']['reasoning']}")
    alts = r["classification"]["alternatives"]
    if alts:
        alt_str = " | ".join(f"{a['intent']} ({a['confidence']:.0%})" for a in alts)
        lines.append(f"   Alternatives: {alt_str}")
    lines.append("")

    # Drafted reply
    d = r["draft"]
    err_note = f"  ⚠️  DRAFTER ERROR: {d['error']}" if d["error"] else ""
    lines.append(f"✉️  DRAFTED REPLY  (tone={d['tone']}, {d['reply_length']} chars, "
                 f"action={d['action_taken']}){err_note}")
    lines.append(f'   "{d["reply"]}"')
    lines.append("")

    # Escalation
    esc = r["escalation"]
    lines.append(f"⚖️  ESCALATION: {emoji_d}")
    lines.append(f"   Rule      : {esc['triggered_rule']}")
    lines.append(f"   Urgency   : {esc['urgency_level']}")
    if esc["urgency_signals"]:
        for sig in esc["urgency_signals"]:
            lines.append(f"   Signal    : {sig}")
    lines.append(f"   Reason    : {esc['reason']}")
    lines.append("")

lines += [DIVIDER, "END OF DEMO", DIVIDER]
report = "\n".join(lines)
print(report)

txt_path = os.path.join(RESULTS_DIR, "pipeline_demo.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nSaved → {json_path}")
print(f"Saved → {txt_path}")
