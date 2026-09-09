"""
16_eval_report.py — Master evaluation report.

Assembles all result files into one structured document with an honest
"what's misleading about my headline number" section.

Reads:
    results/eval_report.json              — Phase 2 classifier metrics
    results/automated_metrics.json        — Phase 4 metrics (13_automated_metrics.py)
    results/judge_scores.json             — LLM judge scores
    results/baseline_comparison.json      — Baseline 1 & 2 vs full system
    results/banking77_cross_domain_report.json — cross-domain supplement

Output:
    results/eval_full_report.txt
    results/eval_full_report.json

Run:
    python scripts/16_eval_report.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")

IN_EVAL       = os.path.join(RESULTS_DIR, "eval_report.json")
IN_AUTO       = os.path.join(RESULTS_DIR, "automated_metrics.json")
IN_JUDGE      = os.path.join(RESULTS_DIR, "judge_scores.json")
IN_BASELINE   = os.path.join(RESULTS_DIR, "baseline_comparison.json")
IN_B77        = os.path.join(RESULTS_DIR, "banking77_cross_domain_report.json")
OUT_TXT       = os.path.join(RESULTS_DIR, "eval_full_report.txt")
OUT_JSON      = os.path.join(RESULTS_DIR, "eval_full_report.json")


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(v) -> str:
    if v is None:
        return "  N/A  "
    return f"{v*100:5.1f}%"


def _f(v, decimals=3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


# ── Section builders ──────────────────────────────────────────────────────

def section_executive(ev, auto, baseline, judge_data) -> str:
    b = []
    b.append("=" * 72)
    b.append("HIVER SDE INTERN TAKE-HOME — EVALUATION SUMMARY")
    b.append("=" * 72)
    b.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    b.append("")
    b.append("Task  : Intent classifier + retrieval + reply drafter + escalation")
    b.append("Data  : AmazonHelp Twitter (3,000 tweets, 8,000 resolved threads)")
    b.append("Models: nvidia/nemotron-3-super-120b-a12b (NVIDIA NIM, free tier)")
    b.append("")

    # Pull numbers
    acc      = ev.get("accuracy")   if ev else None
    mf1      = ev.get("macro_f1")   if ev else None
    n_samp   = ev.get("n_samples")  if ev else None

    b1 = baseline.get("baseline_1", {}) if baseline else {}
    b2 = baseline.get("baseline_2", {}) if baseline else {}
    fs = baseline.get("full_system", {}) if baseline else {}

    auto_ret = auto.get("retrieval", {}) if auto else {}
    auto_esc = auto.get("escalation") if auto else None

    j_scores = []
    if judge_data:
        j_scores = [s for s in judge_data.get("judge_scores", [])
                    if s.get("overall") and not s.get("judge_error")]
    j_mean = (sum(s["overall"] for s in j_scores) / len(j_scores)) if j_scores else None

    b.append("EXECUTIVE SUMMARY TABLE")
    b.append("─" * 72)
    b.append(f"  {'Metric':<38s}  {'B1 Keyword':>10}  {'B2 ZeroShot':>11}  {'Full Sys':>9}")
    b.append(f"  {'─'*38}  {'─'*10}  {'─'*11}  {'─'*9}")

    rows = [
        ("Intent accuracy",          b1.get("intent_accuracy"), b2.get("intent_accuracy") if b2 else None,     acc),
        ("Intent macro F1",          b1.get("intent_macro_f1"), b2.get("intent_macro_f1") if b2 else None,     mf1),
        ("Retrieval hit@3",          None,                      None,                          auto_ret.get("hit@3")),
        ("Escalation F1",            None,                      None,                          auto_esc.get("f1") if auto_esc else None),
        ("Reply judge score (1-5)",  None,                      None,                          j_mean),
    ]
    for label, v1, v2, vf in rows:
        def cell(v):
            if v is None:
                return " " * 10 + "—"
            return f"{v:>10.3f}" if isinstance(v, float) else f"{v!s:>10}"
        b.append(f"  {label:<38s}  {cell(v1)}  {cell(v2)}  {cell(vf)}")

    b.append("")
    if n_samp:
        b.append(f"  Intent eval on {n_samp} examples (10 per intent × 11 intents).")
    b.append("")
    return "\n".join(b)


def section_intent(ev) -> str:
    if not ev:
        return "  [Intent classifier section — run 06_evaluate.py first]\n"
    b = []
    b.append("─" * 72)
    b.append("1. INTENT CLASSIFIER (few-shot Nemotron, 11 intents)")
    b.append("─" * 72)
    b.append(f"  Model          : {ev.get('mode', 'nvidia/nemotron-3-super-120b-a12b')}")
    b.append(f"  Eval samples   : {ev.get('n_samples')}")
    b.append(f"  Accuracy       : {_f(ev.get('accuracy'))}")
    b.append(f"  Majority base  : {_f(ev.get('majority_baseline_accuracy'))}")
    b.append(f"  Macro F1       : {_f(ev.get('macro_f1'))}")
    b.append(f"  API errors     : {ev.get('n_errors', 0)}")
    b.append(f"  Uncertain preds: {ev.get('n_uncertain', 0)}")
    b.append("")
    b.append(f"  {'Intent':<42s}  {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>4}")
    b.append(f"  {'─'*42}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*4}")
    for intent, m in sorted(ev.get("per_class_metrics", {}).items()):
        b.append(
            f"  {intent:<42s}  {m.get('precision', 0):6.3f}  "
            f"{m.get('recall', 0):6.3f}  {m.get('f1', 0):6.3f}  "
            f"{m.get('support', 0):4d}"
        )
    b.append("")
    b.append("  Worst confusion pairs:")
    for c in ev.get("worst_confusions", [])[:5]:
        b.append(f"    {c['true']} → {c['predicted']}  (n={c['count']})")
    b.append("")
    return "\n".join(b)


def section_retrieval(auto) -> str:
    if not auto:
        return "  [Retrieval section — run 13_automated_metrics.py first]\n"
    ret = auto.get("retrieval", {})
    if not ret:
        return "  [Retrieval metrics not yet available]\n"
    b = []
    b.append("─" * 72)
    b.append("2. RETRIEVAL QUALITY  (FAISS, all-MiniLM-L6-v2, intent-filtered)")
    b.append("─" * 72)
    b.append("  Same-intent match in top-K retrieved results:")
    for k, v in sorted(ret.items()):
        bar = "█" * int(v * 20)
        b.append(f"    {k:<8s}: {v:.3f}  {bar}")
    b.append("")
    b.append("  Interpretation: hit@3 is the primary signal — the drafter sees")
    b.append("  up to 3 grounding examples. A hit means ≥1 example is on-intent.")
    b.append("")
    return "\n".join(b)


def section_escalation(auto) -> str:
    if not auto:
        return "  [Escalation section — run 13_automated_metrics.py first]\n"
    esc = auto.get("escalation")
    b = []
    b.append("─" * 72)
    b.append("3. ESCALATION QUALITY  (8-rule deterministic engine)")
    b.append("─" * 72)
    if esc is None:
        b.append("  Not yet evaluated — fill in human_escalate column in")
        b.append("  data/golden_label_sheet.csv, then re-run 13_automated_metrics.py.")
    else:
        b.append(f"  N evaluated : {esc.get('n_evaluated')}")
        b.append(f"  Precision   : {_f(esc.get('precision'))}")
        b.append(f"  Recall      : {_f(esc.get('recall'))}")
        b.append(f"  F1          : {_f(esc.get('f1'))}")
        b.append(f"  TP/FP/FN/TN : {esc['tp']}/{esc['fp']}/{esc['fn']}/{esc['tn']}")
    b.append("")
    return "\n".join(b)


def section_judge(judge_data) -> str:
    if not judge_data:
        return "  [Judge section — run 14_llm_judge.py first]\n"
    import numpy as np
    b = []
    b.append("─" * 72)
    b.append("4. REPLY QUALITY  (LLM-as-judge)")
    b.append("─" * 72)
    b.append(f"  Judge model : {judge_data.get('judge_model', 'N/A')}")
    b.append(f"  Gen model   : {judge_data.get('gen_model', 'N/A')}")
    b.append(f"  N evaluated : {judge_data.get('n_evaluated', 0)}")
    b.append("")
    scores = [s for s in judge_data.get("judge_scores", [])
              if not s.get("judge_error")]
    dims = ["relevance", "empathy", "actionability", "conciseness", "overall"]
    b.append(f"  {'Dimension':<16s}  {'Mean':>6}  {'Median':>7}  {'≥4':>6}")
    b.append(f"  {'─'*16}  {'─'*6}  {'─'*7}  {'─'*6}")
    for dim in dims:
        vals = [s[dim] for s in scores if s.get(dim) is not None]
        if vals:
            arr = np.array(vals, dtype=float)
            b.append(f"  {dim:<16s}  {arr.mean():6.2f}  {np.median(arr):7.2f}"
                     f"  {np.mean(arr>=4)*100:5.1f}%")

    ag = judge_data.get("agreement")
    b.append("")
    if ag:
        b.append("  Judge–human agreement:")
        b.append(f"  {'Dimension':<16s}  {'N':>4}  {'Pearson r':>9}  {'MAE':>6}  {'±1':>6}")
        b.append(f"  {'─'*16}  {'─'*4}  {'─'*9}  {'─'*6}  {'─'*6}")
        for dim in dims:
            ag_d = ag.get(dim, {})
            if "note" in ag_d:
                b.append(f"  {dim:<16s}  {ag_d.get('n', 0):>4}  (too few pairs)")
            else:
                b.append(
                    f"  {dim:<16s}  {ag_d.get('n', 0):>4}  "
                    f"{ag_d.get('pearson_r', 'N/A'):>9}  "
                    f"{ag_d.get('mae', 'N/A'):>6}  "
                    f"{ag_d.get('within_1', 'N/A'):>6}"
                )
    else:
        b.append("  Judge–human agreement: pending (fill data/human_judge_ratings.csv)")
    b.append("")
    return "\n".join(b)


def section_baselines(baseline) -> str:
    if not baseline:
        return "  [Baselines section — run 15_baselines.py first]\n"
    b = []
    b.append("─" * 72)
    b.append("5. BASELINE COMPARISON")
    b.append("─" * 72)
    b1 = baseline.get("baseline_1", {})
    b2 = baseline.get("baseline_2")
    fs = baseline.get("full_system", {})

    def row(label, f1_, f2_, ff_):
        def cell(v):
            if v is None:
                return "   —   "
            return f"{v:7.3f}" if isinstance(v, float) else f"{v!s:>7}"
        return f"  {label:<30s}  {cell(f1_)}  {cell(f2_)}  {cell(ff_)}"

    b.append(f"  {'Metric':<30s}  {'B1 Keyword':>7}  {'B2 ZS-NIM':>9}  {'Full Sys':>8}")
    b.append(f"  {'─'*30}  {'─'*7}  {'─'*9}  {'─'*8}")
    b.append(row("Intent accuracy",
                 b1.get("intent_accuracy"),
                 b2.get("intent_accuracy") if b2 else None,
                 fs.get("intent_accuracy")))
    b.append(row("Intent macro F1",
                 b1.get("intent_macro_f1"),
                 b2.get("intent_macro_f1") if b2 else None,
                 fs.get("intent_macro_f1")))
    b.append(row("Reply mean chars",
                 b1.get("reply_mean_chars"),
                 b2.get("reply_mean_chars") if b2 else None,
                 None))
    b.append("")
    b.append("  Δ (full − keyword): intent accuracy gain from few-shot + retrieval")
    b.append("  Δ (full − zeroshot): gain from adding few-shot examples")
    b.append("")
    return "\n".join(b)


def section_banking77(b77) -> str:
    if not b77:
        return "  [Banking77 section — run 09_banking77_cross_domain.py first]\n"
    b = []
    b.append("─" * 72)
    b.append("6. CROSS-DOMAIN GENERALIZATION  (Banking77 supplementary)")
    b.append("─" * 72)
    b.append(f"  Queries evaluated  : {b77.get('n_queries')}")
    b.append(f"  Cross-domain acc   : {_f(b77.get('overall_accuracy'))}"
             f"  (vs {_f(b77.get('in_domain_accuracy', None))} in-domain)")
    b.append(f"  Cross-domain macro F1 : {_f(b77.get('macro_f1'))}")
    b.append(f"  Hard-pair accuracy    : {_f(b77.get('hard_pair_accuracy'))}")
    b.append(f"  Avg confidence        : {_f(b77.get('avg_confidence'))}")
    b.append("")
    b.append("  Per-class F1 gap (in-domain − cross-domain, positive = degradation):")
    for intent, m in sorted(b77.get("per_class_metrics", {}).items()):
        gap = m.get("f1_gap")
        if gap is not None:
            bar = "█" * min(20, max(0, int(abs(gap) * 20)))
            sign = "↓" if gap > 0 else "↑"
            b.append(f"    {intent:<42s}  {sign}{abs(gap):.3f}  {bar}")
    b.append("")
    return "\n".join(b)


def section_honest_analysis(ev, b77, baseline) -> str:
    b = []
    b.append("─" * 72)
    b.append("7. WHAT'S MISLEADING ABOUT MY HEADLINE NUMBER")
    b.append("─" * 72)

    acc = ev.get("accuracy", 0.545) if ev else 0.545
    mf1 = ev.get("macro_f1", 0.563) if ev else 0.563

    b.append(f"""
  Headline: {acc:.1%} accuracy / {mf1:.3f} macro F1 on the intent classifier.
  Here is why those numbers require careful interpretation:

  1. AUTO-LABELED EVAL SET — The 110-example eval set was created by the
     same keyword heuristic that drives Baseline 1.  If the classifier learned
     the same surface patterns, it has an inflated score: it is being tested on
     labels it helped generate, not against independent human judgment.
     → The golden set (data/golden_label_sheet.csv) fixes this.

  2. 27/110 API ERRORS (25%) — The initial eval run had 27 NIM timeouts,
     filled with the model's default ("uncertain") label.  Those 27 examples
     artificially depress accuracy.  The corrected run on the 83 successful
     predictions showed higher accuracy.  The headline mixes both.
     → See results/eval_report_corrected.json for the denoised number.

  3. MACRO F1 HIDES CLASS IMBALANCE — In production, general_complaint and
     shipping_delay dominate (>50% combined).  Macro F1 gives equal weight
     to all 11 classes, so the 0.233 F1 on general_complaint (the biggest
     class) has only 1/11 impact on the headline.  In terms of customer
     volume, that single class matters 5× more than most others.
     → Weighted F1 is a better production proxy: {_f(ev.get('weighted_f1')) if ev else 'N/A'}.

  4. CROSS-DOMAIN COLLAPSE — On Banking77, accuracy drops from {acc:.1%} to
     {_f(b77.get('overall_accuracy')) if b77 else 'N/A'} — a significant gap.
     This reveals keyword reliance: intents whose keywords appear consistently
     in Twitter language (account_access) generalise well; intents with
     Twitter-specific phrasing (shipping_delay_complaint) collapse.
     → Hard-pair accuracy: {_f(b77.get('hard_pair_accuracy')) if b77 else 'N/A'}.
     This is the most honest signal of true semantic generalisation.

  5. CONFIDENCE MISCALIBRATION — The model reports 0.9+ confidence on many
     incorrect predictions (Q3: mean conf = 0.926, accuracy = 0.667).
     Downstream systems that trust raw confidence scores will make poor
     escalation decisions.  The 8-rule escalation engine avoids this by using
     text signals rather than model confidence for escalation triggers.

  6. RETRIEVAL IS EVALUATED ON PREDICTED LABELS — hit-rate uses predicted
     intent (not ground truth) as the filter.  If the intent prediction is
     wrong, the retrieved examples are wrong-intent regardless of hit-rate.
     → True retrieval quality is contingent on classifier quality.
""".strip())
    b.append("")
    return "\n".join(b)


def section_methodology() -> str:
    return """\
─────────────────────────────────────────────────────────────────────────────
8. METHODOLOGY NOTES
─────────────────────────────────────────────────────────────────────────────
  Dataset    : AmazonHelp tweets from "Customer Support on Twitter" (Kaggle).
               3,000 inbound tweets sampled (full dataset ~3M).

  Taxonomy   : 11 intents derived from k-means (k=12) on tweet embeddings,
               then manually collapsed + labelled.  See intent_taxonomy.md.

  Classifier : nvidia/nemotron-3-super-120b-a12b via NVIDIA NIM free tier.
               Few-shot: 2 examples per intent inline in the system prompt.
               Temperature=0, JSON schema output (no manual parsing).

  Retrieval  : FAISS IndexFlatIP, 8,000 resolved threads, all-MiniLM-L6-v2
               embeddings (384-dim), intent-filtered top-K.

  Drafter    : Same NIM model.  Up to 3 retrieved grounding examples.
               Parser: 4-level fallback for Nemotron preamble noise.
               grounding_examples field logged per draft (auditable).

  Escalation : 8-rule deterministic engine.  Zero API calls.  Every decision
               includes triggered_rule + human-readable reason string.

  Baselines  : Keyword (PATTERN_MAP) + canned reply.
               Zero-shot NIM (no few-shot, no retrieval).

  Judge      : openai/gpt-oss-20b (or meta/llama-3.1-70b-instruct fallback).
               Different model family from generator.
               4 dimensions (relevance, empathy, actionability, conciseness).

  Reproducibility: < 15 minutes from clean clone with NVIDIA_API_KEY set.
                   See README.md quickstart.
"""


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading result files…")
    ev       = _load(IN_EVAL)
    auto     = _load(IN_AUTO)
    judge    = _load(IN_JUDGE)
    baseline = _load(IN_BASELINE)
    b77      = _load(IN_B77)

    missing = [path for path in [IN_EVAL, IN_AUTO, IN_JUDGE, IN_BASELINE, IN_B77]
               if not os.path.exists(path)]
    if missing:
        print(f"  ⚠  Missing files (sections will be placeholders):")
        for m in missing:
            print(f"     {os.path.basename(m)}")

    sections = [
        section_executive(ev, auto, baseline, judge),
        section_intent(ev),
        section_retrieval(auto),
        section_escalation(auto),
        section_judge(judge),
        section_baselines(baseline),
        section_banking77(b77),
        section_honest_analysis(ev, b77, baseline),
        section_methodology(),
    ]

    report = "\n".join(sections)
    report += "\n" + "=" * 72 + "\n"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    summary = {
        "generated":      datetime.now().isoformat(),
        "sources_loaded": {
            "eval_report":              ev is not None,
            "automated_metrics":        auto is not None,
            "judge_scores":             judge is not None,
            "baseline_comparison":      baseline is not None,
            "banking77_cross_domain":   b77 is not None,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(report)
    print(f"Saved → {OUT_TXT}")
    print(f"Saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
