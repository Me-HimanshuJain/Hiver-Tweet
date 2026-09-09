"""
escalation.py — Module 4: Escalation decision engine.

Combines three independent signals into a binary auto_handle / escalate
decision.  Every decision emits a human-readable reason string so the
routing is fully auditable.

Signals (applied in priority order)
------------------------------------
1. Classifier availability  — error / uncertain intent → escalate
2. Confidence gate          — below threshold → escalate
3. Legal / media / threat   — high-urgency keywords → escalate
4. Security breach          — "hacked / stolen / fraud" + account_access → escalate
5. Repeated contacts        — "X times", "multiple times", "still" → escalate
6. Sentiment overload       — high negative keyword density → escalate
7. Intent-specific rules    — e.g. refund_request + "never received" → escalate
8. Default                  — auto-handle

Public API
----------
    decide(tweet, classification)  -> EscalationDecision
    decide_batch(items)            -> list[EscalationDecision]
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# ── Thresholds ────────────────────────────────────────────────────────────
CONFIDENCE_ESCALATE_THRESHOLD = 0.55   # below → escalate

# ── Keyword lists ─────────────────────────────────────────────────────────
_LEGAL_PATTERNS = re.compile(
    r"\b(su(?:e|ing|ed)|lawsuit|legal\s+action|attorney|lawyer|court|police|"
    r"report\s+(?:you|amazon|this)|bbc|media|news|journalist|press|twitter\s+war|"
    r"charge(?:back|d\s+back)|dispute(?:d)?|ombudsman|trading\s+standards)\b",
    re.IGNORECASE,
)

_SECURITY_PATTERNS = re.compile(
    r"\b(hack(?:ed|er|ing)?|sto(?:len|le)|comprom(?:ised|ise)|fraud(?:ulent)?|"
    r"unauthori[sz]ed|identity\s+theft|phish(?:ing|ed)?|breach)\b",
    re.IGNORECASE,
)

_REPETITION_PATTERNS = re.compile(
    r"\b((?:called|contacted|messaged|tried|emailed|chatted)\s+(?:\w+\s+)?"
    r"(?:\d+|\bseveral\b|\bmultiple\b|\bmany\b|\brepeatedly\b)\s+times|"
    r"still\s+(?:no\s+(?:reply|response|update|resolution|help)|waiting|haven['\u2019]t)|"
    r"(?:\d+)\s+days?\s+(?:and\s+)?(?:still|no|nothing|without))\b",
    re.IGNORECASE,
)

# High-negativity words — escalate when ≥ NEGATIVITY_THRESHOLD found
_NEGATIVE_WORDS = re.compile(
    r"\b(terrible|horrible|disgusting|outrageous|unacceptable|appalling|"
    r"furious|livid|fuming|enraged|hate|despise|pathetic|incompetent|useless|"
    r"worst\s+ever|never\s+(?:again|use|buy|shop)|ruined|disgrace|shameful|"
    r"cancelled?\s+my\s+(?:account|prime|subscription)|done\s+with\s+amazon)\b",
    re.IGNORECASE,
)
NEGATIVITY_THRESHOLD = 2    # ≥ 2 high-negativity signals → escalate

# Intent-specific escalation patterns
_INTENT_ESCALATE: dict[str, re.Pattern] = {
    "account_access": re.compile(
        r"\b(hack|stolen|fraud|unauthorized|compromised|breach|someone\s+(?:else|got))\b",
        re.IGNORECASE,
    ),
    "refund_request": re.compile(
        r"\b(never\s+received|double\s+charg|charged\s+twice|still\s+waiting\s+for\s+(?:my\s+)?refund)\b",
        re.IGNORECASE,
    ),
    "delivery_issue": re.compile(
        r"\b(stolen|theft|neighbour|neighbor|wrong\s+address|delivered\s+to\s+wrong)\b",
        re.IGNORECASE,
    ),
}


# ── Urgency detector ──────────────────────────────────────────────────────
def _urgency_signals(tweet: str) -> tuple[str, List[str]]:
    """
    Returns (urgency_level, list_of_matched_signals).
    urgency_level: "high" | "medium" | "low"
    """
    signals: List[str] = []

    legal_matches = _LEGAL_PATTERNS.findall(tweet)
    if legal_matches:
        signals.append(f"legal/media mention: {', '.join(set(legal_matches[:3]))}")

    security_matches = _SECURITY_PATTERNS.findall(tweet)
    if security_matches:
        signals.append(f"security keywords: {', '.join(set(security_matches[:3]))}")

    repetition_matches = _REPETITION_PATTERNS.findall(tweet)
    if repetition_matches:
        signals.append(f"repeated-contact language: '{repetition_matches[0]}'")

    neg_matches = _NEGATIVE_WORDS.findall(tweet)
    if len(neg_matches) >= NEGATIVITY_THRESHOLD:
        signals.append(f"high-negativity density: {len(neg_matches)} signals")

    # Level determination
    if legal_matches or security_matches:
        level = "high"
    elif len(signals) >= 2:
        level = "high"
    elif signals:
        level = "medium"
    else:
        level = "low"

    return level, signals


# ── Data class ────────────────────────────────────────────────────────────
@dataclass
class EscalationDecision:
    tweet: str
    intent: str
    confidence: float
    urgency_level: str           # "low" | "medium" | "high"
    urgency_signals: List[str]   # matched signal descriptions
    negative_count: int          # number of high-negativity keywords found
    triggered_rule: str          # which rule fired (or "none")
    decision: str                # "auto_handle" | "escalate"
    reason: str                  # human-readable, complete sentence

    def to_dict(self) -> dict:
        return asdict(self)


# ── Decision engine ───────────────────────────────────────────────────────
def decide(
    tweet: str,
    classification,          # ClassificationResult (typed loosely to avoid circular import)
) -> EscalationDecision:
    """
    Evaluate one tweet and return a binary escalation decision with reason.

    Rule priority (first matching rule wins):
    1. Classifier error / uncertain intent
    2. Confidence below threshold
    3. Legal / media / threat keywords (high urgency)
    4. Security breach keywords on account_access intent
    5. Repeated-contact language detected
    6. High-negativity density (≥2 signals)
    7. Intent-specific keyword rules
    8. Default: auto-handle
    """
    intent     = classification.intent
    confidence = classification.confidence
    error      = classification.error

    urgency_level, urgency_signals = _urgency_signals(tweet)
    neg_count = len(_NEGATIVE_WORDS.findall(tweet))

    # ── Rule 1: classifier error / uncertain ──────────────────────────────
    if error or intent == "uncertain":
        reason = (
            f"Classifier could not determine a confident intent "
            f"(confidence={confidence:.0%}). Human review required to avoid "
            f"a misdirected automated reply."
        )
        return EscalationDecision(
            tweet=tweet, intent=intent, confidence=confidence,
            urgency_level=urgency_level, urgency_signals=urgency_signals,
            negative_count=neg_count, triggered_rule="classifier_uncertain",
            decision="escalate", reason=reason,
        )

    # ── Rule 2: confidence below threshold ────────────────────────────────
    if confidence < CONFIDENCE_ESCALATE_THRESHOLD:
        reason = (
            f"Classifier confidence is {confidence:.0%} (below {CONFIDENCE_ESCALATE_THRESHOLD:.0%} "
            f"threshold). Auto-routing at this certainty level risks sending an "
            f"off-topic reply that could further frustrate the customer."
        )
        return EscalationDecision(
            tweet=tweet, intent=intent, confidence=confidence,
            urgency_level=urgency_level, urgency_signals=urgency_signals,
            negative_count=neg_count, triggered_rule="low_confidence",
            decision="escalate", reason=reason,
        )

    # ── Rule 3: legal / media threat (always escalate) ───────────────────
    legal_matches = _LEGAL_PATTERNS.findall(tweet)
    if legal_matches:
        matched = ", ".join(set(str(m) if isinstance(m, str) else m[0] for m in legal_matches[:2]))
        reason = (
            f"Customer mentions legal or media action ({matched}). "
            f"These cases must be routed to a senior support specialist — "
            f"automated replies carry reputational and legal risk."
        )
        return EscalationDecision(
            tweet=tweet, intent=intent, confidence=confidence,
            urgency_level="high", urgency_signals=urgency_signals,
            negative_count=neg_count, triggered_rule="legal_threat",
            decision="escalate", reason=reason,
        )

    # ── Rule 4: security breach + account_access ─────────────────────────
    security_matches = _SECURITY_PATTERNS.findall(tweet)
    if security_matches and intent == "account_access":
        matched = ", ".join(set(str(m) if isinstance(m, str) else m[0] for m in security_matches[:2]))
        reason = (
            f"Customer reports a potential account security breach ({matched}) "
            f"classified as account_access. This requires immediate action through "
            f"a secure, authenticated channel — not an automated public reply."
        )
        return EscalationDecision(
            tweet=tweet, intent=intent, confidence=confidence,
            urgency_level="high", urgency_signals=urgency_signals,
            negative_count=neg_count, triggered_rule="security_breach",
            decision="escalate", reason=reason,
        )

    # ── Rule 5: repeated contact frustration ─────────────────────────────
    rep_matches = _REPETITION_PATTERNS.findall(tweet)
    if rep_matches:
        reason = (
            f"Customer indicates repeated unresolved contact "
            f"('{str(rep_matches[0])[:60]}'). Auto-handling a customer who has "
            f"already tried standard channels risks further erosion of trust."
        )
        return EscalationDecision(
            tweet=tweet, intent=intent, confidence=confidence,
            urgency_level=urgency_level, urgency_signals=urgency_signals,
            negative_count=neg_count, triggered_rule="repeated_contact",
            decision="escalate", reason=reason,
        )

    # ── Rule 6: high-negativity density ──────────────────────────────────
    if neg_count >= NEGATIVITY_THRESHOLD:
        neg_found = _NEGATIVE_WORDS.findall(tweet)
        reason = (
            f"High-negativity language density detected ({neg_count} signals: "
            f"{', '.join(neg_found[:3])}). An automated reply is unlikely to "
            f"de-escalate a customer expressing this level of frustration."
        )
        return EscalationDecision(
            tweet=tweet, intent=intent, confidence=confidence,
            urgency_level=urgency_level, urgency_signals=urgency_signals,
            negative_count=neg_count, triggered_rule="high_negativity",
            decision="escalate", reason=reason,
        )

    # ── Rule 7: intent-specific keyword rules ────────────────────────────
    pattern = _INTENT_ESCALATE.get(intent)
    if pattern and pattern.search(tweet):
        m = pattern.search(tweet)
        reason = (
            f"Intent '{intent}' with keyword '{m.group()[:40]}' matches a known "
            f"high-sensitivity scenario that requires a personalised human response."
        )
        return EscalationDecision(
            tweet=tweet, intent=intent, confidence=confidence,
            urgency_level=urgency_level, urgency_signals=urgency_signals,
            negative_count=neg_count, triggered_rule=f"intent_rule:{intent}",
            decision="escalate", reason=reason,
        )

    # ── Rule 8: default — auto-handle ─────────────────────────────────────
    reason = (
        f"Intent classified as '{intent}' with {confidence:.0%} confidence. "
        f"No high-urgency signals detected. Retrieved grounding examples are available "
        f"to produce an accurate, on-topic automated reply."
    )
    return EscalationDecision(
        tweet=tweet, intent=intent, confidence=confidence,
        urgency_level=urgency_level, urgency_signals=urgency_signals,
        negative_count=neg_count, triggered_rule="none",
        decision="auto_handle", reason=reason,
    )


def decide_batch(
    items: List[tuple],   # each item: (tweet, ClassificationResult)
) -> List[EscalationDecision]:
    """Synchronous batch — escalation is CPU-only so no thread pool needed."""
    return [decide(tweet, clf) for tweet, clf in items]
