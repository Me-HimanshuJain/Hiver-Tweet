"""
drafter.py — Module 3: NIM-powered reply drafter.

Generates a Twitter-style customer support reply conditioned on the
classified intent and historically retrieved grounding examples.

Public API
----------
    draft(tweet, classification, retrieved) -> DraftedReply
    draft_batch(items, max_workers=2)       -> list[DraftedReply]

Every DraftedReply records exactly which retrieved examples were passed
to the model so the generation is fully auditable.

Environment
-----------
    NVIDIA_API_KEY — shared with classifier.py
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from openai import RateLimitError, APIStatusError

# ── Share the rate-limiter + client from classifier ────────────────────────
# Both modules use the same NIM endpoint and count against the same
# free-tier 40 req/min cap.  Importing the private helpers is deliberate
# — they share the global lock _last_call_ts / _rate_lock.
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from classifier import (
    _throttle, _client, _parse_structured,
    MODEL_ID, RETRY_ATTEMPTS, RETRY_BACKOFF, RETRY_503_BASE,
    ClassificationResult,
)
from retrieval import RetrievalResult

# ── Constants ─────────────────────────────────────────────────────────────
DRAFTER_MAX_TOKENS    = 400    # reply JSON is short
DRAFTER_TEMPERATURE   = 0.3   # slight creativity for natural replies
TWITTER_CHAR_LIMIT    = 280
MAX_GROUNDING_EXAMPLES = 3     # cap how many examples go into the prompt


# ── Prompt ────────────────────────────────────────────────────────────────
# Minimal 2-sentence prompt. No char limit (causes counting preambles). No
# markdown headers. JSON schema is the LAST token — Nemotron continues from
# it into JSON output, suppressing chain-of-thought preambles.
_DRAFTER_SYSTEM = (
    "You are an Amazon customer support agent. Write an empathetic, specific "
    "reply with a concrete next step. Do not copy examples verbatim. "
    'Output JSON: {"reply": "<reply text>", '
    '"tone": "empathetic|informational|apologetic|positive", '
    '"action_taken": "<one phrase>"}'
)

# Minimum / maximum reply length validation
_MIN_REPLY_CHARS = 20

# Tone heuristic — inferred from reply text, avoids extra API call
_TONE_EMPATHY   = re.compile(r"\b(sorry|apologize|apologi[sz]|understand|frustrat|trouble|inconveni)", re.I)
_TONE_APOLOGETIC = re.compile(r"\b(apologize|apologi[sz]|sincerely sorry|deeply sorry|regret)", re.I)
_TONE_POSITIVE  = re.compile(r"\b(great|happy|glad|thank|congratul|wonderful|excellent|pleasure)", re.I)


def _infer_tone(reply: str) -> str:
    if _TONE_APOLOGETIC.search(reply):
        return "apologetic"
    if _TONE_EMPATHY.search(reply):
        return "empathetic"
    if _TONE_POSITIVE.search(reply):
        return "positive"
    return "informational"


def _build_user_message(
    tweet: str,
    intent: str,
    retrieved: List[RetrievalResult],
) -> str:
    """Pure-data user message — no trailing instructions that the model echoes."""
    lines = [
        f"Customer intent: {intent}",
        f"Customer tweet: {tweet}",
        "",
    ]
    examples = retrieved[:MAX_GROUNDING_EXAMPLES]
    if examples:
        lines.append("Similar resolved cases (for tone/approach reference only):")
        for i, ex in enumerate(examples, 1):
            lines.append(f"  {i}. Customer: {ex.customer_tweet[:130]!r}")
            lines.append(f"     Reply:    {ex.brand_reply[:130]!r}")
    return "\n".join(lines)





# ── Data class ────────────────────────────────────────────────────────────
@dataclass
class DraftedReply:
    tweet: str
    intent: str
    confidence: float
    grounding_examples: List[dict]      # serialised RetrievalResult dicts
    n_grounding_examples: int           # how many examples were used
    reply: str                          # the generated reply text
    reply_length: int                   # character count
    tone: str                           # empathetic | informational | apologetic | positive
    action_taken: str                   # one-phrase description
    model: str
    latency_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _call_drafter(
    tweet: str,
    intent: str,
    retrieved: List[RetrievalResult],
) -> tuple[dict, float, List[RetrievalResult]]:
    """
    Call NIM to draft a plain-text reply (no JSON format required).
    Returns (result_dict, latency_ms, examples_used).
    Retries on API errors AND on too-short / placeholder responses.
    """
    examples_used = retrieved[:MAX_GROUNDING_EXAMPLES]
    user_msg = _build_user_message(tweet, intent, examples_used)
    last_err: Exception = RuntimeError("No attempts made")

    for attempt in range(RETRY_ATTEMPTS):
        _throttle()   # shared rate limiter with classifier
        try:
            t0 = time.perf_counter()
            response = _client().chat.completions.create(
                model=MODEL_ID,
                temperature=DRAFTER_TEMPERATURE,
                max_tokens=DRAFTER_MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _DRAFTER_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = (response.choices[0].message.content or "").strip()

            # _parse_plain_reply handles JSON, preamble+reply, and plain prose.
            # If model returns valid JSON (json_object mode), step 2 extracts it.
            # If it returns preamble+reply, steps 1/3 extract the reply paragraph.
            parsed = _parse_plain_reply(raw)
            return parsed, latency_ms, examples_used

        except RateLimitError as exc:
            last_err = exc
            wait = RETRY_BACKOFF * (2 ** attempt) + (attempt + 1) * 0.5
            print(f"\n  [drafter] Rate limit (attempt {attempt+1}), sleeping {wait:.1f}s…",
                  flush=True)
            time.sleep(wait)

        except APIStatusError as exc:
            last_err = exc
            if exc.status_code == 503:
                wait = RETRY_503_BASE * (2 ** attempt) + (attempt + 1) * 1.0
                print(f"\n  [drafter] 503 (attempt {attempt+1}), sleeping {wait:.1f}s…",
                      flush=True)
            else:
                wait = RETRY_BACKOFF * (2 ** attempt) + (attempt + 1) * 0.3
                print(f"\n  [drafter] API {exc.status_code} (attempt {attempt+1}), "
                      f"sleeping {wait:.1f}s…", flush=True)
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(wait)

        except ValueError as exc:
            last_err = exc
            wait = RETRY_BACKOFF + attempt * 1.5
            print(f"\n  [drafter] Reply invalid (attempt {attempt+1}): {exc!s:.60s} "
                  f"— retrying in {wait:.0f}s…", flush=True)
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(wait)

    raise RuntimeError(f"All {RETRY_ATTEMPTS} drafter attempts failed: {last_err}")


def _parse_plain_reply(raw: str) -> dict:
    """
    Extract the best reply candidate from a potentially verbose model response.

    Strategy (in order):
    1. If response starts after 'Reply:' or 'Amazon:' label, use that.
    2. If response starts with a preamble paragraph, skip it and use
       the next non-empty paragraph.
    3. Use the last paragraph that fits Twitter length as the reply.
    4. Raise ValueError only if nothing usable was found.
    """
    if not raw:
        raise ValueError("Empty drafter response")

    text = raw.strip()

    # 1. Check if model included the 'Reply:' label in its response
    for label in ("Reply:", "Amazon:", "Response:"):
        if label.lower() in text.lower():
            idx = text.lower().index(label.lower()) + len(label)
            candidate = text[idx:].strip().strip('"\' ').strip()
            if len(candidate) >= _MIN_REPLY_CHARS:
                return _finalise(candidate)

    # 1.5. Regex-extract "reply" field — handles malformed/nested JSON like
    #      {{"reply":"..."}} that json.loads rejects but the field is still present.
    reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if reply_match:
        candidate = reply_match.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
        if len(candidate) >= _MIN_REPLY_CHARS:
            return _finalise(candidate)

    # 2. If model produced well-formed JSON, extract reply field gracefully
    if re.match(r"^\s*\{", text):
        try:
            obj = json.loads(text)
            candidate = str(obj.get("reply", "")).strip()
            if len(candidate) >= _MIN_REPLY_CHARS:
                return _finalise(candidate)
        except json.JSONDecodeError:
            pass

    # 3. Split into paragraphs and pick best candidate
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    PREAMBLE_PAT = re.compile(
        r'^(we need to|i need to|i will|i should|let me|here is|here\'s|'
        r'as an amazon|as a customer support|rules:|rule:|output:|'
        r'the following|this is a|to draft|to write|your task|'
        r'count:|string:|start:|check length|let\'s count|counting)',
        re.IGNORECASE,
    )

    candidates = [
        p for p in paragraphs
        if not PREAMBLE_PAT.match(p)
        and _MIN_REPLY_CHARS <= len(p) <= TWITTER_CHAR_LIMIT * 2
    ]

    if candidates:
        # Prefer the last qualifying candidate (most likely to be the reply
        # rather than an intermediate reasoning step)
        return _finalise(candidates[-1])

    # 4. Nothing usable — raise so the caller uses the fallback
    raise ValueError(f"No usable reply found in response (len={len(text)}): {text[:120]!r}")


def _finalise(reply: str) -> dict:
    """Strip, truncate, validate, infer tone/action for a raw reply string."""
    reply = reply.strip('"\' ').strip()
    # Remove 'Reply:' prefix if model echoed it
    reply = re.sub(r'^(?:reply|response|amazon)[:\s]+', '', reply, flags=re.IGNORECASE).strip()

    if set(reply) <= set('. -_\n'):
        raise ValueError(f"Placeholder reply: {reply!r}")
    if len(reply) < _MIN_REPLY_CHARS:
        raise ValueError(f"Reply too short ({len(reply)} chars): {reply!r}")

    if len(reply) > TWITTER_CHAR_LIMIT:
        reply = reply[:TWITTER_CHAR_LIMIT - 1] + "…"

    tone = _infer_tone(reply)
    action = _summarise_action(reply)

    return {"reply": reply, "tone": tone, "action_taken": action}


def _summarise_action(reply: str) -> str:
    """Infer a one-phrase action description from the reply text."""
    low = reply.lower()
    if "dm" in low or "direct message" in low or "private" in low:
        return "directed to DM for resolution"
    if "call" in low or "phone" in low:
        return "directed to call support"
    if "chat" in low:
        return "directed to live chat"
    if "track" in low or "carrier" in low:
        return "offered to check tracking"
    if "refund" in low:
        return "offered refund investigation"
    if "order number" in low or "order #" in low:
        return "requested order details"
    if "password" in low or "account" in low:
        return "directed to account recovery"
    if re.search(r"\b(restart|reset|charge|wifi|update)", low):
        return "offered device troubleshooting"
    return "provided next-step guidance"





# ── Public API ────────────────────────────────────────────────────────────
def draft(
    tweet: str,
    classification: ClassificationResult,
    retrieved: List[RetrievalResult],
) -> DraftedReply:
    """
    Draft a reply for a single tweet.

    Parameters
    ----------
    tweet          : raw tweet text
    classification : output of classify()
    retrieved      : output of RetrievalIndex.query()

    Returns
    -------
    DraftedReply — always returns; errors captured in .error
    """
    intent = classification.intent
    if intent == "uncertain":
        intent = "general_complaint_or_feedback"

    try:
        parsed, latency_ms, examples_used = _call_drafter(tweet, intent, retrieved)
        return DraftedReply(
            tweet=tweet,
            intent=intent,
            confidence=classification.confidence,
            grounding_examples=[r.to_dict() for r in examples_used],
            n_grounding_examples=len(examples_used),
            reply=parsed["reply"],
            reply_length=len(parsed["reply"]),
            tone=parsed["tone"],
            action_taken=parsed["action_taken"],
            model=MODEL_ID,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        # Fallback reply that is always safe to send
        fallback = (
            "Hi! We're sorry you're experiencing this issue. Please DM us "
            "your order number and we'll look into it right away."
        )
        return DraftedReply(
            tweet=tweet,
            intent=intent,
            confidence=classification.confidence,
            grounding_examples=[r.to_dict() for r in retrieved[:MAX_GROUNDING_EXAMPLES]],
            n_grounding_examples=min(len(retrieved), MAX_GROUNDING_EXAMPLES),
            reply=fallback,
            reply_length=len(fallback),
            tone="empathetic",
            action_taken="fallback — drafter error",
            model=MODEL_ID,
            latency_ms=0.0,
            error=str(exc),
        )


def draft_batch(
    items: List[tuple],   # each item: (tweet, ClassificationResult, List[RetrievalResult])
    max_workers: int = 2,
    progress: bool = True,
) -> List[DraftedReply]:
    """
    Draft replies for a batch of (tweet, classification, retrieved) tuples.
    max_workers=2 keeps combined classifier + drafter within NIM 40 req/min cap.
    """
    results: dict[int, DraftedReply] = {}
    total = len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(draft, tweet, clf, ret): i
            for i, (tweet, clf, ret) in enumerate(items)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            if progress:
                done = len(results)
                print(f"\r  Drafted {done}/{total} ({done/total:.0%})",
                      end="", flush=True)

    if progress:
        print()
    return [results[i] for i in range(total)]
