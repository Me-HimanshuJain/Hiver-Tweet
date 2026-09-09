"""
classifier.py — Module 1: NVIDIA NIM (Nemotron) few-shot intent classifier.

Uses the OpenAI-compatible NIM endpoint with native Structured Output
(JSON Schema response_format) — no manual JSON parsing needed.

Public API
----------
    classify(tweet: str) -> ClassificationResult
    classify_batch(tweets: list[str], max_workers: int = 2) -> list[ClassificationResult]

Each ClassificationResult carries:
    intent        : str   — one of the 11 taxonomy labels (or "uncertain")
    confidence    : float — self-reported model confidence in [0, 1]
    reasoning     : str   — one-sentence explanation
    alternatives  : list  — up to 2 runner-up {intent, confidence} dicts
    raw_response  : str   — raw JSON string from the API (for debugging)
    latency_ms    : float — round-trip time in milliseconds
    error         : str | None — set on irrecoverable failures

Environment
-----------
    NVIDIA_API_KEY  — required; obtain a free key at build.nvidia.com
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from openai import OpenAI, RateLimitError, APIStatusError

# ── Import taxonomy from sibling module ───────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from taxonomy import INTENTS, FEW_SHOT_EXAMPLES, get_intent_definitions_block

# ── Constants ─────────────────────────────────────────────────────────────
NIM_BASE_URL   = "https://integrate.api.nvidia.com/v1"
MODEL_ID       = "nvidia/nemotron-3-super-120b-a12b"
TEMPERATURE    = 0.0           # deterministic
MAX_TOKENS     = 1000          # 700 still too tight for some verbose responses
CONFIDENCE_THRESHOLD = 0.50   # below → "uncertain"
RETRY_ATTEMPTS = 5
RETRY_BACKOFF  = 3.0          # seconds, doubles per retry
RETRY_503_BASE = 10.0         # 503 = server overload — back off much harder

# ── NIM rate-limit throttle ────────────────────────────────────────────────
# Free tier: ~40 requests/minute shared across all workers.
# MIN_INTERVAL enforces ≥ 1.6 s between any two outgoing calls globally,
# giving ~37 req/min headroom.  max_workers=2 keeps burst safe.
MIN_INTERVAL   = 1.6           # seconds between calls
_rate_lock     = threading.Lock()
_last_call_ts  = [0.0]         # mutable singleton — shared across threads


def _throttle() -> None:
    """Block the caller until it is safe to fire the next API call."""
    with _rate_lock:
        now = time.perf_counter()
        gap = MIN_INTERVAL - (now - _last_call_ts[0])
        if gap > 0:
            time.sleep(gap)
        _last_call_ts[0] = time.perf_counter()


# ── JSON Schema for structured output ────────────────────────────────────
# NIM honours OpenAI's response_format={"type": "json_schema", ...} spec.
# Defining the schema here means the model MUST return valid JSON that
# matches — no fallback regex parsing needed.
# Use json_object (not json_schema) — universally supported across NIM model versions.
# The schema is still described in the system prompt so the model knows the exact
# fields and enum values to produce.
_RESPONSE_FORMAT = {"type": "json_object"}

# ── System prompt ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "IMPORTANT: Respond with ONLY a valid JSON object — no preamble, no explanation, "
    "no text before or after the JSON.\n\n"
    "You are an intent classification engine for Amazon customer support tweets.\n\n"
    "## Task\n"
    "Given a customer tweet directed at @AmazonHelp, return a JSON object classifying "
    "it into exactly ONE of the intent categories below.\n\n"
    "## Intent Categories\n"
    + get_intent_definitions_block()
    + "\n\n## Output JSON Schema (respond with ONLY this object):\n"
    '{"intent": "<one of the 11 labels>", "confidence": <float 0-1>, '
    '"reasoning": "<≤15 words>", "alternatives": [{"intent": "<label>", "confidence": <float>}]}\n\n'
    "## Rules\n"
    "- intent: choose the single most specific, actionable label. Prefer a specific label "
    "over general_complaint_or_feedback whenever ANY specific signal is present.\n"
    "- confidence: lower when ambiguous; use 0.0–0.49 only when genuinely unsure.\n"
    "- reasoning: maximum 15 words describing the key signal.\n"
    "- alternatives: 0–2 entries only; use [] if no credible second choice.\n"
    "- Non-English tweet: classify by meaning, set confidence ≤ 0.50.\n\n"
    "## Few-shot Examples\n"
    + "\n".join(
        f'Tweet: "{tweet}"\n'
        f'{{"intent": "{label}", "confidence": 0.95, '
        f'"reasoning": "Customer requests {label.replace("_", " ")}.", '
        f'"alternatives": []}}'
        for tweet, label in FEW_SHOT_EXAMPLES[:10]
    )
)


# ── Data classes ──────────────────────────────────────────────────────────
@dataclass
class Alternative:
    intent: str
    confidence: float


@dataclass
class ClassificationResult:
    tweet: str
    intent: str
    confidence: float
    reasoning: str
    alternatives: List[Alternative] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def is_uncertain(self) -> bool:
        return self.confidence < CONFIDENCE_THRESHOLD or self.intent == "uncertain"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_uncertain"] = self.is_uncertain
        return d


# ── Client (lazy singleton) ───────────────────────────────────────────────
_CLIENT: Optional[OpenAI] = None


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        # User: Paste your API key below! e.g., api_key = "nvapi-..."
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "NVIDIA_API_KEY not set.\n"
                "Get a free key at https://build.nvidia.com then run:\n"
                "  $env:NVIDIA_API_KEY = 'nvapi-...'\n"
            )
        _CLIENT = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)
    return _CLIENT


# ── Core NIM call ─────────────────────────────────────────────────────────
def _preprocess(tweet: str) -> str:
    """
    Clean a tweet before sending to NIM.

    - Remove purely numeric @mentions (support-ticket IDs like @119351, @275233).
      These look like anonymous handles and trigger NIM content filtering.
    - Remove @AmazonHelp (already implied; reduces prompt length).
    - Collapse repeated whitespace.
    Keep all other real @handles so context is preserved.
    """
    text = re.sub(r"@\d+\b", "", tweet)               # ticket IDs
    text = re.sub(r"@AmazonHelp\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _call_nim(tweet: str) -> tuple[dict, float]:
    """
    Fire a NIM request and parse the response.
    Retries on API errors (503, rate limit) AND on parse errors.
    Returns (parsed_dict, latency_ms).
    Raises RuntimeError after all attempts are exhausted.
    """
    user_msg = f'Tweet: "{_preprocess(tweet)}"'
    last_err: Exception = RuntimeError("No attempts made")

    for attempt in range(RETRY_ATTEMPTS):
        _throttle()
        raw = ""
        try:
            t0 = time.perf_counter()
            response = _client().chat.completions.create(
                model=MODEL_ID,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format=_RESPONSE_FORMAT,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = (response.choices[0].message.content or "").strip()

            # Parse inside the retry loop so bad JSON triggers a retry
            parsed = _parse_structured(raw)
            return parsed, latency_ms

        except RateLimitError as exc:
            last_err = exc
            jitter = (attempt + 1) * 0.5
            wait = RETRY_BACKOFF * (2 ** attempt) + jitter
            print(f"\n  [classifier] Rate limit (attempt {attempt+1}/{RETRY_ATTEMPTS}), "
                  f"sleeping {wait:.1f}s…", flush=True)
            time.sleep(wait)

        except APIStatusError as exc:
            last_err = exc
            if exc.status_code == 503:
                jitter = (attempt + 1) * 1.0
                wait = RETRY_503_BASE * (2 ** attempt) + jitter
                print(f"\n  [classifier] 503 overload (attempt {attempt+1}/{RETRY_ATTEMPTS}), "
                      f"sleeping {wait:.1f}s…", flush=True)
            else:
                jitter = (attempt + 1) * 0.3
                wait = RETRY_BACKOFF * (2 ** attempt) + jitter
                print(f"\n  [classifier] API {exc.status_code} (attempt {attempt+1}/{RETRY_ATTEMPTS}), "
                      f"sleeping {wait:.1f}s…", flush=True)
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(wait)

        except (ValueError, KeyError, TypeError) as exc:
            # Parse / format error — retry with short backoff
            last_err = exc
            wait = RETRY_BACKOFF + attempt * 1.5
            print(f"\n  [classifier] Parse error (attempt {attempt+1}/{RETRY_ATTEMPTS}): "
                  f"{exc!s:.60s} — retrying in {wait:.0f}s…", flush=True)
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(wait)

    raise RuntimeError(
        f"All {RETRY_ATTEMPTS} NIM attempts failed: {last_err}"
    )


def _parse_structured(raw: str) -> dict:
    """
    Parse the JSON returned by NIM.
    Handles: markdown fences, prose preamble, truncated JSON, and the
    model quirk of returning alternatives as a flat list of strings.
    """
    if not raw:
        raise ValueError("Empty response from NIM")

    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fallback 1: extract first { ... } block (handles prose preamble)
        match = re.search(r"\{.*", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object in response: {raw[:120]}")
        fragment = match.group()
        # Fallback 2: try to repair truncated JSON by appending common closings
        repaired = False
        for closing in ["", '"}"}', '","alternatives":[]}', '"}', "}"]:
            try:
                parsed = json.loads(fragment + closing)
                repaired = True
                break
            except json.JSONDecodeError:
                continue
        if not repaired:
            raise ValueError(f"Unrecoverable JSON: {raw[:120]}")

    intent = str(parsed.get("intent", "")).strip().lower().replace(" ", "_")
    if intent not in INTENTS:
        intent = "general_complaint_or_feedback"

    confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
    reasoning  = str(parsed.get("reasoning", "")).strip()

    # Alternatives: handle both list-of-dicts AND list-of-strings (model quirk)
    raw_alts = parsed.get("alternatives") or []
    alternatives = []
    for a in raw_alts[:2]:
        if isinstance(a, str):
            # Model returned ["intent_name", ...] — normalise
            a_intent = a.lower().replace(" ", "_")
            if a_intent in INTENTS:
                alternatives.append(Alternative(intent=a_intent, confidence=0.0))
        elif isinstance(a, dict):
            a_intent = str(a.get("intent", "")).lower().replace(" ", "_")
            if a_intent in INTENTS:
                alternatives.append(Alternative(
                    intent=a_intent,
                    confidence=float(a.get("confidence", 0.0)),
                ))

    return {
        "intent":       intent,
        "confidence":   confidence,
        "reasoning":    reasoning,
        "alternatives": alternatives,
    }


# ── Public API ────────────────────────────────────────────────────────────
def classify(tweet: str) -> ClassificationResult:
    """
    Classify a single tweet using NVIDIA NIM (Nemotron).

    Parameters
    ----------
    tweet : str  — raw tweet text (with @mentions, hashtags, etc.)

    Returns
    -------
    ClassificationResult — always returns; errors captured in .error
    """
    raw, latency_ms = "", 0.0
    try:
        parsed, latency_ms = _call_nim(tweet)  # parses inside the retry loop

        intent     = parsed["intent"]
        confidence = parsed["confidence"]
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "uncertain"

        return ClassificationResult(
            tweet=tweet,
            intent=intent,
            confidence=confidence,
            reasoning=parsed["reasoning"],
            alternatives=parsed["alternatives"],
            raw_response="",          # raw string no longer returned by _call_nim
            latency_ms=latency_ms,
        )

    except Exception as exc:
        return ClassificationResult(
            tweet=tweet,
            intent="uncertain",
            confidence=0.0,
            reasoning="",
            alternatives=[],
            raw_response=raw,
            error=str(exc),
            latency_ms=latency_ms,
        )



def classify_batch(
    tweets: List[str],
    max_workers: int = 2,      # ← throttled for NIM free tier (~40 req/min)
    progress: bool = True,
) -> List[ClassificationResult]:
    """
    Classify a list of tweets.  Safe for the NIM free-tier rate limit.

    max_workers=2 + MIN_INTERVAL=1.6s → ~37 req/min, well under 40.

    Parameters
    ----------
    tweets      : list of raw tweet strings
    max_workers : parallel workers (default 2 — do not raise above 3 on free tier)
    progress    : print a live counter

    Returns
    -------
    List[ClassificationResult] in the same order as input.
    """
    results: dict[int, ClassificationResult] = {}
    total = len(tweets)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(classify, tweet): i
            for i, tweet in enumerate(tweets)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if progress:
                pct = done / total * 100
                print(f"\r  Classified {done}/{total} ({pct:.0f}%)", end="", flush=True)

    if progress:
        print()
    return [results[i] for i in range(total)]


# ── CLI smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_TWEETS = [
        "@AmazonHelp where is my order? It's been 5 days and no update.",
        "@AmazonHelp my prime video app keeps crashing on my Fire Stick.",
        "@AmazonHelp I want to return the headphones I bought yesterday.",
        "@AmazonHelp you guys have the absolute worst customer service!!!",
        "@AmazonHelp my account got locked and I can't reset my password.",
        "@AmazonHelp my prime membership renewed and charged wrong card.",
        "@AmazonHelp package says delivered but I never received it.",
    ]

    print(f"NIM Classifier smoke test — {len(TEST_TWEETS)} tweets")
    print(f"Model    : {MODEL_ID}")
    print(f"Throttle : {MIN_INTERVAL}s between calls, max_workers=2")
    print(f"Prompt   : {len(_SYSTEM_PROMPT):,} chars\n")

    for tweet in TEST_TWEETS:
        r = classify(tweet)
        status = "UNCERTAIN" if r.is_uncertain else "OK"
        print(f"[{status}] {tweet[:72]}")
        print(f"       intent     : {r.intent}")
        print(f"       confidence : {r.confidence:.2f}")
        print(f"       reasoning  : {r.reasoning}")
        if r.alternatives:
            alts = ", ".join(
                f"{a.intent}({a.confidence:.2f})" for a in r.alternatives
            )
            print(f"       alternatives: {alts}")
        if r.error:
            print(f"       ERROR: {r.error}")
        print(f"       latency    : {r.latency_ms:.0f}ms\n")
