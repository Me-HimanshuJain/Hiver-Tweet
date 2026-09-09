"""Quick unit test for _parse_plain_reply and _build_user_message."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drafter import _parse_plain_reply, _build_user_message, _DRAFTER_SYSTEM
from retrieval import RetrievalResult

# ── Parser tests ──────────────────────────────────────────────────────────
CASES = [
    # (name, raw_model_response, should_pass, expected_reply_fragment)
    ("preamble_only",
     "We need to produce a Twitter reply as Amazon customer support agent. "
     "Must be <=280 chars, empathetic, specific concrete next step.",
     False, None),   # only preamble, no usable reply

    ("preamble_then_reply",
     "We need to produce a Twitter reply as Amazon customer support agent. Must be <=280 chars.\n\n"
     "I understand your frustration! DM us your order number and we'll track it down. ^AH",
     True, "frustration"),

    ("with_reply_label",
     "We need to think about this.\n\nReply: I understand the delay is frustrating. "
     "DM us your order number and we'll get this sorted. ^AH",
     True, "frustrating"),

    ("json_response",
     '{"reply": "Hi! I understand the delay. DM us your order number and we\'ll track it. ^AH", '
     '"tone": "empathetic", "action_taken": "request order details"}',
     True, "delay"),

    ("short_direct",
     "I understand the delay is frustrating. Please DM us your order number "
     "and we will get this sorted for you right away. ^AH",
     True, "frustrating"),

    ("counting_preamble_then_reply",
     "Count: Let's count characters.\n@AmazonHelp (11) space (1) = 12...\n\n"
     "I'm sorry your Prime delivery arrived 6 days late. That's not acceptable — "
     "please DM us your order number and we'll investigate the delay and your Prime benefits. ^AH",
     True, "late"),

    ("json_with_counting_preamble",
     'Let\'s count to verify: "@User" = 5...\n\n'
     '{"reply": "So sorry about the delay! DM us your order number to investigate. ^AH", '
     '"tone": "empathetic", "action_taken": "request order details"}',
     True, "delay"),

]

passed = failed = 0
for name, raw, should_pass, fragment in CASES:
    try:
        result = _parse_plain_reply(raw)
        if should_pass:
            ok = fragment is None or fragment in result["reply"]
            status = "PASS" if ok else "FAIL (wrong content)"
            print(f"{status} [{name}]: {result['reply'][:70]!r}  tone={result['tone']}")
            if ok:
                passed += 1
            else:
                failed += 1
        else:
            print(f"FAIL [{name}]: expected ValueError but got reply: {result['reply'][:60]!r}")
            failed += 1
    except ValueError as e:
        if not should_pass:
            print(f"PASS [{name}]: correctly raised ValueError: {e!s:.60}")
            passed += 1
        else:
            print(f"FAIL [{name}]: unexpected ValueError: {e!s:.80}")
            failed += 1

print()
print(f"{passed}/{passed+failed} parser tests passed")

# ── User message trigger test ─────────────────────────────────────────────
fake_results = [
    RetrievalResult("1", "tweet1", "reply1", "refund_request", 0.9),
]
# Verify user message structure (no trailing instructions)
msg = _build_user_message("Where is my order?", "order_status_inquiry", fake_results)
assert "Similar resolved cases" in msg, "Missing grounding examples section"
# The bare "Reply:" trigger must NOT appear as its own line (it was causing counting preambles)
assert "Reply:" not in [l.strip() for l in msg.splitlines()], \
    "Bare 'Reply:' trigger line should not be in user message"
print("User message structure: PASS")



print()
print("System prompt:")
print(_DRAFTER_SYSTEM)
