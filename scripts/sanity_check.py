"""Quick sanity check for the final classifier fixes."""
import sys, json
sys.path.insert(0, 'src')
from classifier import (
    MAX_TOKENS, RETRY_ATTEMPTS, _RESPONSE_FORMAT,
    _preprocess, _parse_structured, Alternative,
)

print(f"MAX_TOKENS     = {MAX_TOKENS}")
print(f"RETRY_ATTEMPTS = {RETRY_ATTEMPTS}")
print(f"response_format= {_RESPONSE_FORMAT}")

# preprocess: strips ticket IDs and @AmazonHelp, keeps real content
cleaned = _preprocess('@AmazonHelp @119351 I want a refund')
assert '@119351' not in cleaned and 'refund' in cleaned
print("_preprocess: OK")

# alternatives as list-of-strings
raw1 = json.dumps({'intent':'refund_request','confidence':0.92,'reasoning':'wants refund',
                   'alternatives':['delivery_issue','order_status_inquiry']})
p1 = _parse_structured(raw1)
assert p1['intent'] == 'refund_request'
assert len(p1['alternatives']) == 2
assert isinstance(p1['alternatives'][0], Alternative)
intents = [a.intent for a in p1['alternatives']]
print(f"  list-of-str alternatives: {intents}  OK")

# prose preamble
raw3 = 'Here is my analysis: {"intent": "account_access", "confidence": 0.85, "reasoning": "locked out", "alternatives": []}'
p3 = _parse_structured(raw3)
assert p3['intent'] == 'account_access'
print(f"  prose preamble stripped: intent={p3['intent']}  OK")

# empty response
try:
    _parse_structured('')
    assert False, "should have raised"
except ValueError:
    print("  empty response raises ValueError  OK")

# truncated JSON repair
raw2 = '{"intent": "prime_membership", "confidence"'
try:
    p2 = _parse_structured(raw2)
    print(f"  truncated JSON repaired: intent={p2['intent']}  OK")
except ValueError as e:
    print(f"  truncated JSON unrecoverable (expected fallback): {str(e)[:60]}")

print("\nAll checks passed.")
