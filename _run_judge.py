"""
Judge model retry script: try candidates with exponential backoff.
Selects openai/gpt-oss-20b or nvidia/llama-3.1-nemotron-70b-instruct first.
Falls back to nemotron-3-super only as last resort with warning.
"""
import sys, os, time
sys.path.insert(0, "src")
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])

CANDIDATES = [
    "openai/gpt-oss-20b",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-3-super-120b-a12b",   # last resort
]

probe = '{"relevance": 4, "empathy": 4, "actionability": 4, "conciseness": 4, "overall": 4, "justification": "probe ok"}'
msg = f"Rate a reply. Output only: {probe}"

chosen = None
for candidate in CANDIDATES:
    for attempt in range(4):
        wait = 2 ** attempt * 5
        try:
            r = client.chat.completions.create(
                model=candidate, temperature=0, max_tokens=80,
                messages=[{"role": "user", "content": msg}],
                timeout=45,
            )
            print(f"SELECTED: {candidate} (attempt {attempt+1})")
            chosen = candidate
            break
        except Exception as e:
            print(f"  Attempt {attempt+1}/4 FAIL {candidate}: {str(e)[:80]}")
            if attempt < 3:
                print(f"  Waiting {wait}s...")
                time.sleep(wait)
    if chosen:
        break

if chosen:
    if chosen == "nvidia/nemotron-3-super-120b-a12b":
        print("WARNING: Using same-family judge (last resort). Setting JUDGE_MODEL env var.")
    os.environ["JUDGE_MODEL"] = chosen
    print(f"\nRunning 14_llm_judge.py with JUDGE_MODEL={chosen}")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/14_llm_judge.py", "--n", "30"],
        env={**os.environ, "JUDGE_MODEL": chosen, "PYTHONIOENCODING": "utf-8"},
        cwd=os.getcwd()
    )
    sys.exit(result.returncode)
else:
    print("FATAL: All judge candidates failed all retry attempts. NIM infrastructure issue.")
    sys.exit(1)
