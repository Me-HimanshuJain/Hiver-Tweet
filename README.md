# AmazonHelp Intent System — Hiver SDE Intern Assignment

## Quickstart (under 15 minutes from a clean clone)

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data + build retrieval index  (~3 min, downloads ~170 MB once)
python scripts/01_download_and_explore.py   # sample 3 000 AmazonHelp tweets
python scripts/02_cluster_analysis.py        # K-Means taxonomy derivation
python scripts/04_build_index.py             # build FAISS index (8 000 threads)
python scripts/05_label_sample.py            # create 110-tweet eval set

# 3a. Evaluate with NVIDIA NIM on the golden set (Headline result)
$env:NVIDIA_API_KEY = 'nvapi-...'
python scripts/06_evaluate.py --golden

# 3b. Evaluate on auto-labeled set (faster, but label-leakage caveat applies)
python scripts/06_evaluate.py

# 3c. Dry-run / mock mode (no API key needed)
python scripts/06_evaluate.py --mock
```

## Project Layout

```
hiver/
├── src/
│   ├── taxonomy.py     # Intent labels, definitions, few-shot examples
│   ├── classifier.py   # nvidia/nemotron-3-super-120b-a12b few-shot intent classifier
│   └── retrieval.py    # sentence-transformer + FAISS retrieval layer
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_download_and_explore.py
│   ├── 02_cluster_analysis.py
│   ├── 03_subcluster_and_inspect.py
│   ├── 04_build_index.py
│   ├── 05_label_sample.py
│   ├── 06_evaluate.py
│   └── 07_retry_failed.py to 16_eval_report.py
├── data/               (generated — not committed)
├── results/            (generated — not committed)
├── frontend/           (React dashboard)
├── legacy/             (Archived Streamlit dashboard)
└── requirements.txt
```

## Module APIs

### `src/classifier.py`
```python
from src.classifier import classify, classify_batch

# Requires: $env:NVIDIA_API_KEY = 'nvapi-...'
# Model: nvidia/nemotron-3-super-120b-a12b  (NIM free tier)
result = classify("@AmazonHelp where is my order?")
# result.intent      → "order_status_inquiry"
# result.confidence  → 0.97
# result.reasoning   → "Customer asks about order location/status"
# result.alternatives → [Alternative(intent="...", confidence=0.02)]
# result.is_uncertain → False  (confidence < 0.50)
# result.latency_ms  → ~2000ms (NIM free tier)

# Batch — throttled to respect 40 req/min free-tier limit
batch = classify_batch(tweets, max_workers=2)
```

### `src/retrieval.py`
```python
from src.retrieval import RetrievalIndex

idx = RetrievalIndex.load("data/faiss_index.bin", "data/faiss_meta.jsonl")

results = idx.query(
    tweet="where is my order",
    intent="order_status_inquiry",  # optional filter
    top_k=3
)
# results[0].customer_tweet → "@AmazonHelp Where Is My Order?..."
# results[0].brand_reply    → "Hi there, we see your order is still..."
# results[0].similarity     → 0.733
```

## Active Paths Reference

| Path | Status | Purpose |
|---|---|---|
| `scripts/` | **Active** — all evaluation entry points | Pipeline scripts 00–16 |
| `src/` | **Active** — imported at runtime by scripts | classifier, retrieval, drafter, escalation, taxonomy |
| `data/` | **Generated/committed** | Raw inputs + golden set (see .gitignore for what's excluded) |
| `results/` | **Generated** | All canonical metric outputs |
| `frontend/` | **Active** | React dashboard reads `public/data/*.json` |
| `legacy/` | **Archived** | Streamlit dashboard (superseded); not in any import path |

## Final Metrics

See [`PROJECT_STATE.md`](./PROJECT_STATE.md) for the full authoritative snapshot.
Quick reference (golden-set–based where applicable):

| Metric | Value | Source |
|---|---|---|
| Full system accuracy (golden set, n=189) | **38.6%** | scripts/06_evaluate.py --golden |
| Full system accuracy (auto-label set, n=110) | **64.6%** (Secondary/biased) | scripts/06_evaluate.py |
| Keyword baseline accuracy (golden set, n=189) | **52.9%** | scripts/15_baselines.py |
| Retrieval hit@3 | **0.619** | scripts/13_automated_metrics.py |
| Escalation F1 | **0.058** (P=0.50, R=0.031) | scripts/13_automated_metrics.py |
| LLM judge overall/5 (n=30) | **4.41** | scripts/14_llm_judge.py |
| Banking77 cross-domain accuracy | **40.0%** (vs 38.6% in-domain) | scripts/09_banking77_cross_domain.py |

> NOTE: For a full breakdown of why the 64.6% auto-labeled number is misleading and why the human-labeled golden set (38.6%) is the correct primary metric, see [PROJECT_STATE.md Section 4](./PROJECT_STATE.md#4-whats-misleading-about-my-headline-number).

## Running the Real Classifier

```powershell
$env:NVIDIA_API_KEY = 'nvapi-YOUR_KEY'   # free at https://build.nvidia.com
python scripts/06_evaluate.py             # 110 API calls, NIM free tier
# → results/eval_report.json + eval_report.txt
```

Full reproduction sequence is in PROJECT_STATE.md §7.
