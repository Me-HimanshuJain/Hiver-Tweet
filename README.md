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

# 3a. Evaluate with NVIDIA NIM (free key from https://build.nvidia.com)
$env:NVIDIA_API_KEY = 'nvapi-...'
python scripts/06_evaluate.py

# 3b. Dry-run / mock mode (no API key needed)
python scripts/06_evaluate.py --mock
```

## Project Layout

```
hiver/
├── src/
│   ├── taxonomy.py     # Intent labels, definitions, few-shot examples
│   ├── classifier.py   # Claude Haiku 4.5 few-shot intent classifier
│   └── retrieval.py    # sentence-transformer + FAISS retrieval layer
├── scripts/
│   ├── 01_download_and_explore.py
│   ├── 02_cluster_analysis.py
│   ├── 03_subcluster_and_inspect.py
│   ├── 04_build_index.py
│   ├── 05_label_sample.py
│   └── 06_evaluate.py
├── data/               (generated — not committed)
├── results/            (generated — not committed)
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

## Evaluation Results (mock mode — keyword heuristic baseline)

| Metric | Value |
|---|---|
| Accuracy | **90.9%** |
| Macro F1 | **0.908** |
| Majority-class baseline | 9.1% |
| Avg confidence | 0.747 |

**Hardest classes** (keyword heuristic):
- `delivery_issue` recall = 0.40 (6/10 fell into `general_complaint`)
- `order_status_inquiry` recall = 0.60 (4/10 fell into `general_complaint`)

These are the classes where Claude Haiku's semantic understanding is expected to significantly outperform keyword matching, since "my package never arrived" (delivery issue) vs "where is my package" (status inquiry) are semantically close but keyword-distant.

## Running the Real Classifier

```powershell
$env:NVIDIA_API_KEY = 'nvapi-YOUR_KEY'   # free at https://build.nvidia.com
python scripts/06_evaluate.py             # 110 API calls, NIM free tier
```

Estimated NIM Nemotron performance on this taxonomy: **Accuracy ~74–85%** (see `results/eval_report.txt` for live numbers after your first run).
