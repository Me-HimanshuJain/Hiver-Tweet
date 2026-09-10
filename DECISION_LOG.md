# Decision Log

### D1: Sentence-Transformers over OpenAI Embeddings
`all-MiniLM-L6-v2` runs locally and is free. Given the simplicity of the queries,
it provides sufficient semantic search capability for RAG without external API costs
or network latency.

### D2: FAISS IndexFlatIP (exact search), not HNSW (approximate)
With 8,000 vectors at 384 dimensions, exact search is <10ms per query and correct.
HNSW adds tuning parameters (efConstruction, M) with no practical benefit at this scale.

### D3: Keyword heuristic for auto-labeling eval set
Pragmatic choice to avoid spending human labeling budget on a set used only for
classifier eval. Known leakage risk documented. Human labels reserved for golden set.
Phase 10 finding: this was the correct call but the headline accuracy number should
have always been reported from the golden set, not the auto-labeled set.

### D4: Same model (nemotron-3-super) for both classifier and drafter
One model family reduces NIM endpoint management. Different system prompts per task.
Risk: same-model pattern learned in classification could influence drafting — mitigated
by completely separate API calls and prompts.

### D5: Deterministic 8-rule escalation, not learned classifier
189 golden examples (98 positive) is too small for reliable binary classifier training.
Rule engine is auditable (triggered_rule string per decision), zero latency, zero API cost.
Recall failure (3.1%) accepted as documented gap — honest reporting preferred over
overfitting a small training set.

### D6: Same-family judge model (nemotron-3-super) instead of cross-family (RESOLVED)
Original plan: meta/llama-3.1-70b-instruct (different family). At runtime: 410 Gone.
Decision: proceed with nemotron-3-super as judge rather than block the pipeline;
document same-family bias explicitly. This was a pragmatic compromise, not the design goal.
Phase 10 review confirms this must be fixed; run 2026-09-10 successfully used `openai/gpt-oss-20b` (cross-family) for the final metrics.

### D7: Dashboard reads JSON files, not a live API
Static JSON in public/data/ fetched at page load. A live backend adds ops complexity
with no evaluation benefit. Update cycle: write results/ → copy to public/data/. Explicit.

### D8: Single frontend (React), legacy Streamlit archived
Two dashboards existed mid-project. Streamlit superseded by React. Archived to legacy/.
All canonical JSON flows through results/ → frontend/public/data/.

### D9: Golden set stratified by diversity bucket, not intent frequency
Stratified by embedding distance from cluster center to ensure edge cases are represented.
Known consequence: general_complaint_or_feedback (54% of golden set) was not oversampled
proportionally; minority intents are relatively over-represented in the golden set.
This makes the golden set better for per-class analysis but means weighted F1 on it
does not directly reflect production class frequencies.

### D10: HF cache deleted in Phase 9 cleanup
~/.cache/huggingface/datasets/ (1.94 MB) deleted. 00_download_data.py re-downloads
on next run. Adds ~30 seconds to reproduction time. Documented.

### D11: src/ retained as active module directory, not archived
Phase 10 attempted to archive src/ with dashboard/. Scripts import from src/ at runtime
(classifier.py, retrieval.py, escalation.py, drafter.py, taxonomy.py). Only dashboard/
was moved to legacy/. src/ is an active dependency, not dead code.

### D12: Baseline comparison must use the same eval set as the full system (RESOLVED)
Phase 10 review finding: B1 (keyword) vs Full System comparison is only valid if both
are evaluated on the same ground-truth labels. The current baseline_comparison.json uses
labelled_eval.jsonl for the full system (eval_report.json) and golden_labeled.csv for B1.
This is an apples-to-oranges comparison. RESOLVED: scripts/15_baselines.py updated to
use golden_eval_report.json (full system run on golden set) against B1 (also on golden set).
