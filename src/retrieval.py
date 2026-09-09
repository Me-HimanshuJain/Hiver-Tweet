"""
retrieval.py — Module 2: Embedding + FAISS vector retrieval over resolved
AmazonHelp threads.

Public API
----------
    build_index(threads: list[dict], index_path: str, meta_path: str) -> None
    load_index(index_path: str, meta_path: str) -> RetrievalIndex
    RetrievalIndex.query(tweet: str, intent: str | None, top_k: int) -> list[RetrievalResult]

A "resolved thread" is a dict with at minimum:
    {
        "customer_tweet": str,
        "brand_reply":    str,
        "intent":         str,   # one of the 11 taxonomy labels
        "thread_id":      str,   # any unique identifier
    }

The index is persisted as two files:
    <index_path>.bin  — FAISS binary index (L2-normalised vectors → cosine sim)
    <meta_path>.jsonl — one JSON object per line, same order as index rows
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Constants ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
EMBED_BATCH_SIZE = 64
DEFAULT_TOP_K = 5

# ── Data classes ──────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    thread_id: str
    customer_tweet: str
    brand_reply: str
    intent: str
    similarity: float   # cosine similarity in [0, 1]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Model singleton ───────────────────────────────────────────────────────
_MODEL: Optional[SentenceTransformer] = None


def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print(f"  [retrieval] Loading embedding model '{EMBEDDING_MODEL}'...")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of strings, return L2-normalised float32 array."""
    vecs = _model().encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # for cosine sim via inner product
    ).astype(np.float32)
    return vecs


# ── Index construction ─────────────────────────────────────────────────────
def build_index(
    threads: List[dict],
    index_path: str,
    meta_path: str,
    *,
    field_to_embed: str = "customer_tweet",
) -> None:
    """
    Build a FAISS index from a list of resolved thread dicts.

    Parameters
    ----------
    threads        : list of dicts with keys customer_tweet, brand_reply,
                     intent, thread_id
    index_path     : path to write the .bin FAISS index
    meta_path      : path to write the .jsonl metadata file
    field_to_embed : which field to index ("customer_tweet" by default)
    """
    if not threads:
        raise ValueError("Cannot build an index from an empty thread list.")

    texts = [t[field_to_embed] for t in threads]
    print(f"  [retrieval] Embedding {len(texts)} threads...")
    vecs = _embed(texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on L2-normalised = cosine
    index.add(vecs)
    print(f"  [retrieval] FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Persist
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)

    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"  [retrieval] Saved index → {index_path}")
    print(f"  [retrieval] Saved meta  → {meta_path}")


# ── Index loader ─────────────────────────────────────────────────────────
class RetrievalIndex:
    """
    Wraps a loaded FAISS index + metadata for querying.

    Usage
    -----
        idx = RetrievalIndex.load("data/faiss_index.bin", "data/faiss_meta.jsonl")
        results = idx.query("my package never arrived", intent="delivery_issue", top_k=3)
    """

    def __init__(self, index: faiss.Index, meta: List[dict]) -> None:
        self._index = index
        self._meta = meta

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "RetrievalIndex":
        """Load a previously built index from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        index = faiss.read_index(index_path)
        meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    meta.append(json.loads(line))

        print(f"  [retrieval] Loaded index: {index.ntotal} vectors from {index_path}")
        return cls(index, meta)

    def query(
        self,
        tweet: str,
        intent: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-k most similar resolved threads.

        Parameters
        ----------
        tweet  : incoming customer tweet text
        intent : if provided, pre-filter results to this intent label
        top_k  : number of results to return (after filtering)

        Returns
        -------
        List[RetrievalResult] sorted by similarity descending.
        """
        # Embed query
        q_vec = _embed([tweet])   # shape (1, dim)

        # Search — oversample to allow intent filtering
        search_k = top_k * 10 if intent else top_k
        search_k = min(search_k, self._index.ntotal)

        scores, indices = self._index.search(q_vec, search_k)
        scores = scores[0]    # flatten to 1-D
        indices = indices[0]

        results: List[RetrievalResult] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue   # FAISS returns -1 for empty slots
            m = self._meta[idx]
            # Intent filter
            if intent and m.get("intent") != intent:
                continue
            results.append(
                RetrievalResult(
                    thread_id=str(m.get("thread_id", str(idx))),
                    customer_tweet=m.get("customer_tweet", ""),
                    brand_reply=m.get("brand_reply", ""),
                    intent=m.get("intent", ""),
                    similarity=float(score),
                )
            )
            if len(results) >= top_k:
                break

        return results

    @property
    def size(self) -> int:
        return self._index.ntotal

    def intents_present(self) -> dict[str, int]:
        """Return count of each intent in the index."""
        counts: dict[str, int] = {}
        for m in self._meta:
            intent = m.get("intent", "unknown")
            counts[intent] = counts.get(intent, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ── Convenience loader alias ──────────────────────────────────────────────
def load_index(index_path: str, meta_path: str) -> RetrievalIndex:
    return RetrievalIndex.load(index_path, meta_path)


# ── CLI smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
    META_PATH = os.path.join(DATA_DIR, "faiss_meta.jsonl")

    if not os.path.exists(INDEX_PATH):
        print("Index not built yet. Run scripts/04_build_index.py first.")
        sys.exit(1)

    idx = RetrievalIndex.load(INDEX_PATH, META_PATH)
    print(f"\nIndex size: {idx.size} threads")
    print(f"Intent distribution: {json.dumps(idx.intents_present(), indent=2)}")

    test_queries = [
        ("@AmazonHelp where is my order? It's been a week.", "order_status_inquiry"),
        ("@AmazonHelp my Kindle won't charge.", "device_and_digital_support"),
        ("@AmazonHelp I want a refund, this is ridiculous.", "refund_request"),
    ]

    for tweet, intent in test_queries:
        print(f"\nQuery: {tweet[:70]}")
        print(f"Intent filter: {intent}")
        results = idx.query(tweet, intent=intent, top_k=3)
        for r in results:
            print(f"  [{r.similarity:.3f}] {r.customer_tweet[:80]}")
            print(f"           Reply: {r.brand_reply[:80]}")
