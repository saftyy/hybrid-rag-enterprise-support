"""
retrieve.py — Hybrid retriever: Pinecone vector search + BM25 keyword search,
combined via Reciprocal Rank Fusion (RRF).

Why hybrid: pure vector (semantic) search can miss queries that hinge on an
exact term — an error code, a specific API field name, an exact phrase from
a resolved ticket. BM25 (keyword/lexical scoring) catches those. Vector
search catches paraphrased/semantic matches BM25 would miss. RRF combines
both rankings by position, so no blend weight needs to be hand-tuned.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "evals" / "results" / "chunks.json"

# Standard RRF smoothing constant (from the original RRF paper).
RRF_K = 60


def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run `python -m src.ingest --corpus <path>` first."
        )
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self, default_k: int = 5):
        self.default_k = default_k
        self.chunks = _load_chunks()
        self.chunk_by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])
        self.embeddings = OpenAIEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "helix-checkpoint-rag"))

    def _vector_search(self, query: str, top_n: int) -> list[str]:
        vec = self.embeddings.embed_query(query)
        result = self.index.query(vector=vec, top_k=top_n, include_metadata=False)
        return [match["id"] for match in result["matches"]]

    def _bm25_search(self, query: str, top_n: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [self.chunks[i]["id"] for i in ranked_idx]

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.default_k
        # Pull a wider candidate pool from each method before fusing, so RRF
        # has enough signal to work with.
        candidate_pool = max(k * 4, 20)

        vector_ids = self._vector_search(query, candidate_pool)
        bm25_ids = self._bm25_search(query, candidate_pool)

        scores: dict[str, float] = {}
        for rank, cid in enumerate(vector_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self.chunk_by_id[cid] for cid in top_ids if cid in self.chunk_by_id]


@lru_cache(maxsize=1)
def get_retriever(k: int = 5) -> HybridRetriever:
    """Singleton — BM25 index + chunk list are loaded once, reused across calls."""
    return HybridRetriever(default_k=k)


def retrieve(query: str, k: int = 5) -> list[dict]:
    return get_retriever().retrieve(query, k=k)