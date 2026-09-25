"""
rag.py — Config-aware, traced wrapper around the Module 1 pipeline.

src/ is left untouched. This module reuses src.retrieve.HybridRetriever (same BM25,
same RRF maths) and src.generate.generate (same prompt, same model, temperature=0),
and adds three things:

  1. Config selection (v1 / v2 / v2-vector — see ops/config.py).
  2. Retrieval scores + timings on every result, so Phoenix and the eval runner can
     see *why* a chunk was chosen, not just that it was.
  3. LangSmith tracing via @traceable. The generation step is a LangChain LCEL chain,
     so LangSmith traces it automatically; retrieval is plain Python (Pinecone + BM25),
     so it is wrapped explicitly as a run_type="retriever" span.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from langsmith import traceable

from ops.config import PipelineConfig, get_config
from src.retrieve import RRF_K, HybridRetriever, _tokenize

load_dotenv()


class ConfigurableRetriever(HybridRetriever):
    """HybridRetriever that reads its chunks file / namespace / mode from a PipelineConfig.

    Heavy clients (embeddings, Pinecone) are only created in `from_config`, so tests can
    build one with fakes via `__new__` + `_setup`.
    """

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "ConfigurableRetriever":
        from langchain_openai import OpenAIEmbeddings
        from pinecone import Pinecone

        obj = cls.__new__(cls)
        embeddings = OpenAIEmbeddings(
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(config.index_name)
        with open(config.chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        obj._setup(config, chunks, embeddings, index)
        return obj

    def _setup(self, config: PipelineConfig, chunks: list[dict], embeddings, index) -> None:
        from rank_bm25 import BM25Okapi

        self.config = config
        self.default_k = config.k
        self.chunks = chunks
        self.chunk_by_id = {c["id"]: c for c in chunks}
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
        self.embeddings = embeddings
        self.index = index

    # --- search primitives (same as Module 1, plus namespace + scores) -----------------

    def _vector_search_scored(self, query: str, top_n: int) -> list[tuple[str, float]]:
        vec = self.embeddings.embed_query(query)
        kwargs = {"vector": vec, "top_k": top_n, "include_metadata": False}
        if self.config.namespace:
            kwargs["namespace"] = self.config.namespace
        result = self.index.query(**kwargs)
        return [(m["id"], float(m["score"])) for m in result["matches"]]

    def _vector_search(self, query: str, top_n: int) -> list[str]:
        return [cid for cid, _ in self._vector_search_scored(query, top_n)]

    # --- retrieval ------------------------------------------------------------------------

    def retrieve_scored(self, query: str, k: int | None = None) -> list[dict]:
        """Returns chunk dicts with extra keys: score, vector_rank, bm25_rank."""
        k = k or self.default_k
        candidate_pool = max(k * 4, 20)  # identical to Module 1

        vector_hits = self._vector_search_scored(query, candidate_pool)
        vector_rank = {cid: r for r, (cid, _) in enumerate(vector_hits)}

        if self.config.retrieval_mode == "vector":
            ranked = [(cid, score) for cid, score in vector_hits[:k]]
            bm25_rank: dict[str, int] = {}
        else:
            bm25_ids = self._bm25_search(query, candidate_pool)
            bm25_rank = {cid: r for r, cid in enumerate(bm25_ids)}
            # Reciprocal Rank Fusion — same maths and ordering as src/retrieve.py
            scores: dict[str, float] = {}
            for rank, (cid, _) in enumerate(vector_hits):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            for rank, cid in enumerate(bm25_ids):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
            ranked = [(cid, scores[cid]) for cid in top_ids]

        out = []
        for cid, score in ranked:
            if cid not in self.chunk_by_id:
                continue
            c = dict(self.chunk_by_id[cid])
            c["score"] = score
            c["vector_rank"] = vector_rank.get(cid)
            c["bm25_rank"] = bm25_rank.get(cid)
            out.append(c)
        return out

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        return self.retrieve_scored(query, k)


@lru_cache(maxsize=8)
def get_retriever(config_name: str) -> ConfigurableRetriever:
    return ConfigurableRetriever.from_config(get_config(config_name))


# --- traced pipeline -----------------------------------------------------------------------


@traceable(run_type="retriever", name="retrieve")
def _traced_retrieve(query: str, config_name: str) -> list[dict]:
    chunks = get_retriever(config_name).retrieve_scored(query)
    # LangSmith renders retriever runs nicely when outputs look like Documents.
    return [
        {
            "page_content": c["text"],
            "type": "Document",
            "metadata": {k: v for k, v in c.items() if k != "text"},
        }
        for c in chunks
    ]


def retrieve(query: str, config_name: str = "v1") -> list[dict]:
    docs = _traced_retrieve(query, config_name)
    return [{"text": d["page_content"], **d["metadata"]} for d in docs]


@traceable(run_type="chain", name="rag_pipeline")
def answer_query(query: str, config_name: str = "v1") -> dict:
    """End-to-end: retrieve -> generate. Returns a plain dict (JSON-safe for LangSmith)."""
    from src.generate import generate

    t0 = time.perf_counter()
    chunks = retrieve(query, config_name)
    t1 = time.perf_counter()
    result = generate(query, [{"doc_id": c["doc_id"], "text": c["text"]} for c in chunks])
    t2 = time.perf_counter()

    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "contexts": [c["text"] for c in chunks],
        "retrieved_chunk_ids": [c["id"] for c in chunks],
        "retrieved_doc_ids": [c["doc_id"] for c in chunks],
        "retrieval_latency_ms": round((t1 - t0) * 1000, 1),
        "generation_latency_ms": round((t2 - t1) * 1000, 1),
        "config": config_name,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    p.add_argument("--config", default="v1")
    a = p.parse_args()
    out = answer_query(" ".join(a.question), a.config)
    out.pop("contexts")
    print(json.dumps(out, indent=2))
