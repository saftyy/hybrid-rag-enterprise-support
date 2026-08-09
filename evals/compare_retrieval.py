"""
compare_retrieval.py — One-off comparison: vector-only search vs hybrid (RRF)
search, measured against test_set.json's expected_sources. Generates the
before/after numbers cited in the README's Retrieval Strategy section.

Run:
    python -m evals.compare_retrieval
"""

import json
from pathlib import Path

from src.retrieve import get_retriever

TEST_SET_PATH = Path(__file__).parent / "test_set.json"


def doc_ids_for(retriever, chunk_ids: list[str]) -> list[str]:
    return [retriever.chunk_by_id[i]["doc_id"] for i in chunk_ids if i in retriever.chunk_by_id]


def main():
    retriever = get_retriever()
    data = json.load(open(TEST_SET_PATH, encoding="utf-8"))["queries"]

    vector_hits = 0
    hybrid_hits = 0
    vector_precision_sum = 0.0
    hybrid_precision_sum = 0.0
    k = 5

    for q in data:
        expected = set(q["expected_sources"])

        vector_ids = retriever._vector_search(q["query"], k)
        vector_docs = set(doc_ids_for(retriever, vector_ids))

        hybrid_chunks = retriever.retrieve(q["query"], k=k)
        hybrid_docs = {c["doc_id"] for c in hybrid_chunks}

        if expected & vector_docs:
            vector_hits += 1
        if expected & hybrid_docs:
            hybrid_hits += 1

        vector_precision_sum += len(vector_docs & expected) / k
        hybrid_precision_sum += len(hybrid_docs & expected) / k

    n = len(data)
    print(f"n = {n} queries, k = {k}\n")
    print(
        f"Vector-only  — Hit@{k}: {vector_hits}/{n} ({vector_hits/n:.1%})  "
        f"|  Doc-Precision@{k}: {vector_precision_sum/n:.3f}"
    )
    print(
        f"Hybrid (RRF) — Hit@{k}: {hybrid_hits}/{n} ({hybrid_hits/n:.1%})  "
        f"|  Doc-Precision@{k}: {hybrid_precision_sum/n:.3f}"
    )


if __name__ == "__main__":
    main()