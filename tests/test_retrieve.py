"""Retrieval returns >= 1 result for every query in the eval test set."""

import json
from pathlib import Path

import pytest

from src.retrieve import retrieve, CHUNKS_PATH

TEST_SET_PATH = Path(__file__).resolve().parent.parent / "evals" / "test_set.json"


def _load_queries() -> list[str]:
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [item["query"] for item in data["queries"]]


@pytest.mark.skipif(
    not CHUNKS_PATH.exists(),
    reason="No ingested chunks found; run `python -m src.ingest` first",
)
@pytest.mark.skipif(
    not TEST_SET_PATH.exists(),
    reason="evals/test_set.json not found",
)
def test_retrieve_returns_results_for_every_query():
    queries = _load_queries()
    assert len(queries) > 0

    failures = []
    for q in queries:
        results = retrieve(q, k=5)
        if len(results) < 1:
            failures.append(q)

    assert not failures, f"No results returned for {len(failures)} queries: {failures[:5]}"
