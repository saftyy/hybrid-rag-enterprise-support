"""Eval harness executes end-to-end (on a small subset, to keep the test fast and cheap)."""

import pytest

from src.retrieve import CHUNKS_PATH
from evals.harness import run_eval, TEST_SET_PATH


@pytest.mark.skipif(
    not CHUNKS_PATH.exists(), reason="No ingested chunks found; run ingestion first"
)
@pytest.mark.skipif(not TEST_SET_PATH.exists(), reason="evals/test_set.json not found")
def test_eval_harness_executes_end_to_end(tmp_path):
    # Write to a temp directory so this test never overwrites your real,
    # full 50-query results in evals/results/.
    output = run_eval(limit=2, output_dir=tmp_path)

    assert "overall" in output
    assert "faithfulness" in output["overall"]
    assert "context_precision" in output["overall"]
    assert len(output["per_query"]) == 2
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "summary.md").exists()