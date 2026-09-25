"""
setup.py — LangSmith project, tracer and versioned eval dataset.

Run once (safe to re-run):
    python -m ops.langsmith.setup

Versioning rule: the dataset name carries the test-set version
(`helix-csm-eval-v1.0`) and its description stores a SHA-256 of the test set content.
If someone edits test_set.json without bumping its "version" field, this script refuses
to silently reuse the old dataset — otherwise two eval runs with the same dataset name
could be scored against different questions, and the comparison would be meaningless.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SET_PATH = REPO_ROOT / "evals" / "test_set.json"
DEFAULT_PROJECT = "helix-csm-assistant"


def configure_tracing(project: str | None = None) -> str:
    """Turn LangSmith tracing on for this process. Accepts LANGSMITH_* or legacy LANGCHAIN_*."""
    load_dotenv()
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        raise SystemExit("Set LANGSMITH_API_KEY in .env (see .env.example).")
    project = project or os.environ.get("LANGSMITH_PROJECT") \
        or os.environ.get("LANGCHAIN_PROJECT") or DEFAULT_PROJECT
    # Set both spellings: langsmith 0.1.x and langchain-core 0.3.x read either.
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGCHAIN_PROJECT"] = project
    return project


def get_client():
    from langsmith import Client
    return Client()


def load_test_set(path: Path = TEST_SET_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_set_hash(data: dict) -> str:
    canonical = json.dumps(data["queries"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def dataset_name_for(data: dict) -> str:
    return f"helix-csm-eval-v{data.get('version', '0')}"


def to_examples(data: dict) -> list[dict]:
    """test_set.json queries -> LangSmith example dicts."""
    return [
        {
            "inputs": {"question": q["query"], "qid": q["id"]},
            "outputs": {
                "reference_answer": q["ground_truth_answer"],
                "expected_sources": q["expected_sources"],
                "notes": q.get("notes", ""),
            },
            "metadata": {"qid": q["id"], "difficulty": q["difficulty"], "category": q["category"]},
        }
        for q in data["queries"]
    ]


def ensure_dataset(client, data: dict | None = None):
    data = data or load_test_set()
    name = dataset_name_for(data)
    digest = test_set_hash(data)
    description = f"Helix CSM eval set ({len(data['queries'])} queries). sha256:{digest}"

    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
        if f"sha256:{digest}" not in (ds.description or ""):
            raise SystemExit(
                f"Dataset '{name}' exists but test_set.json content changed "
                f"(hash {digest}). Bump the 'version' field in test_set.json."
            )
        return ds

    ds = client.create_dataset(dataset_name=name, description=description)
    ex = to_examples(data)
    client.create_examples(
        inputs=[e["inputs"] for e in ex],
        outputs=[e["outputs"] for e in ex],
        metadata=[e["metadata"] for e in ex],
        dataset_id=ds.id,
    )
    print(f"Created dataset '{name}' with {len(ex)} examples.")
    return ds


def main():
    project = configure_tracing()
    client = get_client()
    ds = ensure_dataset(client)
    print(f"LangSmith project : {project}")
    try:
        print(f"Project URL       : {client.read_project(project_name=project).url}")
    except Exception:
        print("Project URL       : (created on first traced run — re-run this after run_evals)")
    print(f"Dataset           : {ds.name}")
    print(f"Dataset URL       : {getattr(ds, 'url', '(see LangSmith > Datasets)')}")


if __name__ == "__main__":
    main()
