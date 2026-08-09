"""
harness.py — Runs the eval test set, reports faithfulness, answer relevancy,
and context precision via RAGAs. Writes machine-readable JSON and a
human-readable markdown summary.

Run:
    python -m evals.harness
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.retrieve import retrieve
from src.generate import generate

load_dotenv()

TEST_SET_PATH = Path(__file__).parent / "test_set.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Required pass thresholds (per checkpoint spec)
FAITHFULNESS_THRESHOLD = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.60


def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def run_pipeline_over_test_set(queries: list[dict]) -> list[dict]:
    """Run every query through retrieve -> generate, collecting what RAGAs needs."""
    records = []
    for q in tqdm(queries, desc="Running pipeline over test set"):
        try:
            chunks = retrieve(q["query"])
            result = generate(q["query"], chunks)
            records.append(
                {
                    "id": q["id"],
                    "user_input": q["query"],
                    "response": result.answer,
                    "retrieved_contexts": [c["text"] for c in chunks] or [""],
                    "reference": q["ground_truth_answer"],
                    "expected_sources": q["expected_sources"],
                    "difficulty": q["difficulty"],
                    "category": q["category"],
                    "sources_cited": result.sources,
                    "confidence": result.confidence,
                }
            )
        except Exception as e:
            records.append(
                {
                    "id": q["id"],
                    "user_input": q["query"],
                    "response": f"[PIPELINE ERROR: {e}]",
                    "retrieved_contexts": [""],
                    "reference": q["ground_truth_answer"],
                    "expected_sources": q["expected_sources"],
                    "difficulty": q["difficulty"],
                    "category": q["category"],
                    "sources_cited": [],
                    "confidence": "low",
                }
            )
    return records


def score_with_ragas(records: list[dict]):
    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(model=os.environ.get("GENERATION_MODEL", "gpt-4o-mini"), temperature=0)
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    )

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r["reference"],
            }
            for r in records
        ]
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    return result.to_pandas()


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r[key] == r[key]]  # filters out NaN
    return sum(vals) / len(vals) if vals else 0.0


def run_eval(limit: int | None = None, output_dir: Path | None = None) -> dict:
    results_dir = output_dir or RESULTS_DIR
    queries = load_test_set()
    if limit:
        queries = queries[:limit]
    print(f"Loaded {len(queries)} test queries.")

    records = run_pipeline_over_test_set(queries)

    print("Scoring with RAGAs...")
    scored_df = score_with_ragas(records)

    per_query_results = []
    for i, r in enumerate(records):
        row = scored_df.iloc[i]
        per_query_results.append(
            {
                "id": r["id"],
                "query": r["user_input"],
                "difficulty": r["difficulty"],
                "category": r["category"],
                "answer": r["response"],
                "sources_cited": r["sources_cited"],
                "expected_sources": r["expected_sources"],
                "confidence": r["confidence"],
                "faithfulness": float(row.get("faithfulness", float("nan"))),
                "answer_relevancy": float(row.get("answer_relevancy", float("nan"))),
                "context_precision": float(row.get("context_precision", float("nan"))),
            }
        )

    overall = {
        "faithfulness": _mean(per_query_results, "faithfulness"),
        "answer_relevancy": _mean(per_query_results, "answer_relevancy"),
        "context_precision": _mean(per_query_results, "context_precision"),
    }
    passed = {
        "faithfulness_pass": overall["faithfulness"] >= FAITHFULNESS_THRESHOLD,
        "context_precision_pass": overall["context_precision"] >= CONTEXT_PRECISION_THRESHOLD,
    }

    output = {
        "overall": overall,
        "thresholds": {
            "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
            "context_precision_threshold": CONTEXT_PRECISION_THRESHOLD,
        },
        "passed": passed,
        "per_query": per_query_results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = results_dir / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    _write_markdown_summary(output, results_dir / "summary.md")

    print(
        f"\nOverall: faithfulness={overall['faithfulness']:.3f} "
        f"({'PASS' if passed['faithfulness_pass'] else 'FAIL'}), "
        f"answer_relevancy={overall['answer_relevancy']:.3f}, "
        f"context_precision={overall['context_precision']:.3f} "
        f"({'PASS' if passed['context_precision_pass'] else 'FAIL'})"
    )
    print(f"Results: {results_json_path}")
    print(f"Summary: {results_dir / 'summary.md'}")

    return output


def _write_markdown_summary(output: dict, path: Path) -> None:
    overall = output["overall"]
    passed = output["passed"]
    lines = [
        "# Eval Results Summary",
        "",
        f"- **Faithfulness:** {overall['faithfulness']:.3f} "
        f"(threshold >= {output['thresholds']['faithfulness_threshold']}) — "
        f"{'PASS' if passed['faithfulness_pass'] else 'FAIL'}",
        f"- **Answer Relevancy:** {overall['answer_relevancy']:.3f}",
        f"- **Context Precision:** {overall['context_precision']:.3f} "
        f"(threshold >= {output['thresholds']['context_precision_threshold']}) — "
        f"{'PASS' if passed['context_precision_pass'] else 'FAIL'}",
        "",
        "## Per-query results",
        "",
        "| ID | Difficulty | Category | Faithfulness | Answer Relevancy | Context Precision | Confidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in output["per_query"]:
        lines.append(
            f"| {r['id']} | {r['difficulty']} | {r['category']} | "
            f"{r['faithfulness']:.2f} | {r['answer_relevancy']:.2f} | "
            f"{r['context_precision']:.2f} | {r['confidence']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_eval()