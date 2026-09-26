"""
Evaluator registry.

Three LLM-as-judge evaluators (the checkpoint requirement):
    faithfulness            — grounded in retrieved context?          (standard)
    answer_relevance        — answers the question asked?             (standard)
    csm_resolution_quality  — could a CSM safely act on it?           (custom)

Plus cheap deterministic diagnostics (no LLM call, no cost). They exist so that when a
judge score drops you can immediately tell *which stage* broke:
    source_hit        — was at least one expected source document retrieved?  (retrieval)
    source_recall     — fraction of expected source documents retrieved       (retrieval)
    abstained         — did the assistant say it couldn't answer?              (generation)
    citation_valid    — are all cited sources among the retrieved documents?   (generation)
"""

from __future__ import annotations

from ops.langsmith.evaluators._judge import ABSTAIN_RE, is_abstention  # noqa: F401
from ops.langsmith.evaluators.answer_relevance import answer_relevance
from ops.langsmith.evaluators.custom_judge import csm_resolution_quality
from ops.langsmith.evaluators.faithfulness import faithfulness

def source_hit(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected = set(reference_outputs.get("expected_sources", []))
    got = set(outputs.get("retrieved_doc_ids", []))
    return {"key": "source_hit", "score": float(bool(expected & got))}


def source_recall(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected = set(reference_outputs.get("expected_sources", []))
    got = set(outputs.get("retrieved_doc_ids", []))
    score = round(len(expected & got) / len(expected), 4) if expected else 1.0
    return {"key": "source_recall", "score": score,
            "comment": f"missing: {sorted(expected - got)}" if expected - got else None}


def abstained(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "abstained", "score": float(is_abstention(outputs.get("answer", "")))}


def citation_valid(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    cited = set(outputs.get("sources", []))
    retrieved = set(outputs.get("retrieved_doc_ids", []))
    bad = cited - retrieved
    return {"key": "citation_valid", "score": float(not bad),
            "comment": f"cited but not retrieved: {sorted(bad)}" if bad else None}


LLM_EVALUATORS = [faithfulness, answer_relevance, csm_resolution_quality]
DETERMINISTIC_EVALUATORS = [source_hit, source_recall, abstained, citation_valid]
ALL_EVALUATORS = LLM_EVALUATORS + DETERMINISTIC_EVALUATORS
