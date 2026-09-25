"""
answer_relevance.py — Does the answer address the question that was actually asked?

Deliberately *not* a correctness check (that is the custom judge's job) and not a
grounding check (faithfulness). This isolates one failure mode: the answer is accurate
and grounded but answers a different or broader question, rambles, or dodges.

Method: 1–5 rubric with an explicit written anchor for each level; reasoning is produced
*before* the score (so the score is conditioned on the reasoning); normalised to 0–1 as
(score - 1) / 4.

Every question in the Helix test set is answerable from the corpus, so an abstention
("the context doesn't cover this") is scored 1 — it does not address the question.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ops.langsmith.evaluators._judge import JudgeFn, openai_judge

SYSTEM = """You grade whether an ANSWER addresses the QUESTION a Customer Success Manager asked.
Judge ONLY relevance and directness — not factual accuracy.

5 = Directly and completely addresses the specific question; nothing important missing.
4 = Addresses the question; minor omission or small unnecessary digression.
3 = Partially addresses it; a key part of the question is left unanswered, or the answer
    is generic where the question was specific.
2 = Mostly off-target: answers a related but different question.
1 = Does not address the question (refusal, "the context doesn't cover this", or unrelated).

Write your reasoning first, then the score."""

USER = """QUESTION:
{question}

ANSWER:
{answer}"""


class RelevanceVerdict(BaseModel):
    reasoning: str = Field(description="2-3 sentences.")
    score: Literal[1, 2, 3, 4, 5]


def score_answer_relevance(question: str, answer: str, judge: JudgeFn = openai_judge) -> dict:
    v = judge(RelevanceVerdict, SYSTEM, USER.format(question=question, answer=answer))
    return {"key": "answer_relevance", "score": (v.score - 1) / 4,
            "comment": f"{v.score}/5 — {v.reasoning}"}


def answer_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LangSmith evaluator entry point."""
    return score_answer_relevance(inputs["question"], outputs.get("answer", ""))
