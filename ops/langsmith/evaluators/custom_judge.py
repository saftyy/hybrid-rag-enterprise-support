"""
custom_judge.py — CSM Resolution Quality: "Could a senior CSM safely act on this answer
in front of a customer?"

Why a custom judge: faithfulness and relevance are generic. Neither notices that an answer
is grounded and on-topic but *wrong about the one number that matters* (the test set's
`notes` field, e.g. "Answer must include the number 100"), or that it promises a customer
something (a refund, an SLA credit, a timeline) no document authorises. For 40 senior CSMs
talking to paying customers, those are the failures that cost money and trust.

The judge makes three categorical calls; Python turns them into a score:

    key_facts        all | partial | none | contradicted   (vs. reference answer + notes)
    actionability    high | medium | low | not_applicable  (can the CSM act on it now?)
    unauthorised_commitment  bool  (promises/commitments not in the reference or policy)

    facts_score  = {all: 1.0, partial: 0.5, none: 0.0, contradicted: 0.0}
    action_score = {high: 1.0, medium: 0.6, low: 0.2}
    score = facts_score                                   if actionability == not_applicable
          = 0.7 * facts_score + 0.3 * action_score        otherwise
    if unauthorised_commitment: score = min(score, 0.25)  # hard cap, a safety failure

Why 0.7 / 0.3: a well-written answer with the wrong fact is worse than a terse answer with
the right one. Why a hard cap: one invented refund promise outweighs any amount of polish.
The weights are hypotheses — they are checked against human labels by calibrate_judge.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ops.langsmith.evaluators._judge import JudgeFn, openai_judge

FACTS_SCORE = {"all": 1.0, "partial": 0.5, "none": 0.0, "contradicted": 0.0}
ACTION_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.2}
FACTS_WEIGHT, ACTION_WEIGHT = 0.7, 0.3
COMMITMENT_CAP = 0.25

SYSTEM = """You are a senior Customer Success lead at Helix, a B2B SaaS workflow platform.
An internal AI assistant answered a question from one of your CSMs. Decide whether the CSM
could safely act on the answer in front of a customer.

You are given the QUESTION, the assistant's ANSWER, a REFERENCE answer written by an expert,
and GRADING NOTES stating which facts are mandatory.

Make three judgements:

1. key_facts — compare ANSWER to REFERENCE and GRADING NOTES:
   all          = every mandatory fact is present and correct (wording may differ)
   partial      = some mandatory facts present and correct, some missing
   none         = mandatory facts missing (including "the context doesn't cover this")
   contradicted = the answer states something that conflicts with the reference
                  (wrong number, wrong plan, wrong setting, wrong procedure)

2. actionability — could the CSM act or respond to the customer right now?
   high           = clear next steps / exact settings path / who to escalate to, as needed
   medium         = right direction but the CSM would need to look something up
   low            = vague or generic ("check the docs", "contact support")
   not_applicable = a pure factual lookup (e.g. a price or a limit) where stating the
                    fact IS the complete action

3. unauthorised_commitment — true if the answer promises or implies a commitment to the
   customer (refund, credit, discount, exception, deadline, roadmap feature) that is NOT
   supported by the REFERENCE. Otherwise false.

Give short reasoning first, then the judgements. Extra correct detail beyond the reference
is fine; do not penalise it."""

USER = """QUESTION:
{question}

ANSWER:
{answer}

REFERENCE:
{reference}

GRADING NOTES:
{notes}"""


class CSMVerdict(BaseModel):
    reasoning: str = Field(description="2-4 sentences.")
    key_facts: Literal["all", "partial", "none", "contradicted"]
    actionability: Literal["high", "medium", "low", "not_applicable"]
    unauthorised_commitment: bool


def combine(v: CSMVerdict) -> float:
    facts = FACTS_SCORE[v.key_facts]
    if v.actionability == "not_applicable":
        score = facts
    else:
        score = FACTS_WEIGHT * facts + ACTION_WEIGHT * ACTION_SCORE[v.actionability]
    if v.unauthorised_commitment:
        score = min(score, COMMITMENT_CAP)
    return round(score, 4)


def score_csm_resolution_quality(
    question: str, answer: str, reference: str, notes: str, judge: JudgeFn = openai_judge
) -> dict:
    v = judge(CSMVerdict, SYSTEM, USER.format(
        question=question, answer=answer, reference=reference, notes=notes or "(none)"))
    comment = (f"facts={v.key_facts}, actionability={v.actionability}, "
               f"commitment={v.unauthorised_commitment} — {v.reasoning}")
    return {"key": "csm_resolution_quality", "score": combine(v), "comment": comment}


def csm_resolution_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LangSmith evaluator entry point."""
    return score_csm_resolution_quality(
        inputs["question"],
        outputs.get("answer", ""),
        reference_outputs.get("reference_answer", ""),
        reference_outputs.get("notes", ""),
    )
