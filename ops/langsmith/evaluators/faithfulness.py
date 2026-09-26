"""
faithfulness.py — Is every factual claim in the answer supported by the retrieved context?

Method (claim decomposition, same idea as RAGAS faithfulness):
    1. Judge splits the answer into atomic factual claims.
    2. Judge labels each claim SUPPORTED / UNSUPPORTED against the retrieved context only.
    3. Python computes score = supported / total.

What it catches: hallucination — the model adding facts that are not in the context,
even if they happen to be true.

Explicit edge case: an answer with no factual claims (e.g. "The provided context doesn't
cover this") scores 1.0 — it asserted nothing unsupported. That is correct for this metric
but means faithfulness alone can be gamed by abstaining. That is why it is paired with
answer_relevance and csm_resolution_quality, which both punish unnecessary abstention,
and why run_evals also reports an abstention rate.

Enforcement (added after the first v1/v2/v3 runs): the judge did not apply the rule above
consistently. It scored the identical answer "The provided context doesn't cover this." as
1.0 for Q037 but as a single *unsupported claim* (0.0) for Q017, Q033 and Q044 — costing v1
~0.06 faithfulness for abstaining, which is exactly what this metric is not meant to punish.
Claims that are themselves abstention statements are now removed in code before scoring, so
the documented definition is applied deterministically instead of depending on the judge.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ops.langsmith.evaluators._judge import JudgeFn, format_contexts, is_abstention, openai_judge

SYSTEM = """You are a meticulous fact-checker for a retrieval-augmented support assistant.
You are given CONTEXT (the only documents the assistant was allowed to use) and an ANSWER.

Step 1 — List every atomic factual claim made in the ANSWER. A claim is a single checkable
statement (a number, a limit, a price, a setting location, a procedure step, a policy).
Do NOT list: statements that information is missing or unknown, greetings, or suggestions
to contact someone that make no factual assertion.

Step 2 — For each claim decide `supported`:
  true  = the CONTEXT explicitly states it, or it follows directly and unambiguously.
  false = the CONTEXT does not state it, contradicts it, or it relies on outside knowledge
          (even if it might be true in the real world).
Be strict: a claim with a wrong number, plan name or setting path is unsupported."""

USER = """CONTEXT:
{context}

ANSWER:
{answer}"""


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    reason: str = Field(description="One short sentence citing where in the context, or why not.")


class FaithfulnessVerdict(BaseModel):
    claims: list[ClaimCheck]


def score_faithfulness(answer: str, contexts: list[str], judge: JudgeFn = openai_judge) -> dict:
    verdict = judge(
        FaithfulnessVerdict, SYSTEM, USER.format(context=format_contexts(contexts), answer=answer)
    )
    dropped = [c for c in verdict.claims if is_abstention(c.claim)]
    claims = [c for c in verdict.claims if not is_abstention(c.claim)]
    if not claims:
        note = f" ({len(dropped)} abstention statement(s) ignored)" if dropped else ""
        return {"key": "faithfulness", "score": 1.0,
                "comment": "No factual claims (abstention or non-answer)." + note}
    supported = sum(c.supported for c in claims)
    unsupported = [c.claim for c in claims if not c.supported]
    comment = f"{supported}/{len(claims)} claims supported."
    if unsupported:
        comment += " Unsupported: " + " | ".join(unsupported[:5])
    return {"key": "faithfulness", "score": round(supported / len(claims), 4), "comment": comment}


def faithfulness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LangSmith evaluator entry point."""
    return score_faithfulness(outputs.get("answer", ""), outputs.get("contexts", []))
