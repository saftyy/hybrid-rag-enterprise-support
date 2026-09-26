"""
_judge.py — Shared LLM-as-judge plumbing.

Design decisions (worth saying out loud in the Loom):
  * The judge model (default gpt-4o) is deliberately *stronger* than the generator
    (gpt-4o-mini). A model grading its own family at the same size tends to share its blind
    spots. Override with JUDGE_MODEL.
  * temperature=0 and structured output (Pydantic) — the judge returns sub-judgements, not
    free text, so scores are parseable and reruns are stable.
  * Where possible the LLM makes *categorical* calls and plain Python turns them into a
    number. LLMs are poor at picking "0.73"; they are decent at "was this claim supported?".
  * Every evaluator takes an injectable `judge` callable, so tests run offline with a fake.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Callable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# judge(schema, system_prompt, user_prompt) -> instance of schema
JudgeFn = Callable[[type[T], str, str], T]

DEFAULT_JUDGE_MODEL = "gpt-4o"


def judge_model_name() -> str:
    return os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


@lru_cache(maxsize=None)
def _structured_llm(schema: type[BaseModel]):
    from langchain_openai import ChatOpenAI

    # max_retries: the OpenAI client backs off exponentially on 429s. Low-tier accounts have a
    # 30k tokens/min cap on gpt-4o, which three judges x several parallel examples exceed.
    llm = ChatOpenAI(model=judge_model_name(), temperature=0, seed=42, max_retries=10)
    return llm.with_structured_output(schema)


def openai_judge(schema: type[T], system: str, user: str) -> T:
    return _structured_llm(schema).invoke([("system", system), ("human", user)])


def format_contexts(contexts: list[str], max_chars_each: int = 4000) -> str:
    if not contexts:
        return "(no context was retrieved)"
    return "\n\n".join(
        f"[Context {i + 1}]\n{c[:max_chars_each]}" for i, c in enumerate(contexts)
    )


# Phrases the generator uses to say the context lacks the answer. Shared by the faithfulness
# evaluator (to drop abstention "claims") and the deterministic `abstained` evaluator.
ABSTAIN_RE = re.compile(
    r"(doesn't|does not|do not|don't) (cover|contain|include|mention|provide|have)"
    r"|not enough information|no information|unable to (find|answer)|cannot answer",
    re.IGNORECASE,
)


def is_abstention(text: str) -> bool:
    return bool(ABSTAIN_RE.search(text or ""))
