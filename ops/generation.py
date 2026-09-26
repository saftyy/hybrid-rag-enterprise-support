"""
generation.py — Prompt variants, selected by PipelineConfig.prompt.

"module1"  -> src.generate.generate, byte-for-byte the Module 1 prompt and chain.
"grounded" -> same model, temperature, schema and context format; the system prompt adds
              three rules targeting the exact unsupported claims the faithfulness judge
              flagged in the v2 eval:

   Q026  invented a 3-step cancellation procedure ("go to account settings, navigate to
         billing...") that no document contains                       -> rule 5
   Q045  added commentary ("It's important to understand how this change might affect
         their engagement")                                           -> rule 6
   Q043  extrapolated beyond the docs about plan retention            -> rule 7

Rule 7 also says to answer the covered part of a partially-covered question. Without it,
a stricter prompt tends to "win" faithfulness by abstaining more — which is why every
eval of v3 must be read together with abstention rate, answer relevance and CSM quality.
"""

from __future__ import annotations

import os
from functools import lru_cache

from src.generate import SYSTEM_PROMPT, RAGResponse, _format_context
from src.generate import generate as module1_generate

GROUNDING_RULES = """
Additional grounding rules:
5. Every step, setting name, menu path, number, plan name and limit you state must appear \
in the context. Never add generic procedure steps (e.g. "go to account settings") that are \
not written in the context, even if they seem likely.
6. Do not add commentary, recommendations or reassurance that the context does not support \
(e.g. "it's important to consider..."). State the facts and stop.
7. If the context answers only part of the question, answer that part and name the specific \
part that is not covered. Do not infer or extrapolate beyond what the context states.
"""

GROUNDED_SYSTEM_PROMPT = SYSTEM_PROMPT.replace("\nContext:\n{context}\n", GROUNDING_RULES + "\nContext:\n{context}\n")
assert GROUNDED_SYSTEM_PROMPT != SYSTEM_PROMPT, "grounding rules were not inserted"


@lru_cache(maxsize=1)
def _grounded_chain():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=os.environ.get("GENERATION_MODEL", "gpt-4o-mini"), temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", GROUNDED_SYSTEM_PROMPT), ("human", "{question}")])
    return prompt | model.with_structured_output(RAGResponse)


def generate_for(prompt_variant: str, query: str, context_chunks: list[dict]) -> RAGResponse:
    if prompt_variant == "module1":
        return module1_generate(query, context_chunks)
    if prompt_variant == "grounded":
        return _grounded_chain().invoke(
            {"context": _format_context(context_chunks), "question": query})
    raise ValueError(f"unknown prompt variant {prompt_variant!r}")
