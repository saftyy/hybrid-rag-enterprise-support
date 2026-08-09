"""
generate.py — LCEL chain: retrieved context -> structured response.

Structured output contract:
    {
        "answer": str,
        "sources": list[str],       # doc IDs actually used
        "confidence": "low" | "medium" | "high"
    }

Design notes (see README "Generation Prompt" section for full reasoning):
    - temperature=0: a support assistant should give deterministic, repeatable
      answers, not creative variation, for the same question + context.
    - The prompt explicitly instructs the model to admit when context is
      insufficient rather than fill gaps — this is what the faithfulness
      metric is actually measuring.
    - `sources` must reflect chunks the model actually relied on, not every
      chunk that was retrieved, so citations stay meaningful.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class RAGResponse(BaseModel):
    answer: str = Field(
        description="Answer to the user's question, grounded only in the provided context."
    )
    sources: list[str] = Field(
        description="doc_id values of the context chunks actually used to produce the answer."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "high: context directly and fully answers the question. "
            "medium: context is relevant but partial, or requires inference across chunks. "
            "low: context is only tangentially related, contradictory, or largely absent."
        )
    )


SYSTEM_PROMPT = """You are the internal support assistant for Helix, a B2B SaaS workflow \
automation platform. You answer Customer Success team questions using ONLY the context \
provided below — product docs, internal runbooks, and resolved support tickets.

Rules:
1. Base your answer strictly on the provided context. Do not use outside knowledge about \
Helix — the context is your only source of truth.
2. If the context does not contain enough information to answer confidently, say so \
explicitly (e.g. "The provided context doesn't cover this") rather than guessing or filling \
gaps with plausible-sounding information. This matters more than sounding complete.
3. In `sources`, list ONLY the raw doc_id value (e.g. "product-docs/api/errors.md") of every \
context chunk you actually relied on — never include the word "doc_id" or brackets, and do not \
invent doc_ids that weren't provided.
4. Set `confidence` using the definitions provided for that field.

Context:
{context}
"""


def _format_context(context_chunks: list[dict]) -> str:
    if not context_chunks:
        return "(no context retrieved)"
    blocks = [f"SOURCE: {c['doc_id']}\n{c['text']}" for c in context_chunks]
    return "\n\n---\n\n".join(blocks)


def build_generation_chain():
    model = ChatOpenAI(
        model=os.environ.get("GENERATION_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    structured_model = model.with_structured_output(RAGResponse)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    return prompt | structured_model


_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = build_generation_chain()
    return _chain


def generate(query: str, context_chunks: list[dict]) -> RAGResponse:
    chain = _get_chain()
    context_str = _format_context(context_chunks)
    return chain.invoke({"context": context_str, "question": query})