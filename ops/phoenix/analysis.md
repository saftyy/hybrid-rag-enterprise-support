# Phoenix trace analysis

> **How to fill this in:** every `⟨…⟩` is a number you take from
> `ops/phoenix/results/phoenix_stats_v1.json` / `_v2.json` or from a screenshot. Everything
> not in `⟨…⟩` is already verified from the corpus and chunk files. Delete this note when done.

## Setup

`python -m ops.phoenix.instrument --config v1` traced 15 queries (the 7 known Module 1
failures — Q017, Q029, Q033, Q037, Q038, Q044, Q046 — plus 8 passing queries across
categories). Each query produces `rag_query → retrieve (RETRIEVER) → generate → ChatOpenAI
(LLM)` with document ids, retrieval scores, chunk text, token counts and per-stage latency.
Screenshots: `screenshots/`.

## Baseline numbers (v1)

| Signal | p50 | p95 | Source |
|---|---|---|---|
| Retrieval latency (RETRIEVER span) | ⟨ ⟩ ms | ⟨ ⟩ ms | phoenix_spans.retrieval_latency_ms |
| LLM latency (LLM span) | ⟨ ⟩ ms | ⟨ ⟩ ms | phoenix_spans.llm_latency_ms |
| End-to-end (`rag_query`) | ⟨ ⟩ ms | ⟨ ⟩ ms | phoenix_spans.end_to_end_latency_ms |
| Prompt tokens / call | ⟨ ⟩ (mean) | ⟨ ⟩ | tokens_per_llm_call.prompt |
| Est. cost / query | ⟨ ⟩ USD | | est_cost_per_query_usd |

Generation dominates latency (⟨ ⟩% of end-to-end at p50) — expected for a single
gpt-4o-mini call — so retrieval latency is not the lever for user-perceived speed.

## Weakness found: retrieval returns heading fragments, not answers

**What the traces showed.** Opening the RETRIEVER span for **Q037** ("How do I send a
Helix webhook to a custom URL?") shows the correct document,
`product-docs/integrations/custom-webhooks.md`, *was* retrieved — but the chunk handed to
the LLM was ⟨ ⟩ characters long and contained ⟨the heading and one sentence — describe
what you see⟩, with none of the setup steps. The LLM correctly said "the provided context
doesn't cover this". Screenshot: `screenshots/q037_retriever_span.png`.

Across all 15 traced queries, **⟨ ⟩% of retrieved chunks were under 200 characters**
(`retrieved_chunks_under_200_pct`) and the median retrieved chunk was ⟨ ⟩ characters.

**Root cause (verified in the chunk file, not just the traces).** Module 1 splits Markdown
on every `#`/`##`/`###` header. The Helix docs have many short sections, so **232 of 426
chunks (54%) are under 200 characters; median 186**. Examples:
`api/api-versioning.md::1` (58 chars: "## Current version — **v2** is the current stable
version."), `api/runs-endpoint.md::0` (52 chars). Short chunks embed as "a heading about
X", which scores well on similarity for questions about X — so they win retrieval slots
while carrying no instructions. With k=5, every fragment that wins a slot pushes out a
chunk that could have answered the question.

The Great Expectations suite independently flags the same problem (see
`ops/data_quality/results/validation_v1.md`): chunk-length and median-length
expectations fail. It also caught a 19-character orphan chunk,
`runbooks/incident-response/database-failover.pdf::2` = "(RPO is 5 minutes).", cut off
from the failover procedure it belongs to.

## What I changed (config `v2`)

1. Merge consecutive Markdown sections of the same document until each chunk is
   ≥ 400 characters (cap 1500). Never across documents, so citations stay correct.
2. Fold orphan fragments (< 100 chars) in PDFs/tickets back into the previous chunk.
3. Retrieval is unchanged (hybrid RRF, k=5) so the effect of chunking is isolated.

Result: 426 → 176 chunks, median 186 → 637 chars, chunks < 200 chars 232 → 1.

## Before vs after

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| Retrieved chunks < 200 chars | ⟨ ⟩% | ⟨ ⟩% | |
| Median retrieved chunk length | ⟨ ⟩ | ⟨ ⟩ | |
| Q037 outcome | abstained | ⟨ ⟩ | |
| Source hit rate (15 queries) | ⟨ ⟩ | ⟨ ⟩ | |
| Abstention rate (15 queries) | ⟨ ⟩ | ⟨ ⟩ | |
| Prompt tokens / call (mean) | ⟨ ⟩ | ⟨ ⟩ | cost of bigger chunks |
| LLM latency p95 | ⟨ ⟩ ms | ⟨ ⟩ ms | |
| Faithfulness (50q, LangSmith) | ⟨ ⟩ | ⟨ ⟩ | |
| CSM resolution quality (50q) | ⟨ ⟩ | ⟨ ⟩ | |

**Trade-off to watch:** larger chunks mean more prompt tokens per query. The ⟨ ⟩%
increase in prompt tokens is ⟨acceptable / not⟩ given ⟨ ⟩ — see cost metrics in the
monitoring spec.

## Other observations (not fixed)

- ⟨e.g. Q017 "HTTP 401": did `api/errors.md` appear in the candidate pool at all? Check
  `vector_rank` / `bm25_rank` in the document metadata of the RETRIEVER span.⟩
- ⟨e.g. Q046: 4 expected sources, k=5 — how many distinct docs did retrieval return?
  (`rag.retrieval.distinct_docs`)⟩
