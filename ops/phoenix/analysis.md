# Phoenix trace analysis

## Setup

`python -m ops.phoenix.instrument --config <cfg>` traced the same 15 queries for `v1` (Module 1) and `v2-vector` (final): the 7 known Module 1 failures (Q017, Q029, Q033, Q037, Q038, Q044, Q046) plus 8 passing queries across categories. Each query produces `rag_query → retrieve (RETRIEVER) → generate → RunnableSequence → ChatOpenAI (LLM)`, carrying document ids, similarity scores, chunk text, chunk lengths, token counts and per-stage latency. Raw numbers: `results/phoenix_stats_v1.json`, `results/phoenix_stats_v2-vector.json`.

## Baseline numbers

| Signal | v1 p50 | v1 p95 | v2-vector p50 | v2-vector p95 |
|---|---|---|---|---|
| Retrieval (RETRIEVER span) | 284 ms | 348 ms | 360 ms | 1,461 ms* |
| LLM (ChatOpenAI span) | 939 ms | 2,535 ms | 1,029 ms | 2,584 ms |
| End-to-end (`rag_query`) | 1,247 ms | 3,073 ms | 1,603 ms | 3,689 ms |
| Prompt tokens per call | 1,258 (mean) | | 1,552 (mean) | 2,039 |
| Cost per query | $0.00024 | | $0.00030 | |

\* A cold-start artifact. The first query of the run (Q001) took 4,739 ms in retrieval while the client connected to Pinecone and loaded the retriever. Warm retrieval queries took 230–620 ms. With n=15, p95 is effectively the second-slowest query, so the production thresholds in the monitoring spec use warm numbers and should be re-baselined on real pilot traffic.

The LLM call is 60–65% of end-to-end latency (Q037: 947 ms of 1.6 s), so retrieval is not the lever for user-perceived speed.

## Weakness found: retrieval returned heading fragments, not answers

**What the traces showed.** Q037, "How do I send a Helix webhook to a custom URL?", abstained in v1 ("The provided context doesn't cover this"). Its RETRIEVER span (`screenshots/06_v1_q037_attributes.png`) shows `chunk_lengths [96, 209, 489, 243, 85]` and `short_chunks: 2`. The right document, `integrations/custom-webhooks.md`, was retrieved, but as a heading plus one sentence. The LLM received only **786 tokens** for the whole question. Across all 15 v1 traces, **29.3% of retrieved chunks were under 200 characters** (median 316).

**Root cause (verified in the chunk file, not just the traces).** Module 1 splits Markdown on every `#`/`##`/`###` header. The Helix docs have many short sections, so **232 of 426 chunks (54%) are under 200 characters, with a median of 186**. For example, `api/api-versioning.md::1` is 58 characters: "## Current version — **v2** is the current stable version." A short chunk embeds as "a heading about X", which scores well for questions about X, so it wins retrieval slots while carrying no instructions. With k=5, every fragment that wins a slot pushes out a chunk that could have answered the question.

The Great Expectations suite flags the same problem independently (`../data_quality/results/validation_v1.md`: chunk-length and median expectations fail). It also caught a 19-character orphan, `runbooks/incident-response/database-failover.pdf::2` = "(RPO is 5 minutes).", cut off from the failover procedure it belongs to.

## What I changed

1. Merge consecutive Markdown sections of the same document until each chunk is ≥ 400 characters (cap 1,500). Never across documents, so citations stay correct.
2. Fold orphan fragments (< 100 characters) in PDFs and tickets back into the previous chunk.
3. Redact customer PII before embedding.
4. Retrieval: vector-only (`v2-vector`). Module 1's own `compare_retrieval.py` had shown vector-only beating hybrid on Hit@5, and on the merged chunks it also won on every eval metric (CSM 0.843 vs 0.801 for hybrid).

Result: 426 → 176 chunks, median 186 → 637 characters, 232 → ~0 chunks under 200 characters.

## Before vs after

| Metric | v1 | v2-vector | Change |
|---|---|---|---|
| Retrieved chunks < 200 chars (15 q) | 29.3% | **0%** | fragments eliminated |
| Median retrieved chunk length | 316 | 736 | 2.3× |
| Q037 chunk lengths | `[96, 209, 489, 243, 85]` | `[564, 420, 489, 583, 933]` | `short_chunks` 2 → 0 |
| Q037 outcome | abstained | answered, grounded word for word | but see "remaining failures" |
| Source hit (15 q) | 86.7% | **100%** | |
| Abstention (15 q) | 26.7% | **6.7%** | 4 → 1 |
| Prompt tokens per call (mean) | 1,258 | 1,552 | **+23%** |
| LLM latency p95 | 2,535 ms | 2,584 ms | ≈ unchanged |
| CSM resolution quality (50 q, LangSmith) | 0.767 | **0.843** | +0.076 (noise ≈ 0.007) |
| Faithfulness (50 q, LangSmith) | 0.992 | 0.993 | at ceiling |

**Trade-off:** larger chunks cost 23% more prompt tokens, which works out to about $0.00006 more per query. At 40 CSMs × ~15 questions/day (≈600 queries), that's about **$0.18/day in total generation cost**. The quality gain costs almost nothing.

## Remaining failures (traced, not fixed)

**Q044: the answer was retrieved, and the model still abstained** (`screenshots/04a`, `04b`, `05`). "We're getting paged at 3am because production is failing on 401s but the API key wasn't rotated. What's the standard runbook?" Rank 1 was ticket HX-1189, which describes the identical incident (key deleted by another user; audit log; generate a new key; scoped keys). The model returned `"The provided context doesn't cover this", sources: [], confidence: "low"`. Cause: the question asks for a *runbook*, the answer is in a *ticket*, and Module 1's prompt rule 2 ("say so if the context doesn't contain enough information") made the model refuse. It's a generation failure. The v3 prompt's rule 7 ("answer the covered part, name what's missing") fixed exactly this case, but v3 was rejected overall because other answers lost required facts. Next step: test rule 7 alone as a single change. A cheap product guardrail: when `confidence == "low"`, show the CSM the retrieved sources anyway.

Also visible in this trace: the top-ranked document was `on-call-rotation.pdf` (score 0.49), matched on the situation words ("paged", "3am", "runbook") rather than the problem ("401 with a valid key"). Scores were nearly tied (0.49 / 0.49), which is a case for query rewriting, or keeping BM25 for queries with error codes.

**Q037: correct retrieval, ambiguous question** (`screenshots/02`, `03a–03d`). Both features were retrieved: webhook *triggers* (`custom-webhooks.md`, ranks 0/2/3) and webhook *subscriptions* (`api/webhooks.md`, rank 1). Scores were bunched at 0.68 / 0.64 / 0.56 / 0.54 / 0.53, with no clear winner. The model answered from the rank-3 chunk, which literally says "To send a webhook from a workflow, use the HTTP action…". That's faithful (1.0) but not what the reference expects (CSM 0.06). Also, 3 of 5 slots came from one document (`distinct_docs: 3`). Next steps: retrieval diversification (MMR, or at most 2 chunks per document); an instruction to name both features when the context covers two; and rewriting Q037 unambiguously in dataset v1.1. Measure with Q037's CSM score plus 3–4 deliberately ambiguous questions.

**Q050: the answer is in a document the system cannot see.** The expected source `custom-sla-handling.pdf` is one of 5 scanned CSM runbooks skipped at ingestion. The answer covers generic P0 communication but not the custom-SLA obligations. This is the main production risk.

**PII: names are not redacted** (`screenshots/04b`). Ticket HX-1189 shows `[REDACTED_EMAIL]` (the redaction works), but the customer's name and company are still in plain text. Regex detection can't catch names; an NER tool (e.g. Presidio) is needed before any external use.
