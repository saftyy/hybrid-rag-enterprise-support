# Phoenix screenshots

Referenced from `../analysis.md`. Captured from the local Phoenix session (`python -m ops.phoenix.instrument`), 15 traced queries per config.

| File | Config | What it shows |
|---|---|---|
| `01_v2vector_span_list.png` | v2-vector | All 15 `rag_query` traces: P50 1.6 s, P99 6.6 s (dominated by one cold start, 7.1 s on the first query), token counts, cost |
| `02_q037_trace_tree.png` | v2-vector | Full span tree for Q037: `retrieve` 618 ms → `ChatOpenAI` 947 ms (1,151 tokens); the answer describes the outbound HTTP action |
| `03a_q037_rank0_custom_webhooks.png` | v2-vector | Rank 0 (score 0.68): a full `custom-webhooks.md` section, no longer a heading fragment |
| `03b_q037_rank1_api_webhooks.png` | v2-vector | Rank 1 (score 0.64): webhook *subscriptions*, the meaning the reference answer expects |
| `03c_q037_outbound_chunk.png` | v2-vector | Ranks 2–3: the "Outbound webhooks" text the model repeated word for word; faithful, but the other meaning of an ambiguous question |
| `03d_q037_attributes_v2vector.png` | v2-vector | **After:** `chunk_lengths [564, 420, 489, 583, 933]`, `short_chunks: 0`, `distinct_docs: 3` |
| `06_v1_q037_attributes.png` | v1 | **Before:** `chunk_lengths [96, 209, 489, 243, 85]`, `short_chunks: 2`, only 786 tokens sent to the LLM |
| `04a_q044_retrieve_top.png` | v2-vector | Q044: rank 0 is `on-call-rotation.pdf` (score 0.49), matched on "paged / 3am / runbook" rather than the 401 problem |
| `04b_q044_ticket_redacted.png` | v2-vector | Rank 1: ticket HX-1189 contains the full answer; `[REDACTED_EMAIL]` shows PII redaction working, while the customer name shows its limitation |
| `05_q044_llm_span.png` | v2-vector | The LLM output despite that context: `"The provided context doesn't cover this"`, `sources: []`, `confidence: "low"`; model `gpt-4o-mini-2024-07-18`, 1,801 tokens |
