# Helix CSM Assistant — Production Monitoring Specification

**Scope:** internal RAG assistant for 40 senior CSMs. **Audience:** the SRE implementing
alerting. **Status:** specification only; not deployed.

Values in `⟨…⟩` are baselines measured in this repo; fill them from the named file before
handing this to SRE. Each threshold is written relative to that baseline *and* as an
absolute number, so it stays meaningful if the baseline is re-measured.

## 1. Where the signals come from

| Source | What it provides | How |
|---|---|---|
| OpenTelemetry spans → Phoenix (or any OTLP backend) | per-stage latency, tokens, retrieved doc ids/scores, errors | `rag_query` / `retrieve` / `generate` / LLM spans, as emitted by `ops/phoenix/instrument.py` |
| LangSmith traces | full inputs/outputs for debugging, user feedback | `@traceable` in `ops/rag.py` |
| Online judge job | reference-free quality on sampled traffic | hourly job: sample traffic, run `faithfulness` + `answer_relevance` evaluators |
| Nightly regression eval | reference-based quality | `python -m ops.langsmith.run_evals --config <prod> --compare-to <prod>` against the golden set |
| Ingestion job | data contract | `python -m ops.ingest` (GE gate) exit code + `validation_<config>.json` |
| OpenAI / Pinecone billing APIs | spend | daily pull |

**Expected volume:** 40 CSMs × ⟨~15⟩ queries/day ≈ ⟨600⟩ queries/day, ≈ ⟨75⟩/business
hour. Low volume matters: a 5-minute window may hold only ~6 queries, so rate-based alerts
use longer windows and minimum-count guards (stated per alert).

**Severity:** **P1** = page on-call now. **P2** = ticket, fix within 1 business day.
**P3** = weekly review.

## 2. Quality metrics

| # | Metric | Definition | Baseline | Alert condition | Sev | Action |
|---|---|---|---|---|---|---|
| Q1 | Online faithfulness | Mean `faithfulness` over sampled prod answers (20% of traffic, min 30 answers) | ⟨v2 eval faithfulness⟩ | 24h mean < baseline − 0.05 → P2. 24h mean < 0.80 → P1 | P2/P1 | Diff retrieved doc ids vs. last good day in LangSmith; check for a recent ingestion or prompt change; roll back to previous namespace/config |
| Q2 | Abstention rate | % answers matching the abstention pattern (`ops/langsmith/evaluators/__init__.py`) | ⟨v2 abstained⟩ | 1h rate > 3× baseline **and** ≥ 10 queries → P1 (retrieval likely returning nothing useful: empty namespace, wrong index, embedding model mismatch). 7-day rate < ½ baseline while Q1 drops → P2 (model guessing instead of abstaining) | P1/P2 | Check Pinecone vector count for the namespace; check `retrieve` spans for empty/low-score results |
| Q3 | Retrieval top-score | Median of the top vector similarity per query (`retrieval.documents.0.document.score`, vector mode) | ⟨ ⟩ | 24h median < baseline − 0.10 → P2 | P2 | Queries drifting from corpus coverage (new product area?) or index degraded; sample low-score queries and review |
| Q4 | Nightly regression — CSM resolution quality | `csm_resolution_quality` on golden set | ⟨v2 score⟩ | < 0.75 (the ship gate) **or** drop > 0.05 night-over-night → P2; block any deploy until resolved | P2 | Compare experiments in LangSmith; bisect by config / commit |
| Q5 | Unauthorised commitments | Count of golden-set answers where the custom judge set `unauthorised_commitment=true` | 0 | ≥ 1 → P1 | P1 | A CSM could promise a customer something policy doesn't allow. Disable rollout flag, fix prompt, re-run eval |
| Q6 | Negative feedback | 👎 / (👍 + 👎) in LangSmith feedback, 7-day | n/a (new) | > 20% with ≥ 25 ratings → P2 | P2 | Read the 👎 traces; add failures to the golden set (bump dataset version) |

## 3. System metrics

| # | Metric | Definition | Baseline | Alert condition | Sev | Action |
|---|---|---|---|---|---|---|
| S1 | Retrieval latency p95 | Duration of `retrieve` spans | p95 ⟨ ⟩ ms (`phoenix_stats_v2.json`) | p95 > 2× baseline (≈ ⟨ ⟩ ms) sustained 15 min, ≥ 10 queries → P2. > 3 s for 15 min → P1 | P2/P1 | Doubling that persists is Pinecone-side (serverless cold start / regional issue), not one bad query. Check Pinecone status; BM25 is in-process so it is not the cause unless the chunk file grew |
| S2 | End-to-end latency p95 | Duration of `rag_query` spans | p95 ⟨ ⟩ ms | > 10 s sustained 15 min → P2 | P2 | CSMs use this live with customers; beyond ~10 s they stop waiting. LLM span is the usual culprit — check OpenAI status / token growth (C1) |
| S3 | Error rate | `rag_query` spans with status ERROR / all | 0% in eval runs | > 5% over 15 min with ≥ 10 queries → P1. > 1% over 24h → P2 | P1/P2 | Group by exception type: OpenAI 429 (rate limit — raise tier / add retry), 5xx (provider), Pinecone errors, Pydantic structured-output parse failures (prompt/model regression) |
| S4 | Empty retrieval | Queries where `rag.retrieval.k` = 0 | 0 | any occurrence → P1 | P1 | Index/namespace missing or chunk file absent; the model will abstain on everything |
| S5 | Index/chunk consistency | Pinecone vector count in the prod namespace vs. rows in the prod chunk file | equal (176 for v2) | mismatch after an ingestion → P2 | P2 | BM25 and vector search are working over different corpora; re-run `python -m ops.ingest` |

## 4. Cost metrics

| # | Metric | Definition | Baseline | Alert condition | Sev | Action |
|---|---|---|---|---|---|---|
| C1 | Prompt tokens per query | `llm.token_count.prompt` on the LLM span | mean ⟨ ⟩, p95 ⟨ ⟩ | 24h p95 > 1.5× baseline → P2 | P2 | Context bloat: chunk size or k changed, or a runaway ticket chunk. Check GE chunk-length report |
| C2 | Cost per query | tokens × price (gpt-4o-mini: $0.15 / 1M in, $0.60 / 1M out) + embedding | ⟨ ⟩ USD | > 2× baseline over 24h → P2 | P2 | Usually C1, or someone changed `GENERATION_MODEL` |
| C3 | Daily spend (generation + online judge) | from billing APIs | forecast = volume × C2 + judge sampling (20% × ⟨judge cost⟩) ≈ ⟨ ⟩ USD/day | > 150% of forecast → P2. Monthly budget ⟨ ⟩ USD at 80% → P2, 100% → P1 | P2/P1 | Check volume vs. per-query cost; reduce judge sampling rate first (it is the easiest lever) |

## 5. Data-quality contract (ingestion)

| # | Check | Alert | Sev | Action |
|---|---|---|---|---|
| D1 | GE suite `helix_chunks_contract` (18 expectations) on every ingestion | any failure → ingestion blocks before upsert, exits 1 | P2 | The previous namespace stays live, so users are unaffected. Read `ops/data_quality/results/validation_<config>.md`; PII failures go to the data owner, not just engineering |
| D2 | Days since last successful ingestion | > 14 days while docs changed | P3 | Stale answers about changed pricing/limits |

## 6. Top 5 (dashboard front page)

1. **Q5 unauthorised commitments = 0** — the only failure that directly costs money and trust with a customer.
2. **Q1 online faithfulness** — early warning for hallucination without needing references.
3. **Q2 abstention rate** — the cheapest signal that retrieval broke (no LLM call needed).
4. **S2 end-to-end p95 latency** — the thing CSMs actually feel.
5. **C1 prompt tokens / query** — catches context bloat before it shows up on the bill.
