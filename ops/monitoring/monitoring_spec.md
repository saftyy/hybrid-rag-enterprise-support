# Helix CSM Assistant: Production Monitoring Specification

**Scope:** internal RAG assistant (`v2-vector`) for 40 senior CSMs, starting with a 5-CSM pilot. **Audience:** the SRE implementing alerting. **Status:** specification only; not deployed.

Baselines come from `ops/phoenix/results/phoenix_stats_v2-vector.json` (15 traced queries) and `ops/langsmith/results/v2-vector.json` (50-query eval). Both samples are small, so **every threshold is re-baselined after the first pilot week** on real traffic. Thresholds are written as a multiple of the baseline *and* as an absolute number, so the rule survives re-baselining.

## 1. Where the signals come from

| Source | What it provides | How |
|---|---|---|
| OpenTelemetry spans → Phoenix (or any OTLP backend) | per-stage latency, tokens, retrieved doc ids and scores, errors | `rag_query` / `retrieve` / `generate` / `ChatOpenAI` spans, as emitted by `ops/phoenix/instrument.py` |
| LangSmith traces | full inputs/outputs for debugging, user feedback | `@traceable` in `ops/rag.py` |
| Online judge job | reference-free quality on sampled traffic | hourly: sample 20% of answers, run `faithfulness` + `answer_relevance` |
| Nightly regression eval | reference-based quality | `python -m ops.langsmith.run_evals --config v2-vector --compare-to v2-vector --concurrency 1` on the golden set |
| Ingestion job | data contract | `python -m ops.ingest` exit code + `validation_<config>.json` |
| OpenAI / Pinecone billing | spend | daily pull |

**Expected volume:** 40 CSMs × ~15 questions/day ≈ **600 queries/day**, ≈ 75 per business hour (pilot: ~75/day). At this volume a 5-minute window may hold only ~6 queries, so rate-based alerts use 15–60 minute windows and a **minimum-count guard of 10 queries**.

**Severity:** **P1** = page on-call now. **P2** = ticket, fix within 1 business day. **P3** = weekly review.

**Warm-up requirement:** the first query after process start took 4.7 s in retrieval (Pinecone connection + retriever load) versus 230–620 ms warm. The service must run one warm-up query at startup, and latency alerts ignore the first 60 s after a deploy.

## 2. Quality metrics

| # | Metric | Definition | Baseline | Alert condition | Sev | Action |
|---|---|---|---|---|---|---|
| Q1 | Online faithfulness | Mean `faithfulness` over sampled answers (20%, min 30 per window) | 0.993 | 24h mean < 0.95 → P2. 24h mean < 0.90 → P1 | P2/P1 | Diff retrieved doc ids against the last good day in LangSmith; check for a recent ingestion or prompt change; roll back namespace/config |
| Q2 | Abstention rate | % of answers matching the abstention pattern (`ops/langsmith/evaluators/_judge.py::ABSTAIN_RE`) | 4% (eval), 6.7% (traces) | 1h rate > 15% (≈3× baseline) **and** ≥ 10 queries → P1: retrieval is likely returning nothing useful (empty namespace, wrong index, embedding-model mismatch). 7-day rate < 1% while Q1 drops → P2: the model is guessing instead of abstaining | P1/P2 | Check the Pinecone vector count for the namespace (expect 176); check `retrieve` spans for empty or low-score results |
| Q3 | "Refused but had material" | Abstentions whose rank-0 retrieval score ≥ 0.45, i.e. something relevant was retrieved (from `rag_query` output + `retrieve` span) | Q044: abstained with HX-1189 retrieved at score 0.49 | weekly count > 5 → P3 | P3 | Prompt completeness issue (see `ops/phoenix/analysis.md`, Q044); review a sample and add them to the eval set |
| Q4 | Retrieval top score | Median similarity of the rank-0 document (`retrieval.documents.0.document.score`) | observed 0.49 (Q044) – 0.68 (Q037); set the median from pilot week 1 | 24h median < week-1 median − 0.10 → P2 | P2 | Queries drifting away from corpus coverage (a new product area?) or index degradation; review the low-score queries |
| Q5 | Nightly regression: CSM resolution quality | `csm_resolution_quality` on the golden set | 0.843 (run-to-run noise ≈ 0.007) | < 0.75 (the ship gate) **or** a drop > 0.03 (≈4× noise) night-over-night → P2; block any deploy until resolved | P2 | Compare experiments in LangSmith (`python -m ops.langsmith.compare_runs`); bisect by config / commit |
| Q6 | Unauthorised commitments | Golden-set answers where the custom judge set `unauthorised_commitment = true` | 0 | ≥ 1 → P1 | P1 | A CSM could promise a customer something policy doesn't allow. Disable the rollout flag, fix the prompt, re-run the eval |
| Q7 | Negative feedback | 👎 / (👍 + 👎) in LangSmith feedback, 7-day | new | > 20% with ≥ 25 ratings → P2 | P2 | Read the 👎 traces; add the failures to the golden set (bump the dataset version) |

## 3. System metrics

| # | Metric | Definition | Baseline (warm) | Alert condition | Sev | Action |
|---|---|---|---|---|---|---|
| S1 | Retrieval latency p95 | Duration of `retrieve` spans | 230–620 ms per query (p50 360 ms) | p95 > **1.2 s** (≈2× the warm max) for 15 min with ≥ 10 queries → P2. p95 > 3 s for 15 min → P1 | P2/P1 | A sustained doubling is Pinecone-side (serverless latency, regional issue), not one bad query: check Pinecone status. Vector-only retrieval has no in-process BM25, so the app is rarely the cause |
| S2 | LLM latency p95 | Duration of `ChatOpenAI` spans | p50 1,029 ms, p95 2,584 ms | p95 > 6 s for 15 min → P2 | P2 | Check OpenAI status and prompt-token growth (C1) |
| S3 | End-to-end latency p95 | Duration of `rag_query` spans | p50 1,603 ms, p95 3,689 ms | p95 > **8 s** for 15 min → P2 | P2 | CSMs use this live on customer calls; beyond ~8–10 s they stop waiting. Usually S2 |
| S4 | Error rate | `rag_query` spans with status ERROR / all | 0 errors in 3 × 50 eval runs after the retry fix | > 5% over 15 min with ≥ 10 queries → P1. > 1% over 24h → P2 | P1/P2 | Group by exception: OpenAI 429 (we hit gpt-4o's 30k TPM on the judge during evals; generation uses gpt-4o-mini, but raise the tier if it recurs), 5xx (provider), Pinecone errors, structured-output parse failures (prompt/model regression) |
| S5 | Empty retrieval | Queries with `rag.retrieval.k` = 0 | 0 | any occurrence → P1 | P1 | Namespace missing or chunk file absent; the model will abstain on everything |
| S6 | Index/chunk consistency | Pinecone vector count in namespace `v2` vs rows in `ops/data/chunks_v2.json` | 176 = 176 | mismatch after an ingestion → P2 | P2 | Re-run `python -m ops.ingest` |

## 4. Cost metrics

| # | Metric | Definition | Baseline | Alert condition | Sev | Action |
|---|---|---|---|---|---|---|
| C1 | Prompt tokens per query | `llm.token_count.prompt` on the `ChatOpenAI` span | mean 1,552, p95 2,039 | 24h p95 > **3,000** (≈1.5×) → P2 | P2 | Context bloat: chunk size or k changed, or a runaway chunk (the GE max is 3,000 chars). Check the chunk-length report |
| C2 | Generation cost per query | tokens × gpt-4o-mini price ($0.15 / 1M in, $0.60 / 1M out) | $0.00030 | > 2× baseline over 24h → P2 | P2 | Usually C1, or someone changed `GENERATION_MODEL` (the LLM span records the exact model snapshot, `gpt-4o-mini-2024-07-18`) |
| C3 | Daily spend | billing APIs | generation ≈ $0.18/day at 600 queries. Online judging (gpt-4o, 2 evaluators on 20% of traffic) ≈ 120 × ~$0.01 ≈ **$1.20/day**, estimated at list prices; verify on the first bill. Nightly regression eval ≈ $1/run | total > 150% of forecast → P2. Monthly budget of $100 at 80% → P2, at 100% → P1 | P2/P1 | Judging costs ~7× more than answering: reduce the judge sampling rate first |

## 5. Data-quality contract (ingestion)

| # | Check | Alert | Sev | Action |
|---|---|---|---|---|
| D1 | GE suite `helix_chunks_contract` (18 expectations) on every ingestion | any failure → ingestion blocks before upsert and exits 1 | P2 | The previous namespace stays live, so users are unaffected. Read `ops/data_quality/results/validation_<config>.md`. PII failures go to the data owner, not just engineering. Verified in practice: a wrong corpus path loaded 0 documents and the gate blocked the upload |
| D2 | Days since last successful ingestion | > 14 days while docs changed | P3 | Stale answers about changed pricing or limits |

## 6. Top 5 (dashboard front page)

1. **Q6 unauthorised commitments = 0**: the only failure that directly costs money and trust with a customer.
2. **Q2 abstention rate**: the cheapest signal that retrieval broke, with no LLM call needed.
3. **S1 retrieval p95 > 1.2 s (15 min)**: warm retrieval is 230–620 ms, so a sustained doubling means Pinecone, not a bad query.
4. **Q1 online faithfulness**: an early hallucination warning without needing references.
5. **C1 prompt tokens p95 > 3,000**: context bloat shows up here before it shows up on the bill.
