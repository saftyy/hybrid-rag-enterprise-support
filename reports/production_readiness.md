# Production Readiness — Helix CSM Assistant

**Decision requested:** roll out to all 40 senior CSMs?
**Recommendation:** ⟨SHIP / HOLD / KILL⟩ — ⟨one sentence why⟩
**Config evaluated:** ⟨v2 or v2-vector⟩ · commit ⟨ ⟩ · eval dataset `helix-csm-eval-v1.0` · judge `gpt-4o`

> Fill every `⟨…⟩` from the result files named in each section. Section 1 was written
> **before** the v2 results were known, so the decision can't be fitted to the numbers.

## 1. Decision criteria (pre-registered)

| # | Criterion | Threshold | Why this threshold |
|---|---|---|---|
| G1 | CSM resolution quality (custom judge) | ≥ 0.75 | Checkpoint gate; below it, roughly 1 in 4 answers needs a CSM to verify elsewhere, which removes the point of the tool |
| G2 | Faithfulness | ≥ 0.90 **and** ≥ v1 + 0.05 | Hallucinated facts reach customers through the CSM |
| G3 | Unauthorised commitments on eval set | 0 | A single invented refund/SLA promise is a contractual risk |
| G4 | Source hit rate | ≥ 0.95 | Below this, failures are retrieval failures the prompt can't fix |
| G5 | End-to-end p95 latency | < 10 s | Used live on customer calls |
| G6 | Data contract | GE suite passes; no PII in the index | Customer PII must not be retrievable |
| G7 | Coverage of CSM workflows | Critical CSM runbooks ingested and represented in the eval set | The users are CSMs |

**Rule:** all of G1–G7 → **SHIP**. G1–G6 pass but G7 or a named risk is open → **HOLD**
(limited pilot while it is closed). G1 or G2 fails with no credible fix path → **KILL**
this approach.

## 2. Evaluation results (LangSmith)

Source: `ops/langsmith/results/v1.json`, `v2.json` (and `v2-vector.json` if run).
Experiments: ⟨names⟩ · Project: ⟨URL⟩

| Metric | v1 (Module 1) | v2 | v2-vector | Gate |
|---|---|---|---|---|
| Faithfulness | ⟨ ⟩ | ⟨ ⟩ | ⟨ ⟩ | G2 |
| Answer relevance | ⟨ ⟩ | ⟨ ⟩ | ⟨ ⟩ | |
| CSM resolution quality | ⟨ ⟩ | ⟨ ⟩ | ⟨ ⟩ | G1 |
| Source hit | ⟨ ⟩ | ⟨ ⟩ | ⟨ ⟩ | G4 |
| Abstention rate | ⟨ ⟩ | ⟨ ⟩ | ⟨ ⟩ | |
| Unauthorised commitments (count) | ⟨ ⟩ | ⟨ ⟩ | ⟨ ⟩ | G3 |

By difficulty (CSM quality): easy ⟨ ⟩ · medium ⟨ ⟩ · hard ⟨ ⟩. Hard queries are the
cross-corpus ones a senior CSM is most likely to ask.

**Judge calibration** (`ops/langsmith/calibration/calibration_report.md`): n=15, MAE ⟨ ⟩,
Cohen's κ at the 0.75 gate ⟨ ⟩. ⟨Judge harsher/softer than me by ⟨ ⟩ on average; what I
changed, if anything.⟩

## 3. Observability findings (Phoenix)

Source: `ops/phoenix/analysis.md`. Headline: Module 1 retrieval frequently handed the
LLM heading-only fragments (54% of chunks < 200 chars). v2 merges them. Effect:
⟨Q037 now answered / abstention ⟨ ⟩ → ⟨ ⟩⟩. Cost of the fix: prompt tokens ⟨+ ⟩% per
query. Latency p95: retrieval ⟨ ⟩ ms, end-to-end ⟨ ⟩ ms (G5).

## 4. Data quality verdict

Source: `ops/data_quality/results/validation_v1.md`, `validation_v2.md`.

| | v1 | v2 |
|---|---|---|
| Expectations passed | 14/18 | ⟨18/18⟩ |
| Chunks < 200 chars | 232 / 426 (54%) | 1 / 176 |
| Median chunk length | 186 | 637 |
| Chunks containing customer PII | 2 (ticket emails) | 0 |
| Orphan fragments | 1 (`database-failover.pdf::2`) | 0 |
| Embedding dimension | ⟨1536 all⟩ | ⟨1536 all⟩ |

Verdict: ⟨PASS for v2⟩. Known limitation: PII detection is regex-based — it catches
emails, phone numbers, SSNs and card numbers, **not customer names** (ticket headers
contain names like the customer contact and company). An NER-based scanner (e.g.
Microsoft Presidio) is needed before this touches external data.

## 5. Risks

| Risk | Evidence | Impact | Mitigation |
|---|---|---|---|
| **CSM runbooks missing** | 5 scanned PDFs are skipped at ingestion — and they are CSM-critical: `vip-escalation`, `custom-sla-handling`, `plan-downgrade-workflow`, `onboarding-checklist`, `qbr-template` | The assistant will abstain (or worse, answer from product docs) on exactly the escalation and SLA questions senior CSMs handle | OCR them (Tesseract is already in the dependency list) with a GE check on OCR output quality, then re-ingest |
| **Eval set doesn't look like the users' traffic** | Only 1 of 50 test queries is in the `csm` category; 14 are `api`. Only Q050 depends on a skipped runbook | Scores above may overstate quality for CSM-style questions | Add ~15 CSM-workflow queries written with a CSM lead → dataset v1.1 (the version/hash check forces a new dataset) |
| Judge is an LLM | Calibration n=15 | Scores could drift if OpenAI updates `gpt-4o` | Pin a dated judge model snapshot; re-run calibration quarterly |
| Customer names not redacted | See §4 | Minor internally; blocker if exposed externally | Presidio NER before any external use |
| Hybrid vs vector retrieval | Module 1: vector-only Hit@5 100% vs hybrid 94% | ⟨see v2 vs v2-vector⟩ | Adopt whichever wins on CSM quality, not on Hit@5 |

## 6. Recommendation

⟨SHIP / HOLD / KILL⟩, because ⟨map to G1–G7⟩.

⟨If HOLD, example wording: "Pilot with 5 CSMs for 2 weeks with thumbs-up/down feedback on,
while the 5 CSM runbooks are OCR'd and the eval set gains 15 CSM-workflow questions. Re-run
this report; ship to all 40 if G1–G7 hold on eval v1.1."⟩

### What would change my mind

1. ⟨e.g. If CSM quality on the new CSM-workflow questions is below 0.75 even after the
   runbooks are ingested → the corpus, not the pipeline, is the gap: HOLD becomes KILL
   until CS Ops documents those processes.⟩
2. ⟨e.g. If the pilot shows > 20% 👎 despite passing offline evals → the eval set is
   measuring the wrong thing; fix the eval set before trusting any score.⟩
3. ⟨e.g. If any unauthorised commitment appears in pilot traffic → stop the pilot
   immediately regardless of averages.⟩
