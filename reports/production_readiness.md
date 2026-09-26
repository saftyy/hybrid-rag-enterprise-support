# Production Readiness: Helix CSM Assistant

**Decision requested:** roll out to all 40 senior CSMs?
**Recommendation: HOLD.** Pilot with 5 CSMs while the 5 CSM runbooks are ingested and the eval set gains CSM-workflow questions.
**Config evaluated:** `v2-vector` (merged chunks + PII redaction + vector-only retrieval, k=5, Module 1 prompt) · eval dataset `helix-csm-eval-v1.0` (50 q) · judge `gpt-4o` · generator `gpt-4o-mini-2024-07-18`

The pipeline answers product and operational questions well: 84% CSM quality under a judge that is, if anything, harsher than a human. But the documents senior CSMs need most (escalation, custom SLA, downgrade, onboarding and QBR runbooks) were never ingested, and the evaluation barely tests that kind of question. Shipping to 40 people now would mean trusting the scores exactly where they are weakest.

## 1. Decision criteria (pre-registered)

These were written before the v2 results existed. Where a criterion turned out to be badly specified, that is stated rather than the criterion being rewritten.

| # | Criterion | Threshold | Result | Status |
|---|---|---|---|---|
| G1 | CSM resolution quality (custom judge) | ≥ 0.75 | **0.843** | ✅ PASS |
| G2 | Faithfulness | ≥ 0.90 **and** ≥ v1 + 0.05 | **0.993**; Δ **+0.001** vs v1 (0.992) | ✅ absolute / ❌ delta: **mis-specified**, see below |
| G3 | Unauthorised commitments on eval set | 0 | 0 (see below) | ✅ PASS |
| G4 | Source hit rate | ≥ 0.95 | **1.000** | ✅ PASS |
| G5 | End-to-end p95 latency | < 10 s | **3.7 s** (15 traced queries) | ✅ PASS |
| G6 | Data contract; no PII in the index | GE passes; no emails/phones | **18/18**; 2 customer emails redacted | ✅ PASS (names: known limitation) |
| G7 | CSM workflows covered | critical CSM runbooks ingested and represented in the eval set | 5 CSM runbooks missing; 1/50 CSM questions | ❌ **FAIL** |

**Decision rule:** all of G1–G7 → SHIP. G1–G6 pass but G7 or a named risk is open → **HOLD** (limited pilot while it is closed). G1 or G2 fails with no credible fix path → KILL.

**G2's delta clause cannot be met by any system.** Once abstentions are scored consistently, Module 1 already scores 0.992 on faithfulness; the ceiling is 1.0, so the largest possible gain is +0.008. The clause assumed a baseline with room to improve. The part of G2 that protects users, "the system does not hallucinate" (≥ 0.90), passes at 0.993. An earlier run appeared to pass the delta (+0.064). Investigation showed the judge had scored Module 1's abstentions as false claims, contradicting the metric's own definition; after fixing the evaluator and re-running, the "pass" disappeared. That is reported here as a measurement finding, not treated as a system failure.

**G3 evidence:** the custom judge caps any answer with an unauthorised commitment at 0.25. Every v2-vector answer scoring above 0.25 therefore carried no commitment flag. The only two at or below 0.25 are Q044 (an abstention) and Q037 (a description of the wrong webhook feature), and neither makes a promise to a customer.

## 2. Evaluation results (LangSmith)

Source: `ops/langsmith/results/{v1,v2-vector,v3}.json`. Experiments `v1-95772092`, `v2-vector-18afbc62`, `v3-d4e9b0dd`. 50/50 examples scored in every run.

| Metric | v1 (Module 1) | v2 (hybrid)† | **v2-vector** | v3 (rejected) |
|---|---|---|---|---|
| CSM resolution quality | 0.767 | 0.801 | **0.843** | 0.817 |
| Faithfulness | 0.992 | — | 0.993 | 0.988 |
| Answer relevance | 0.935 | 0.930 | 0.980 | 0.990 |
| Source hit | 0.940 | 0.920 | **1.000** | 1.000 |
| Source recall | 0.795 | 0.807 | 0.908 | 0.908 |
| Abstention rate | 8% | 8% | **4%** | 4% |

† v2 was scored before the faithfulness-evaluator fix, so only the unaffected metrics are shown.

**Signal vs noise:** two runs of the unchanged v1 system gave CSM quality 0.774 and 0.767, so run-to-run noise is ≈ 0.007. v2-vector's +0.076 is about 10× that. Individual answers still vary between runs even at temperature 0: Q046 mentioned the goodwill-credit option in the Phoenix run but not in the eval run. Judge the system on averages, not single answers.

**Why v3 was rejected:** its stricter grounding prompt nudged faithfulness but dropped CSM quality from 0.843 to 0.817 (Q025, Q026 and Q038 fell from 1.0 to 0.53–0.65, because the model left out required facts to stay "safe"). The pre-registered rule said v3 only counts if CSM quality does not drop.

**Where the remaining quality gap is:** incompleteness, not hallucination. Most imperfect answers score 0.53, meaning "partial facts, medium actionability": correct, but missing one required fact from the grading notes (e.g. Q046 omits the goodwill-credit option; Q031 omits the 1-hour reactivation).

**Judge calibration** (`ops/langsmith/calibration/calibration_report.md`): 15 stratified answers, labels drafted by an AI assistant and reviewed/adjusted by me (⟨0⟩ changed). MAE **0.079**, 100% within ±0.25, Spearman **0.961**, agreement at the 0.75 gate 80%, Cohen's κ **0.545** (target 0.6, missed), judge − human = **−0.034**. The judge ranks answers almost exactly like the human labels but is slightly harsher. All gate disagreements come from the rubric having no level between "all facts" and "partial", so an answer missing one minor fact fails for the judge and passes for a human. G1 is therefore conservative, not inflated.

## 3. Observability findings (Phoenix)

Source: `ops/phoenix/analysis.md`, screenshots in `ops/phoenix/screenshots/`.

- **Weakness found and fixed:** 54% of Module 1 chunks were heading fragments. Q037's retrieved chunks went from `[96, 209, 489, 243, 85]` (abstained) to `[564, 420, 489, 583, 933]` (answered). Across 15 traced queries: fragments 29.3% → 0%, abstention 26.7% → 6.7%, source hit 86.7% → 100%.
- **Cost of the fix:** +23% prompt tokens (1,258 → 1,552), $0.00024 → $0.00030 per query, ≈ $0.18/day at full rollout. LLM p95 is unchanged (2.5–2.6 s).
- **Latency (warm):** retrieval 230–620 ms, end-to-end p50 1.6 s / p95 3.7 s. One 4.7 s cold start on the first query, which the monitoring spec handles with a startup warm-up.
- **Remaining failure modes, traced:** Q044 abstained although the answer was retrieved (ticket HX-1189 at rank 1; a prompt/completeness issue). Q037 is ambiguous: both webhook features were retrieved with tied scores, and the model picked the one the reference didn't intend.

## 4. Data quality verdict

Source: `ops/data_quality/results/validation_v1.md`, `validation_v2.md`.

| | v1 | v2 |
|---|---|---|
| Expectations passed | 14/18 | **18/18** |
| Chunks | 426 | 176 |
| Chunks < 200 chars | 232 (54.5%) | ~0 |
| Median chunk length | 186 | 637 |
| Chunks containing customer emails | 2 | 0 |
| Orphan fragments | 1 (`database-failover.pdf::2`) | 0 |
| Embedding dimension (read back from Pinecone) | 1536 | 1536 |

**Verdict: PASS for v2.** The gate has also worked in practice: an ingestion with a wrong corpus path loaded 0 documents and was blocked before touching Pinecone. **Limitation:** regex PII detection doesn't catch names. Ticket HX-1189 still shows the customer's name and company next to the redacted email.

## 5. Risks

| Risk | Evidence | Impact | Mitigation |
|---|---|---|---|
| **CSM runbooks missing** | 5 scanned PDFs skipped at ingestion: `vip-escalation`, `custom-sla-handling`, `plan-downgrade-workflow`, `onboarding-checklist`, `qbr-template`. Q050 needs `custom-sla-handling.pdf` and answers only the generic P0 part | Wrong or incomplete answers on exactly the escalation and SLA questions senior CSMs handle | OCR them (Tesseract is already a dependency) with a GE check on OCR text quality, then re-ingest |
| **Eval set doesn't look like the users' traffic** | 1 of 50 questions is `csm`, 14 are `api` | Scores may overstate quality for CSM-style questions | Write ~15 CSM-workflow questions with a CSM lead → dataset v1.1 (the hash check forces a new version) |
| Refusing when the answer is present | Q044: ticket with the answer at rank 1, model abstained (`confidence: low`) | CSM gets nothing on an urgent incident | Test v3's rule 7 alone; show retrieved sources when confidence is low; monitor metric Q3 |
| Ambiguous questions | Q037: two features tied in retrieval; 3 of 5 slots from one doc | Confidently answers the wrong feature | Retrieval diversification (≤ 2 chunks/doc), a "name both features" instruction |
| Incomplete answers | Most imperfect answers are "partial facts" (0.53) | CSM needs to double-check | Target completeness next; add a `most` level to the judge rubric |
| Judge is an LLM | κ 0.545 on n=15; labels AI-drafted, human-reviewed | Scores could drift if OpenAI updates `gpt-4o` | Pin a dated judge snapshot; recalibrate with fully human labels on dataset v1.1 |
| Names not redacted | HX-1189 | Minor internally; a blocker if exposed externally | NER-based PII detection (e.g. Presidio) |

## 6. Recommendation

**HOLD: pilot with 5 CSMs for 2 weeks**, with 👍/👎 feedback on and the monitoring spec's top 5 alerts live. G1–G6 pass; G7 fails, and it fails in exactly the area that matters most for this audience. During the pilot: OCR the 5 CSM runbooks, write the CSM-workflow questions (dataset v1.1), then re-run this report.

### What would change my mind

1. **→ SHIP to all 40:** with the runbooks ingested, CSM quality ≥ 0.75 on the new CSM-workflow questions, and pilot 👎 ≤ 20%.
2. **→ KILL for CSM workflows:** CSM quality stays < 0.75 on those questions even with the runbooks ingested, after two targeted iterations on answer completeness. Then restrict the assistant to product-doc Q&A, where it already performs well.
3. **→ Stop the pilot immediately:** any unauthorised commitment in pilot traffic, regardless of averages.
