# Eval results — config `v2-vector`

Experiment: `v2-vector-18afbc62` · dataset `helix-csm-eval-v1.0` · commit `f8ac26b` · judge `gpt-4o` · n=50

| Metric | Score |
|---|---|
| faithfulness | 0.993 (50/50) |
| answer_relevance | 0.980 (50/50) |
| csm_resolution_quality | 0.843 (50/50) |
| source_hit | 1.000 (50/50) |
| source_recall | 0.908 (50/50) |
| abstained | 0.040 (50/50) |
| citation_valid | 1.000 (50/50) |

| Latency | ms |
|---|---|
| retrieval_p50_ms | 288.0 |
| retrieval_p95_ms | 470.7 |
| generation_p50_ms | 1085.9 |
| generation_p95_ms | 2779.3 |

| Difficulty | n | faithfulness | relevance | csm_quality |
|---|---|---|---|---|
| easy | 20 | 1.000 | 1.000 | 1.000 |
| hard | 10 | 0.983 | 0.900 | 0.636 |
| medium | 20 | 0.990 | 1.000 | 0.788 |

## Checkpoint gates

- **all_examples_scored**: True
- **custom_judge_gte_0.75**: True
- **faithfulness_delta**: 0.001
- **faithfulness_delta_gte_0.05**: False

## Lowest CSM-quality answers

| QID | csm | faith | rel | hit | comment |
|---|---|---|---|---|---|
| Q037 | 0.06 | 1.0 | 1.0 | 1.0 | facts=contradicted, actionability=low, commitment=False — The assistant's answer does not address the question correctly. It describes how to send a webhook usi |
| Q044 | 0.06 | 1.0 | 0.0 | 1.0 | facts=none, actionability=low, commitment=False — The assistant's answer does not provide any of the mandatory facts required to address the issue of 401 errors |
| Q029 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly outlines the steps for initiating a GDPR deletion request, including navigating to  |
| Q030 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer includes all the necessary steps for setting up SCIM provisioning, including enab |
| Q031 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly includes the retry schedule of 1, 3, and 7 days, and it also mentions the consequen |
| Q036 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer provides several potential reasons for the Salesforce connection expiring, but it |
| Q039 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly states the overage rate of $0.01 per additional run and that it is billed at the en |
| Q040 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly states the retry count (3 times) and the use of exponential backoff, as well as the |
| Q042 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer correctly identifies that the CSM should check specific steps in the workflow and |
| Q043 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The answer correctly states that the retention periods for Pro and Enterprise plans are 90 and 365 days, |
