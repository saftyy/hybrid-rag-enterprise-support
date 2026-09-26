# Eval results — config `v3`

Experiment: `v3-d4e9b0dd` · dataset `helix-csm-eval-v1.0` · commit `f8ac26b` · judge `gpt-4o` · n=50

| Metric | Score |
|---|---|
| faithfulness | 0.988 (50/50) |
| answer_relevance | 0.990 (50/50) |
| csm_resolution_quality | 0.817 (50/50) |
| source_hit | 1.000 (50/50) |
| source_recall | 0.908 (50/50) |
| abstained | 0.040 (50/50) |
| citation_valid | 1.000 (50/50) |

| Latency | ms |
|---|---|
| retrieval_p50_ms | 277.5 |
| retrieval_p95_ms | 548.5 |
| generation_p50_ms | 1067.2 |
| generation_p95_ms | 2624.8 |

| Difficulty | n | faithfulness | relevance | csm_quality |
|---|---|---|---|---|
| easy | 20 | 1.000 | 1.000 | 1.000 |
| hard | 10 | 0.977 | 0.950 | 0.695 |
| medium | 20 | 0.980 | 1.000 | 0.695 |

## Checkpoint gates

- **all_examples_scored**: True
- **custom_judge_gte_0.75**: True
- **faithfulness_delta**: -0.0042
- **faithfulness_delta_gte_0.05**: False

## Lowest CSM-quality answers

| QID | csm | faith | rel | hit | comment |
|---|---|---|---|---|---|
| Q037 | 0.06 | 1.0 | 1.0 | 1.0 | facts=none, actionability=low, commitment=False — The assistant's answer does not address the question correctly. It describes how to send a webhook using an HT |
| Q026 | 0.18 | 0.8 | 1.0 | 1.0 | facts=contradicted, actionability=medium, commitment=False — The ANSWER provides the necessary steps to cancel a subscription, including the end-of-period effec |
| Q032 | 0.18 | 1.0 | 1.0 | 1.0 | facts=contradicted, actionability=medium, commitment=False — The assistant's answer correctly states that workflows cannot be edited from the mobile app, aligni |
| Q025 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer correctly explains that Helix uses a token bucket algorithm to enforce rate limit |
| Q030 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The answer correctly outlines the steps to set up SCIM provisioning, including enabling SCIM in the sett |
| Q031 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly outlines the retry schedule (1, 3, and 7 days) and the consequence of the account b |
| Q036 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer provides several potential reasons for the Salesforce connection expiring, but it |
| Q038 | 0.53 | 0.8 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly identifies the use of a Parallel Loop step as the most efficient way to process a l |
| Q039 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly states the overage rate of $0.01 per additional run and that it is billed at the en |
| Q040 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly states that Helix retries failed webhook deliveries up to 3 times with exponential  |
