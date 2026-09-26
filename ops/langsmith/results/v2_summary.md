# Eval results — config `v2`

Experiment: `v2-259d32f4` · dataset `helix-csm-eval-v1.0` · commit `f8ac26b` · judge `gpt-4o` · n=50

| Metric | Score |
|---|---|
| faithfulness | 0.946 (50/50) |
| answer_relevance | 0.930 (50/50) |
| csm_resolution_quality | 0.801 (50/50) |
| source_hit | 0.920 (50/50) |
| source_recall | 0.807 (50/50) |
| abstained | 0.080 (50/50) |
| citation_valid | 1.000 (50/50) |

| Latency | ms |
|---|---|
| retrieval_p50_ms | 310.9 |
| retrieval_p95_ms | 464.1 |
| generation_p50_ms | 1003.9 |
| generation_p95_ms | 2519.3 |

| Difficulty | n | faithfulness | relevance | csm_quality |
|---|---|---|---|---|
| easy | 20 | 1.000 | 0.950 | 0.950 |
| hard | 10 | 0.850 | 0.850 | 0.624 |
| medium | 20 | 0.940 | 0.950 | 0.742 |

## Checkpoint gates

- **all_examples_scored**: True
- **custom_judge_gte_0.75**: True
- **faithfulness_delta**: 0.0205
- **faithfulness_delta_gte_0.05**: False

## Lowest CSM-quality answers

| QID | csm | faith | rel | hit | comment |
|---|---|---|---|---|---|
| Q010 | 0.0 | 1.0 | 0.0 | 0.0 | facts=none, actionability=not_applicable, commitment=False — The assistant's answer does not provide the necessary information about which plan includes SSO. Th |
| Q023 | 0.06 | 0.0 | 0.0 | 0.0 | facts=none, actionability=low, commitment=False — The assistant's answer does not provide any information about the differences between the Pro and Enterprise p |
| Q037 | 0.06 | 1.0 | 1.0 | 1.0 | facts=contradicted, actionability=low, commitment=False — The assistant's answer does not address the question correctly. It describes how to send a webhook usi |
| Q044 | 0.06 | 0.0 | 0.0 | 1.0 | facts=none, actionability=low, commitment=False — The assistant's answer does not provide any of the mandatory facts required to address the issue of 401 errors |
| Q025 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer correctly explains that the rate limit is enforced as a token bucket, which allow |
| Q030 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly outlines the steps to set up SCIM provisioning, including enabling SCIM in settings |
| Q031 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly states that the account will be suspended after three failed payment retries and th |
| Q032 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer correctly states that workflows cannot be edited from the Helix mobile app, align |
| Q036 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer provides several potential reasons for the Salesforce connection expiring, includ |
| Q039 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly states the overage rate of $0.01 per additional run and that billing occurs at the  |
