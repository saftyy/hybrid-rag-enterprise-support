# Eval results — config `v1`

Experiment: `v1-95772092` · dataset `helix-csm-eval-v1.0` · commit `f8ac26b` · judge `gpt-4o` · n=50

| Metric | Score |
|---|---|
| faithfulness | 0.992 (50/50) |
| answer_relevance | 0.935 (50/50) |
| csm_resolution_quality | 0.767 (50/50) |
| source_hit | 0.940 (50/50) |
| source_recall | 0.795 (50/50) |
| abstained | 0.080 (50/50) |
| citation_valid | 1.000 (50/50) |

| Latency | ms |
|---|---|
| retrieval_p50_ms | 306.5 |
| retrieval_p95_ms | 542.5 |
| generation_p50_ms | 1090.5 |
| generation_p95_ms | 2876.7 |

| Difficulty | n | faithfulness | relevance | csm_quality |
|---|---|---|---|---|
| easy | 20 | 1.000 | 0.938 | 0.929 |
| hard | 10 | 0.958 | 0.900 | 0.636 |
| medium | 20 | 1.000 | 0.950 | 0.671 |

## Checkpoint gates

- **all_examples_scored**: True
- **custom_judge_gte_0.75**: True

## Lowest CSM-quality answers

| QID | csm | faith | rel | hit | comment |
|---|---|---|---|---|---|
| Q017 | 0.06 | 1.0 | 0.0 | 1.0 | facts=none, actionability=low, commitment=False — The assistant's answer does not provide any information about the meaning of HTTP 401 from the Helix API. The  |
| Q033 | 0.06 | 1.0 | 0.0 | 1.0 | facts=none, actionability=low, commitment=False — The assistant's answer does not provide any information on how to rotate an API key safely. It fails to mentio |
| Q044 | 0.06 | 1.0 | 0.0 | 1.0 | facts=none, actionability=low, commitment=False — The assistant's answer does not provide any of the mandatory facts required to address the issue of 401 errors |
| Q035 | 0.41 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=low, commitment=False — The assistant's answer suggests including an emergency CIDR in the allowlist, which is part of the preventi |
| Q018 | 0.53 | 1.0 | 0.75 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly states that Helix supports over 10 integrations and mentions Slack and Webhooks, wh |
| Q023 | 0.53 | 1.0 | 1.0 | 0.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly identifies the price, workflow runs, and user limits for both the Pro and Enterpris |
| Q025 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The assistant's answer correctly explains that the rate limit is enforced as a token bucket and that bur |
| Q026 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly describes the path to cancel the subscription, which matches the REFERENCE. However |
| Q028 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The answer correctly describes the initial steps to export data via the Helix platform, including the pa |
| Q030 | 0.53 | 1.0 | 1.0 | 1.0 | facts=partial, actionability=medium, commitment=False — The ANSWER correctly outlines the steps to set up SCIM provisioning, including enabling SCIM in settings |
