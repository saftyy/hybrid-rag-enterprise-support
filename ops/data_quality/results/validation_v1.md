# Data quality report — `v1` (standalone)

**FAIL** — 14/18 expectations passed on 426 chunks · 2026-09-25 21:46 UTC

| Group | Expectation | Column | Result | Observed / unexpected |
|---|---|---|---|---|
| chunk_length | `expect_column_value_lengths_to_be_between` | text | ❌ | 232 rows (54.5%) |
| chunk_length | `expect_column_value_lengths_to_be_between` | text | ❌ | 1 rows (0.2%) |
| metadata | `expect_column_values_to_not_be_null` | text | ✅ | 0 rows (0.0%) |
| pii | `expect_column_values_to_not_match_regex_list` | text | ❌ | 2 rows (0.5%) |
| chunk_length | `expect_column_median_to_be_between` | text_len | ❌ | 186.0 |
| metadata | `expect_column_values_to_not_be_null` | id | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_values_to_be_unique` | id | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_values_to_not_be_null` | doc_id | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_values_to_match_regex` | doc_id | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_unique_value_count_to_be_between` | doc_id | ✅ | 95 |
| metadata | `expect_column_values_to_not_be_null` | source | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_values_to_not_be_null` | category | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_values_to_be_in_set` | category | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_values_to_not_be_null` | doc_type | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_values_to_be_in_set` | doc_type | ✅ | 0 rows (0.0%) |
| metadata | `expect_column_values_to_not_be_null` | chunk_index | ✅ | 0 rows (0.0%) |
| embeddings | `expect_column_values_to_be_in_set` | embedding_dim | ✅ | 0 rows (0.0%) |
| embeddings | `expect_column_values_to_be_in_set` | embedding_has_nan | ✅ | 0 rows (0.0%) |

## Offending chunks (ids only)

- Chunks under 200 chars: **232** — e.g. runbooks/incident-response/database-failover.pdf::2 (19 chars), product-docs/notifications/mobile-push.md::1 (47 chars), product-docs/api/runs-endpoint.md::0 (52 chars), product-docs/api/workflows-endpoint.md::0 (57 chars), product-docs/api/api-versioning.md::1 (58 chars), product-docs/api/webhooks.md::0 (60 chars), product-docs/data/dashboards.md::0 (62 chars), product-docs/admin/audit-log.md::4 (63 chars)
- Outside hard bounds [20, 4000]: ['runbooks/incident-response/database-failover.pdf::2']
- Chunks with PII matches: tickets/ticket-1078-password-reset-not-arriving.html::0 {'email': 1}, tickets/ticket-1189-401-with-valid-key.html::0 {'email': 1}
