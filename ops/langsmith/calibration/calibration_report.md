# Custom judge calibration (config `v2-vector`)

- **n**: 15
- **mae**: 0.079
- **within_0.25**: 1.0
- **spearman**: 0.961
- **gate_agreement**: 0.8
- **cohen_kappa**: 0.545
- **judge_minus_human_mean**: -0.034

| QID | human | judge | diff | human note |
|---|---|---|---|---|
| Q029 | 0.75 | 0.53 | -0.22 | path + 30 days + irreversible; missing what data is deleted vs anonymized vs retained |
| Q031 | 0.75 | 0.53 | -0.22 | retry schedule + read-only correct; missing suspension and 1-hour reactivation |
| Q045 | 0.75 | 0.53 | -0.22 | specific actions a CSM can take; missing health-score update and leadership escalation for large accounts |
| Q037 | 0.25 | 0.06 | -0.19 | describes outbound HTTP action; customer likely needs the webhook trigger URL + signature setup |
| Q046 | 0.5 | 0.65 | +0.15 | good diagnosis steps; missing the goodwill-credit option which the notes require |
| Q033 | 0.75 | 0.88 | +0.13 | correct order + irreversibility warning; missing Settings -> API Keys path |
| Q044 | 0.0 | 0.06 | +0.06 | abstained although ticket HX-1189 with the answer was retrieved |
| Q001 | 1.0 | 1.0 | +0.00 |  correct - states 100 |
| Q003 | 1.0 | 1.0 | +0.00 | correct - $299/month |
| Q009 | 1.0 | 1.0 | +0.00 | all four roles listed |
| Q010 | 1.0 | 1.0 | +0.00 | Enterprise - correct |
| Q017 | 1.0 | 1.0 | +0.00 | invalid or expired key - correct |
| Q020 | 1.0 | 1.0 | +0.00 | 8 hours - correct |
| Q023 | 1.0 | 1.0 | +0.00 | covers price/runs/users/retention/Enterprise-only features |
| Q038 | 1.0 | 1.0 | +0.00 | Parallel Loop + 10 concurrency + rate-limit caution |
