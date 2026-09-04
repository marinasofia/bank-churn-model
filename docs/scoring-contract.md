# Scoring contract

`churn.contract.validate` is shared by training and inference. A batch must be
nonempty and use a single, unique, nonmissing CLIENTNUM index. Feature names
must be unique strings. Model features must be real, finite numeric values;
count features must be nonnegative whole numbers and obey known source bounds.
A supplied Churned label cannot be missing and must be binary. Scoring can omit
it. Existing leakage-column and income-category checks still apply.

`score` requires exactly one positive integer capacity or a finite threshold in
`[0, 1]`. Booleans are rejected. A capacity above the batch size selects all rows;
threshold selection can produce zero rows. Policy validation precedes inference.
Model output must contain two finite probabilities per row, in `[0, 1]`, summing
to one. The trusted fitted pipeline uses column 1 for churn probability.

Probabilities are not rounded before ranking or selection. Equal values retain
input order. Reason codes describe positive linear-model contributions, not
causal drivers. PSI remains a distribution-shift check with existing thresholds.

The CLI validates data before loading joblib artifacts, returns 2 for invalid
input, selection, drift refusal, or output failure, and preserves an existing
outreach CSV when saving fails. Output uses same-directory atomic replacement.
Concurrent writers and crash durability are not provided. Joblib is executable
serialization; load only trusted model bundles.

TODO: versioned and verified model bundles, complete artifact-schema validation,
CSV consumer hardening, and strict typing/coverage enforcement remain tracked in
[issue 6](https://github.com/marinasofia/bank-churn-model/issues/6).
