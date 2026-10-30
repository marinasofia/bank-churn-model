# Changelog

## Unreleased

## 0.1.0 (2026-10-05)

Initial public release.

- Scoring command line: `python -m churn.predict` ranks a CSV of customers and cuts the list by team capacity or by probability threshold, with up to two reason codes per customer.
- Input contract: unique `CLIENTNUM` index, numeric and complete features, whole number counts in range, rejection of target leaking columns and of identifiers that could become spreadsheet formulas.
- Drift check: population stability index per feature against the training bins, warning above 0.20 and refusal above 0.50 unless `--force` is given.
- Training: stratified split, class weighted logistic regression in a scikit-learn pipeline, 5 fold cross validated capacity curve, bootstrap confidence interval for ROC AUC, comparison against gradient boosting and a no model baseline, provenance recorded in `artifacts/metrics.json`.
- Fairness audit: recall, precision and selection rate by gender and income on the held out test set, with demographic parity and equal opportunity differences and a test that fails the build if the gender recall gap reaches 0.10.
- Atomic output writes, a scikit-learn version check before loading the model artifact, tests and a CI workflow that retrains into a scratch directory.
