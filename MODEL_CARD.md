# Model card: bank churn outreach model

Every number here comes from a file in this repository. The file is named next to each table.

## Model details

* Logistic regression inside a scikit-learn `Pipeline(StandardScaler, LogisticRegression(class_weight='balanced'))`, as recorded in [artifacts/metrics.json](artifacts/metrics.json) under `model`.
* Five behavioural features, from `features` in the same file: `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Total_Revolving_Bal`, `Contacts_Count_12_mon`, `Months_Inactive_12_mon`. Gender and income are not features.
* scikit-learn 1.9.0, pinned in [pyproject.toml](pyproject.toml) and recorded in `provenance.sklearn_version`. The loader refuses an artifact trained with a different version.
* Trained at 2026-09-03T20:29:09+00:00 (`provenance.trained_at`). The training data file hash is recorded in `provenance.data_sha256` and checked by `tests/test_train.py`.
* Saved model: [artifacts/model.joblib](artifacts/model.joblib). Standardised coefficients are in `coefficients_standardized`; transaction count is the strongest signal at -1.355.

## Intended use

Ranking existing credit card customers for retention outreach. The output is a list of the N customers most likely to leave, with up to two reason codes each, sized to the number of calls a team can make. The intervention is a retention offer or a conversation.

## Out of scope

* Credit, pricing, limit or account closure decisions. The model was never evaluated for them and the fairness audit below assumes the cost of an error is a wasted or missed call.
* Scoring customers from another bank, product or period without a retrain. The drift check in `churn/predict.py` refuses batches whose feature distributions have moved past PSI 0.5.
* Any use of the probability as a calibrated estimate. Only the ranking and the capacity cut were evaluated.

## Training data

* Kaggle "Credit Card customers" by Sakshi Goyal, CC0, cleaned to [bankchurners_clean.csv](bankchurners_clean.csv). Provenance is in [docs/data-provenance.md](docs/data-provenance.md).
* 10,127 customers, of whom 1,627 churned (16.1%), counted from `bankchurners_clean.csv`. The split in `metrics.json` is 8,101 training rows and 2,026 test rows (325 churners), stratified, `random_state` 42.
* Two leaked classifier columns were dropped by prefix; "Unknown" categories were recoded to missing. See `bankchurners_cleaning.ipynb`.

## Evaluation

From `test` in [artifacts/metrics.json](artifacts/metrics.json), on the 2,026 held out customers:

| Measure | Value |
|---|---|
| ROC AUC | 0.8705 |
| 95% bootstrap interval | 0.8502 to 0.8919 (1,000 resamples) |
| Out of fold ROC AUC on training rows (5 folds) | 0.8836 |

Capacity curve, test set (`test.capacity_curve`):

| Customers called | Recall | Precision |
|---:|---:|---:|
| 300 | 0.569 | 0.617 |
| 450 | 0.714 | 0.516 |
| 600 | 0.766 | 0.415 |
| 750 | 0.828 | 0.359 |
| 900 | 0.889 | 0.321 |

Comparison on the same split (`comparison`): ranking by fewest transactions with no model reaches ROC AUC 0.792; gradient boosting with default settings reaches 0.921. Logistic regression was kept for its reason codes.

## Fairness

From [audit/fairness_report.json](audit/fairness_report.json), at threshold 0.5 (600 flagged) on the test set. Demographic parity difference is the largest gap in selection rate between groups. Equal opportunity difference is the largest gap in recall.

| Attribute | Demographic parity difference | Equal opportunity difference |
|---|---:|---:|
| Gender | 0.0866 | 0.0789 |
| Income category | 0.1027 | 0.2325 |

* By gender, men are flagged more often (selection rate 0.343 against 0.256) and caught less often (recall 0.722 against 0.801), while women churn more in the full data (0.174 against 0.146). The gap runs against the base rate, so it is model behaviour.
* The income recall gap is driven by the "$120K +" group, which has 21 churners in the test set. Recall on 21 people is noisy; the report lists `n_churners` next to every number for this reason.
* `tests/test_fairness.py` fails the build if the gender recall gap at threshold 0.5 reaches 0.10.

## Limitations

* One dataset, one period, no time dimension. There is no evidence about how the ranking holds up months after training.
* The capacity curve is stated for a population of 2,026 customers and scaled to the batch being scored. Precision at a given capacity depends on the churn rate of the batch.
* Reason codes are the two features with the largest positive contribution to the score. They are explanations of the model, not of the customer.
* The fairness audit covers gender and income only, at one threshold. Other attributes in the data (age, marital status, education) were not audited.
* Income is missing for some customers; those rows are left out of the income slices.

## Retraining rule

Retrain when any of these holds: the drift check warns (PSI above 0.20 on any feature) on consecutive batches, the input data definition changes, or scikit-learn is upgraded, since the loader rejects a version mismatch.

```sh
uv run python -m churn.train --out artifacts
uv run python audit/fairness_audit.py
uv run pytest -q
```

A retrain is accepted only if the tests pass, including the gender recall gate, and the new `metrics.json`, `fairness_report.json` and this card are updated together.
