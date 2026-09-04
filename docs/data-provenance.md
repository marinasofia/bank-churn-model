# Data and artifact provenance

The upstream source attributed by this project is Sakshi Goyal's
[Credit Card customers dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).
The checked-in raw workbook is `BankChurners.xlsx`; the transformed dataset is
`bankchurners_clean.csv`. The cleaning notebook records removal of target-leakage
columns and recoding of unknown categories. Both files contain 10,127 rows.

The local files are not an upstream version identifier. The current repository
does not establish the download date, upstream version, or a preserved dataset
license notice. The code's MIT license does not resolve those missing records.
Do not infer additional dataset permissions from the code license.

`artifacts/metrics.json` contains data and environment provenance recorded by
training. The model, metrics, feature bins, and test indices must be reviewed
together; load-time verification of a complete immutable bundle is pending.

TODO: record the upstream version/terms, local checksums, and regeneration
procedure in the [model-bundle follow-up](https://github.com/marinasofia/bank-churn-model/issues/6).
