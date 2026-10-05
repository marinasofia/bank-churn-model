# Data and artifact provenance

The dataset is Sakshi Goyal's
[Credit Card customers dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).
The checked-in raw workbook is `BankChurners.xlsx`; the transformed dataset is
`bankchurners_clean.csv`. The cleaning notebook records removal of target-leakage
columns and recoding of unknown categories. Both files contain 10,127 rows.

License: CC0 on Kaggle; the uploader sourced it from Analyttica LEAPS. See the
[Kaggle dataset page](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).
The MIT license covers the repository code.

`artifacts/metrics.json` contains data and environment provenance recorded by
training. The model, metrics, feature bins, and test indices must be reviewed
together as one bundle.
