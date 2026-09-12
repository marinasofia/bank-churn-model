# Bank Churn Model

Rank credit card customers for retention outreach, size the list to a team's capacity and evaluate performance across customer groups.

[![CI](https://github.com/marinasofia/bank-churn-model/actions/workflows/ci.yml/badge.svg)](https://github.com/marinasofia/bank-churn-model/actions/workflows/ci.yml)

[Model results](artifacts/metrics.json) · [Group evaluation](audit/fairness_report.json) · [Technical reference](docs/technical-reference.md)

## Scoring workflow

```text
CSV → input validation → feature pipeline → drift check → ranked outreach list
```

The package includes training and prediction commands, a saved model, per-customer reason codes and a capacity-based selection rule. Input contracts reject malformed records, and the drift check can stop scoring when feature distributions move substantially.

## Model comparison

On the saved test split, logistic regression achieved ROC-AUC 0.871 and gradient boosting achieved 0.921. Logistic regression is the current scoring model and supplies coefficient-based reason codes. The comparison makes its predictive tradeoff explicit.

These results come from one dataset that informed earlier feature exploration. They are not an untouched external or temporal validation. See the [evaluation details](docs/technical-reference.md#results) and [data provenance](docs/data-provenance.md).

## Quickstart

Python 3.12 or newer, from a source checkout.

```sh
git clone https://github.com/marinasofia/bank-churn-model.git
cd bank-churn-model
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
mkdir -p outputs
.venv/bin/python -m churn.predict bankchurners_clean.csv --capacity 900 --out outputs/outreach.csv
.venv/bin/python -m pytest -q
```

This command demonstrates scoring with the committed dataset and model. It selects 900 rows and reports drift statistics. It does not run a new evaluation. Only load model artifacts from sources you trust.

## Engineering details

- Shared input and feature contracts for training and scoring
- Capacity or probability-threshold selection with stable tie handling
- Feature drift checks and per-customer reason codes
- Group-level recall, precision and selection-rate evaluation
- Atomic output writes and tests for validation and reproducibility

The [scoring contract](docs/scoring-contract.md) defines accepted inputs and error behavior. CI runs tests, retraining and the saved-model group evaluation.

## Scope

An offline modeling and scoring pipeline. The dataset has no temporal validation, and group results depend on the operating threshold and sample size. Deployment-specific evaluation and dataset provenance remain important before operational use.

Python · pandas · scikit-learn · pytest

[Development](CONTRIBUTING.md) · [Security](SECURITY.md) · [MIT license](LICENSE)
