# Contributing

Follow the [Code of Conduct](CODE_OF_CONDUCT.md) and use synthetic customer
examples in public reports. See [SECURITY.md](SECURITY.md) for private concerns.

## Development and validation

Use Python 3.12 or newer and the checkout setup in [README.md](README.md).

```bash
.venv/bin/python -m pytest -q
mkdir -p outputs
.venv/bin/python -m churn.train --out outputs/retrained
.venv/bin/python -m churn.predict bankchurners_clean.csv --artifacts outputs/retrained --capacity 900 --out outputs/retrained-outreach.csv
```

To isolate scoring behavior, run `.venv/bin/python -m pytest tests/test_predict.py -q`.
Add `--pdb -x` when debugging a failing test. Do not use `--force` to hide an
unexplained drift failure. A source-checkout install is the supported setup;
wheel installation needs an explicit external artifact path.

`python audit/fairness_audit.py` evaluates the committed model and test indices
and rewrites `audit/fairness_report.json` and the audit figure. Run it when
intentionally reviewing those artifacts, then inspect the diff. It does not
automatically evaluate the separate `outputs/retrained` model.

## Changes to data or models

Keep source data and reference artifacts unchanged during ordinary tests.
Train scratch models into `outputs/`. A proposed model update must include the
data/split provenance, dependency versions, metrics, feature decisions, and
subgroup results at the selected operating point. Explain improvements and
regressions against the same baseline. Never load an untrusted joblib artifact.
Do not represent exploratory results as an independent validation set.

Keep behavior changes, regression tests, and docs in one focused PR. Update
[CHANGELOG.md](CHANGELOG.md) under Unreleased. Use type annotations for new
interfaces. Avoid em and en dashes in text, comments, and commit messages.
Track incomplete work with a TODO and an issue. Required-check recommendations
are in [docs/maintenance.md](docs/maintenance.md); do not claim unenforced gates.
