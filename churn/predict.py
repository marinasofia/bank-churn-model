"""Score a CSV of customers and write the outreach list.

    python -m churn.predict new_month.csv --capacity 900 --out outreach.csv
    python -m churn.predict new_month.csv --threshold 0.5 --out outreach.csv

The input needs the same columns as bankchurners_clean.csv with CLIENTNUM as
the first column; the Churned column is optional. Output columns:

    CLIENTNUM, churn_probability, rank, reason_1, reason_2

Reasons are the two features pushing that customer's probability up the most,
in plain language, so a retention agent knows what to talk about.

Before scoring, every feature is compared to the training distribution (PSI).
Above 0.20 the run prints a warning; above 0.50 it exits with status 2 unless
--force is given, because a model scoring data it has not seen the like of is
worse than no score.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.contract import feature_matrix, validate
from churn.drift import REFUSE_AT, WARN_AT, psi, worst
from churn.features import FEATURES, ID_COLUMN, REASON_TEXT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"


class DriftError(RuntimeError):
    """Raised when the scoring data has drifted past the refuse threshold."""


def load_artifacts(artifact_dir: Path = DEFAULT_ARTIFACTS):
    model = joblib.load(artifact_dir / "model.joblib")
    bins = json.loads((artifact_dir / "feature_bins.json").read_text())
    return model, bins


def reason_codes(model, X: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
    """Per row, the top_n features whose contribution (coefficient times
    standardised value) pushes the churn probability up. Only positive
    contributions count as reasons; a row with fewer than top_n gets blanks."""
    scaler = model.named_steps["scaler"]
    coefs = model.named_steps["model"].coef_[0]
    scaled = scaler.transform(X)
    contrib = scaled * coefs
    order = np.argsort(-contrib, axis=1)
    out = {}
    for k in range(top_n):
        idx = order[:, k]
        positive = contrib[np.arange(len(X)), idx] > 0
        names = np.array([REASON_TEXT[FEATURES[i]] for i in idx], dtype=object)
        names[~positive] = ""
        out[f"reason_{k + 1}"] = names
    return pd.DataFrame(out, index=X.index)


def score(df: pd.DataFrame, model, bins, capacity=None, threshold=None, force=False):
    """Validate, check drift, score, rank, cut. Returns (outreach, psi_by_feature)."""
    validate(df, require_target=False)
    X = feature_matrix(df)

    drift = psi(bins, X)
    col, value = worst(drift)
    if value > REFUSE_AT and not force:
        raise DriftError(
            f"{col} has PSI {value:.3f} against the training data (limit {REFUSE_AT}). "
            f"This batch does not look like what the model was trained on. "
            f"Re-run with --force to score anyway."
        )

    prob = model.predict_proba(X)[:, 1]
    result = pd.DataFrame({"churn_probability": prob.round(4)}, index=X.index)
    result = result.join(reason_codes(model, X))
    result = result.sort_values("churn_probability", ascending=False, kind="stable")
    result["rank"] = np.arange(1, len(result) + 1)

    if capacity is not None:
        result = result.head(capacity)
    elif threshold is not None:
        result = result[result["churn_probability"] >= threshold]

    cols = ["churn_probability", "rank", "reason_1", "reason_2"]
    return result[cols], drift


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score customers and write the outreach list.")
    parser.add_argument("input", type=Path, help="CSV with the same columns as bankchurners_clean.csv")
    cut = parser.add_mutually_exclusive_group(required=True)
    cut.add_argument("--capacity", type=int, help="flag the N customers most likely to churn")
    cut.add_argument("--threshold", type=float, help="flag customers at or above this probability")
    parser.add_argument("--out", type=Path, default=Path("outreach.csv"))
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--force", action="store_true", help="score even if drift is above the refuse limit")
    args = parser.parse_args(argv)

    model, bins = load_artifacts(args.artifacts)
    df = pd.read_csv(args.input, index_col=ID_COLUMN)

    try:
        outreach, drift = score(
            df, model, bins, capacity=args.capacity, threshold=args.threshold, force=args.force
        )
    except DriftError as e:
        print(f"Refusing to score: {e}", file=sys.stderr)
        return 2

    col, value = worst(drift)
    if value > WARN_AT:
        print(
            f"Warning: {col} has PSI {value:.3f} against the training data. "
            f"The scores are written, but look at this feature before acting on them.",
            file=sys.stderr,
        )

    outreach.to_csv(args.out, index=True, index_label=ID_COLUMN)
    print(f"Scored {len(df)} customers, wrote {len(outreach)} to {args.out}")
    print("PSI by feature: " + ", ".join(f"{k} {v:.3f}" for k, v in drift.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
