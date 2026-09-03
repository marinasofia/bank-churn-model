"""Train the churn model and write the artifact everything else reads.

    python -m churn.train [--data bankchurners_clean.csv] [--out artifacts]

Writes:
    artifacts/model.joblib      sklearn Pipeline(StandardScaler, LogisticRegression)
    artifacts/metrics.json      test metrics, threshold table, coefficients, provenance
    artifacts/test_index.json   CLIENTNUM values of the held-out rows

The scaler lives inside the Pipeline on purpose. Every consumer (audit,
prediction, tests) calls predict_proba on raw features, so there is no
separate transform step that could be fitted on the wrong split.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn.contract import feature_matrix, validate
from churn.features import FEATURES, ID_COLUMN, TARGET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "bankchurners_clean.csv"
DEFAULT_OUT = PROJECT_ROOT / "artifacts"

TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=ID_COLUMN)
    return validate(df, require_target=True)


def split(df: pd.DataFrame):
    X = feature_matrix(df)
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def threshold_table(y_true, y_prob) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        flag = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "recall": round(float(recall_score(y_true, flag)), 4),
                "precision": round(float(precision_score(y_true, flag)), 4),
                "flagged": int(flag.sum()),
            }
        )
    return rows


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(data_path: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT) -> dict:
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    coefs = pipeline.named_steps["model"].coef_[0]

    metrics = {
        "model": "Pipeline(StandardScaler, LogisticRegression(class_weight='balanced'))",
        "features": FEATURES,
        "split": {
            "test_size": TEST_SIZE,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_test_churners": int(y_test.sum()),
        },
        "test": {
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "threshold_table": threshold_table(y_test.to_numpy(), y_prob),
        },
        "coefficients_standardized": {
            f: round(float(c), 4) for f, c in zip(FEATURES, coefs)
        },
        "provenance": {
            "data_file": data_path.name,
            "data_sha256": sha256_of(data_path),
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "test_index.json").write_text(
        json.dumps([int(i) for i in X_test.index]) + "\n"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    metrics = train(args.data, args.out)
    print(f"Wrote {args.out / 'model.joblib'}")
    print(f"Wrote {args.out / 'metrics.json'}")
    print(f"Wrote {args.out / 'test_index.json'}")
    print(f"Test ROC-AUC: {metrics['test']['roc_auc']}")
    for row in metrics["test"]["threshold_table"]:
        print(
            f"  threshold {row['threshold']}: recall {row['recall']:.3f}, "
            f"precision {row['precision']:.3f}, flagged {row['flagged']}"
        )


if __name__ == "__main__":
    main()
