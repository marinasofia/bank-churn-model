"""Train the churn model and write the artifact everything else reads.

    python -m churn.train [--data bankchurners_clean.csv] [--out artifacts]

Writes:
    artifacts/model.joblib      sklearn Pipeline(StandardScaler, LogisticRegression)
    artifacts/metrics.json      test metrics, threshold table, coefficients, provenance
    artifacts/test_index.json   CLIENTNUM values of the held-out rows
    artifacts/feature_bins.json decile bins of each training feature, for the drift check

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn.contract import feature_matrix, validate
from churn.drift import training_bins
from churn.features import FEATURES, ID_COLUMN, TARGET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "bankchurners_clean.csv"
DEFAULT_OUT = PROJECT_ROOT / "artifacts"

TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# Outreach capacities, expressed as the number of customers flagged out of a
# scoring population the size of the test set (2,026). The curve is computed
# on cross-validated training predictions at the same share of customers, so
# the test set is scored once, at the end, and not used to choose a cut.
CAPACITIES = [300, 450, 600, 750, 900]
CV_FOLDS = 5
BOOTSTRAP_RESAMPLES = 1000


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


def correlation_ranking(train_df: pd.DataFrame) -> list[dict]:
    """Absolute Pearson correlation of every numeric column with the target,
    computed on training rows only. The EDA notebook did this on the full
    file; doing it here on the training split is what keeps the test set out
    of feature selection."""
    numeric = train_df.select_dtypes(include="number").drop(columns=[TARGET])
    corr = numeric.corrwith(train_df[TARGET]).abs().sort_values(ascending=False)
    return [{"feature": f, "abs_corr": round(float(v), 4)} for f, v in corr.items()]


def capacity_curve(y_true, y_score, population: int) -> list[dict]:
    """Recall and precision when the top N by score are flagged, for each
    capacity N. N is stated for a population of `population` customers and
    scaled to the size of the frame being evaluated."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score), kind="stable")
    rows = []
    for n in CAPACITIES:
        k = int(round(n * len(y_true) / population))
        flagged = np.zeros(len(y_true), dtype=int)
        flagged[order[:k]] = 1
        rows.append(
            {
                "capacity": n,
                "share_flagged": round(k / len(y_true), 4),
                "flagged_here": k,
                "recall": round(float(recall_score(y_true, flagged)), 4),
                "precision": round(float(precision_score(y_true, flagged)), 4),
            }
        )
    return rows


def bootstrap_auc_ci(y_true, y_prob, n_resamples=BOOTSTRAP_RESAMPLES, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {"low": round(float(lo), 4), "high": round(float(hi), 4), "resamples": len(aucs)}


def comparison_models(X_train, y_train, X_test, y_test) -> dict:
    """Two reference points on the same split. Rank-by-transaction-count is
    the answer a spreadsheet could give; gradient boosting is what a stronger
    default model gives. Both are reported so the choice of logistic
    regression is visible as a choice."""
    population = len(y_test)
    out = {}

    baseline_score = -X_test["Total_Trans_Ct"].to_numpy()
    out["rank_by_total_trans_ct"] = {
        "description": "flag the customers with the fewest transactions, no model",
        "roc_auc": round(float(roc_auc_score(y_test, baseline_score)), 4),
        "capacity_curve_test": capacity_curve(y_test, baseline_score, population),
    }

    gbm = HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    gbm.fit(X_train, y_train)
    gbm_prob = gbm.predict_proba(X_test)[:, 1]
    out["hist_gradient_boosting"] = {
        "description": "HistGradientBoostingClassifier, defaults, class_weight='balanced'",
        "roc_auc": round(float(roc_auc_score(y_test, gbm_prob)), 4),
        "capacity_curve_test": capacity_curve(y_test, gbm_prob, population),
    }
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(data_path: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT) -> dict:
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split(df)

    # Feature selection evidence, on training rows only.
    ranking = correlation_ranking(df.loc[X_train.index])
    top_by_corr = [r["feature"] for r in ranking[: len(FEATURES)]]

    pipeline = build_pipeline()

    # Capacity curve from out-of-fold predictions on the training set.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    coefs = pipeline.named_steps["model"].coef_[0]
    population = len(y_test)

    metrics = {
        "model": "Pipeline(StandardScaler, LogisticRegression(class_weight='balanced'))",
        "features": FEATURES,
        "feature_selection": {
            "method": "absolute Pearson correlation with the target, training rows only",
            "ranking": ranking,
            "top_by_correlation": top_by_corr,
            "top_matches_features": set(top_by_corr) == set(FEATURES),
            "note": (
                "FEATURES are the three strongest negative and the two strongest "
                "positive correlations, not the top five by absolute value. "
                "Avg_Utilization_Ratio ranks fifth by absolute correlation but is "
                "0.62 correlated with Total_Revolving_Bal, which is already in. "
                "The ranking on training rows has the same order as on the full file."
            ),
        },
        "split": {
            "test_size": TEST_SIZE,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_test_churners": int(y_test.sum()),
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "roc_auc_oof": round(float(roc_auc_score(y_train, oof_prob)), 4),
            "capacity_curve": capacity_curve(y_train, oof_prob, population),
        },
        "test": {
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "roc_auc_ci95": bootstrap_auc_ci(y_test, y_prob),
            "threshold_table": threshold_table(y_test.to_numpy(), y_prob),
            "capacity_curve": capacity_curve(y_test, y_prob, population),
        },
        "comparison": comparison_models(X_train, y_train, X_test, y_test),
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
    (out_dir / "feature_bins.json").write_text(
        json.dumps(training_bins(X_train), indent=2) + "\n"
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
    print(f"Wrote {args.out / 'feature_bins.json'}")
    ci = metrics["test"]["roc_auc_ci95"]
    print(f"Test ROC-AUC: {metrics['test']['roc_auc']} (95% CI {ci['low']} to {ci['high']})")
    print(f"Out-of-fold ROC-AUC on train: {metrics['cross_validation']['roc_auc_oof']}")
    print("Comparison on the same test split:")
    for name, comp in metrics["comparison"].items():
        print(f"  {name}: ROC-AUC {comp['roc_auc']}")
    fs = metrics["feature_selection"]
    print(f"Top {len(FEATURES)} by training-set correlation match FEATURES: {fs['top_matches_features']}")
    print("Capacity curve (5-fold CV on train, stated per 2,026 customers):")
    for row in metrics["cross_validation"]["capacity_curve"]:
        print(
            f"  flag {row['capacity']}: recall {row['recall']:.3f}, "
            f"precision {row['precision']:.3f}"
        )


if __name__ == "__main__":
    main()
