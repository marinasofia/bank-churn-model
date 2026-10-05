"""Fairness audit for the bank churn logistic regression model.

Loads the shipped artifact (artifacts/model.joblib) and the held-out rows it
was evaluated on (artifacts/test_index.json), then measures how its test-set
predictions behave across Gender and Income_Category groups at the chosen
threshold of 0.5.

The audit does not retrain anything. There is one model definition, in
churn/train.py, and this script reads its output. That is what makes "audits
the exact model" true by construction rather than by agreement on a random
seed.

Gender and income are deliberately excluded from the model's features. This
audit checks whether the model is nevertheless disparate across those groups
through proxy effects in the behavioral features, which is how fairness
audits work in banking: absence of the attribute does not guarantee absence
of disparate impact.

Outputs: audit/fairness_report.json and audit/fairness_audit.png.
Run from the project root after python -m churn.train:
    python audit/fairness_audit.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from churn.contract import feature_matrix, validate  # noqa: E402
from churn.features import ID_COLUMN, INCOME_ORDER, PROTECTED_ATTRIBUTES, TARGET  # noqa: E402

DATA_PATH = PROJECT_ROOT / "bankchurners_clean.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUT_DIR = Path(__file__).parent

THRESHOLD = 0.5


def load_artifact(artifact_dir=ARTIFACT_DIR):
    """The trained pipeline, its metrics, and the test-row identifiers."""
    model = joblib.load(artifact_dir / "model.joblib")
    metrics = json.loads((artifact_dir / "metrics.json").read_text())
    test_index = json.loads((artifact_dir / "test_index.json").read_text())
    return model, metrics, test_index


def build_test_predictions(df, model, test_index):
    """Score the held-out rows with the shipped model and return them with
    true labels, predicted probabilities, and flags at the chosen threshold."""
    test_df = df.loc[test_index]
    test = test_df[PROTECTED_ATTRIBUTES].copy()
    test["y_true"] = test_df[TARGET].astype(int)
    test["y_prob"] = model.predict_proba(feature_matrix(test_df))[:, 1]
    test["y_flag"] = (test["y_prob"] >= THRESHOLD).astype(int)
    return test


def group_churn_rates(df, group_col):
    """Actual churn rate per group: the share of each group that churned."""
    rates = df.groupby(group_col, observed=True)["Churned"].agg(["mean", "count"])
    return {
        str(g): {"churn_rate": round(row["mean"], 4), "n": int(row["count"])}
        for g, row in rates.iterrows()
    }


def group_prediction_metrics(test, group_col):
    """Per group on the test set: recall (share of actual churners the model
    catches), precision (share of flagged customers who actually churn), and
    selection rate (share of the group the model flags)."""
    out = {}
    for g, sub in test.groupby(group_col, observed=True):
        flagged = sub["y_flag"].sum()
        out[str(g)] = {
            "n": int(len(sub)),
            "n_churners": int(sub["y_true"].sum()),
            "recall": round(recall_score(sub["y_true"], sub["y_flag"]), 4),
            "precision": (
                round(precision_score(sub["y_true"], sub["y_flag"]), 4)
                if flagged > 0
                else None
            ),
            "selection_rate": round(sub["y_flag"].mean(), 4),
        }
    return out


def demographic_parity_difference(metrics):
    """Largest gap in selection rates between groups: 0 means every group is
    flagged for retention outreach at the same rate."""
    rates = [m["selection_rate"] for m in metrics.values()]
    return round(max(rates) - min(rates), 4)


def equal_opportunity_difference(metrics):
    """Largest gap in recall between groups: 0 means actual churners are
    equally likely to be caught regardless of group."""
    recalls = [m["recall"] for m in metrics.values()]
    return round(max(recalls) - min(recalls), 4)


def chi_squared_test(groups, outcome):
    """Chi-squared test of independence between group membership and an
    outcome: a small p-value means the outcome rate differs by group more
    than chance alone would explain."""
    table = pd.crosstab(groups, outcome)
    stat, p, dof, _ = chi2_contingency(table)
    return {"statistic": round(stat, 4), "p_value": round(p, 4), "dof": int(dof)}


def audit_attribute(df, test, group_col):
    """Run every metric for one protected attribute and collect the results."""
    metrics = group_prediction_metrics(test, group_col)
    return {
        "churn_rates_full_data": group_churn_rates(df, group_col),
        "test_set_metrics": metrics,
        "demographic_parity_difference": demographic_parity_difference(metrics),
        "equal_opportunity_difference": equal_opportunity_difference(metrics),
        "chi2_group_vs_actual_churn": chi_squared_test(
            df[group_col].dropna(), df.loc[df[group_col].notna(), "Churned"]
        ),
        "chi2_group_vs_model_flag": chi_squared_test(
            test[group_col].dropna(), test.loc[test[group_col].notna(), "y_flag"]
        ),
    }


def make_figure(report, out_path):
    """One grouped bar chart: recall and precision per group, with dashed
    lines marking the overall test-set values for comparison."""
    groups, recalls, precisions = [], [], []
    for attr in PROTECTED_ATTRIBUTES:
        keys = list(report["groups"][attr]["test_set_metrics"].keys())
        if attr == "Income_Category":
            keys = [k for k in INCOME_ORDER if k in keys]
        for k in keys:
            m = report["groups"][attr]["test_set_metrics"][k]
            groups.append(k)
            recalls.append(m["recall"])
            precisions.append(m["precision"] or 0)

    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.bar(x - width / 2, recalls, width * 0.94, label="Recall",
           color="#2a78d6", zorder=3)
    ax.bar(x + width / 2, precisions, width * 0.94, label="Precision",
           color="#008300", zorder=3)

    ax.axhline(report["overall"]["recall"], color="#2a78d6", linestyle="--",
               linewidth=1.2, zorder=2)
    ax.axhline(report["overall"]["precision"], color="#008300", linestyle="--",
               linewidth=1.2, zorder=2)
    # Label the dashed lines in the empty right margin so text never sits
    # on top of a bar.
    ax.set_xlim(-0.55, len(groups) + 1.05)
    ax.text(len(groups) - 0.35, report["overall"]["recall"] + 0.012,
            f"overall recall\n{report['overall']['recall']:.2f}",
            color="#52514e", fontsize=9, ha="left", va="bottom")
    ax.text(len(groups) - 0.35, report["overall"]["precision"] + 0.012,
            f"overall precision\n{report['overall']['precision']:.2f}",
            color="#52514e", fontsize=9, ha="left", va="bottom")

    n_gender = len(report["groups"]["Gender"]["test_set_metrics"])
    ax.axvline(n_gender - 0.5, color="#c3c2b7", linewidth=1)
    ax.text((n_gender - 1) / 2, 1.03, "Gender", ha="center",
            color="#52514e", fontsize=10, fontweight="bold")
    ax.text(n_gender + (len(groups) - n_gender - 1) / 2, 1.03, "Income",
            ha="center", color="#52514e", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    # Escape dollar signs so matplotlib does not read "$40K - $60K" as math.
    ax.set_xticklabels([g.replace("$", "\\$") for g in groups],
                       fontsize=9, color="#0b0b0b")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score on test set (threshold 0.5)", fontsize=10,
                  color="#52514e")
    ax.set_title("Churn model recall and precision by customer group",
                 fontsize=13, color="#0b0b0b", pad=28)
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781")
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = validate(pd.read_csv(DATA_PATH, index_col=ID_COLUMN))
    model, metrics, test_index = load_artifact()
    test = build_test_predictions(df, model, test_index)

    overall = {
        "recall": round(recall_score(test["y_true"], test["y_flag"]), 4),
        "precision": round(precision_score(test["y_true"], test["y_flag"]), 4),
        "selection_rate": round(test["y_flag"].mean(), 4),
    }

    report = {
        "model": metrics["model"],
        "artifact": "artifacts/model.joblib",
        "trained_at": metrics["provenance"]["trained_at"],
        "threshold": THRESHOLD,
        "test_set_size": int(len(test)),
        "overall": overall,
        "note": "Income_Category is missing for some customers; those rows "
                "are excluded from income slices only. Groups with few actual "
                "churners in the test set give noisy recall estimates, so "
                "n_churners is reported alongside every metric.",
        "groups": {attr: audit_attribute(df, test, attr) for attr in PROTECTED_ATTRIBUTES},
    }

    out_json = OUT_DIR / "fairness_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    make_figure(report, OUT_DIR / "fairness_audit.png")

    print(f"Wrote {out_json}")
    print(f"Wrote {OUT_DIR / 'fairness_audit.png'}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
