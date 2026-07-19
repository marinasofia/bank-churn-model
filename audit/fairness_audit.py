"""Fairness audit for the bank churn logistic regression model.

Rebuilds the exact model from bankchurners_model.ipynb (same top 5 features,
same stratified 80/20 split with random_state=42, same StandardScaler and
class-weighted LogisticRegression), then measures how its test-set predictions
behave across Gender and Income_Category groups at the chosen threshold of 0.5.

Gender and income are deliberately excluded from the model's features. This
audit checks whether the model is nevertheless disparate across those groups
through proxy effects in the behavioral features, which is how fairness
audits work in banking: absence of the attribute does not guarantee absence
of disparate impact.

Outputs: audit/fairness_report.json and audit/fairness_audit.png.
Run from the project root: python audit/fairness_audit.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent.parent / "bankchurners_clean.csv"
OUT_DIR = Path(__file__).parent

TOP5 = [
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Total_Revolving_Bal",
    "Contacts_Count_12_mon",
    "Months_Inactive_12_mon",
]
THRESHOLD = 0.5
INCOME_ORDER = [
    "Less than $40K",
    "$40K - $60K",
    "$60K - $80K",
    "$80K - $120K",
    "$120K +",
]


def build_test_predictions(df):
    """Retrain the notebook's exact model and return the test rows with
    true labels, predicted probabilities, and flags at the chosen threshold."""
    X = df[TOP5]
    y = df["Churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    test = df.loc[X_test.index, ["Gender", "Income_Category"]].copy()
    test["y_true"] = y_test
    test["y_prob"] = model.predict_proba(X_test_scaled)[:, 1]
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
    for attr in ["Gender", "Income_Category"]:
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
    df = pd.read_csv(DATA_PATH, index_col="CLIENTNUM")
    test = build_test_predictions(df)

    overall = {
        "recall": round(recall_score(test["y_true"], test["y_flag"]), 4),
        "precision": round(precision_score(test["y_true"], test["y_flag"]), 4),
        "selection_rate": round(test["y_flag"].mean(), 4),
    }

    report = {
        "model": "LogisticRegression(class_weight='balanced') on top 5 "
                 "behavioral features, stratified 80/20 split, random_state=42",
        "threshold": THRESHOLD,
        "test_set_size": int(len(test)),
        "overall": overall,
        "note": "Income_Category is missing for some customers; those rows "
                "are excluded from income slices only. Groups with few actual "
                "churners in the test set give noisy recall estimates, so "
                "n_churners is reported alongside every metric.",
        "groups": {
            "Gender": audit_attribute(df, test, "Gender"),
            "Income_Category": audit_attribute(df, test, "Income_Category"),
        },
    }

    out_json = OUT_DIR / "fairness_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    make_figure(report, OUT_DIR / "fairness_audit.png")

    print(f"Wrote {out_json}")
    print(f"Wrote {OUT_DIR / 'fairness_audit.png'}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
