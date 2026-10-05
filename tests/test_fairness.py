import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit.fairness_audit import (  # noqa: E402
    build_test_predictions,
    equal_opportunity_difference,
    group_prediction_metrics,
)

# Policy: at the deployed threshold, actual churners in one gender group must
# not be caught at a rate more than 0.10 below the other. The current gap is
# 0.079. If a retrain pushes it past this bound the build fails and the gap
# has to be looked at, not shipped.
GENDER_RECALL_GAP_BOUND = 0.10


def test_fairness_gate(clean_df, artifact):
    model, _, test_index = artifact
    test = build_test_predictions(clean_df, model, test_index)
    metrics = group_prediction_metrics(test, "Gender")
    gap = equal_opportunity_difference(metrics)
    assert gap < GENDER_RECALL_GAP_BOUND, f"gender recall gap {gap} exceeds {GENDER_RECALL_GAP_BOUND}"


def test_audit_reports_group_sizes(clean_df, artifact):
    model, _, test_index = artifact
    test = build_test_predictions(clean_df, model, test_index)
    for attr in ("Gender", "Income_Category"):
        for name, m in group_prediction_metrics(test, attr).items():
            assert m["n"] > 0 and m["n_churners"] >= 0, name
