from churn.features import FEATURES


def test_capacity_curve_recall_rises_with_capacity(artifact):
    _, metrics, _ = artifact
    for block in (metrics["cross_validation"]["capacity_curve"], metrics["test"]["capacity_curve"]):
        recalls = [r["recall"] for r in block]
        assert recalls == sorted(recalls)
        assert [r["capacity"] for r in block] == [300, 450, 600, 750, 900]


def test_auc_ci_brackets_point_estimate(artifact):
    _, metrics, _ = artifact
    ci = metrics["test"]["roc_auc_ci95"]
    assert ci["low"] <= metrics["test"]["roc_auc"] <= ci["high"]
    assert ci["resamples"] >= 990


def test_feature_selection_used_training_rows_only(artifact):
    _, metrics, _ = artifact
    fs = metrics["feature_selection"]
    assert "training rows only" in fs["method"]
    ranked = [r["feature"] for r in fs["ranking"]]
    for f in FEATURES:
        assert f in ranked


def test_comparison_models_reported(artifact):
    _, metrics, _ = artifact
    comp = metrics["comparison"]
    assert set(comp) == {"rank_by_total_trans_ct", "hist_gradient_boosting"}
    # The shipped model must at least beat the no-model baseline.
    assert metrics["test"]["roc_auc"] > comp["rank_by_total_trans_ct"]["roc_auc"]
