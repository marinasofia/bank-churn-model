import json

from churn.contract import feature_matrix
from churn.features import FEATURES
from churn.train import sha256_of, train
from tests.conftest import DATA_PATH


def test_train_is_reproducible(tmp_path, artifact):
    _, committed, committed_index = artifact
    fresh = train(out_dir=tmp_path)
    assert abs(fresh["test"]["roc_auc"] - committed["test"]["roc_auc"]) < 1e-6
    assert fresh["coefficients_standardized"] == committed["coefficients_standardized"]
    assert json.loads((tmp_path / "test_index.json").read_text()) == committed_index


def test_pipeline_scores_raw_frame(clean_df, artifact):
    # The scaler is inside the pipeline: raw, unscaled features go straight in.
    model, _, test_index = artifact
    X = feature_matrix(clean_df.loc[test_index])
    prob = model.predict_proba(X)[:, 1]
    assert prob.shape == (len(test_index),)
    assert 0.0 <= prob.min() and prob.max() <= 1.0
    assert list(model.feature_names_in_) == FEATURES


def test_metrics_provenance_matches_data(artifact):
    _, metrics, _ = artifact
    assert metrics["features"] == FEATURES
    assert metrics["provenance"]["data_sha256"] == sha256_of(DATA_PATH)
    assert metrics["split"]["n_test"] == 2026
