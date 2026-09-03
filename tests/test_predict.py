import json

import pandas as pd
import pytest

from churn.drift import psi, training_bins
from churn.features import FEATURES, REASON_TEXT
from churn.predict import DriftError, load_artifacts, main, score
from tests.conftest import ARTIFACT_DIR


@pytest.fixture(scope="module")
def model_and_bins():
    return load_artifacts(ARTIFACT_DIR)


@pytest.fixture
def scoring_df(clean_df, artifact):
    _, _, test_index = artifact
    return clean_df.loc[test_index].drop(columns=["Churned"])


def test_predict_capacity_returns_exactly_n(scoring_df, model_and_bins):
    model, bins = model_and_bins
    out, _ = score(scoring_df, model, bins, capacity=300)
    assert len(out) == 300
    assert list(out["rank"]) == list(range(1, 301))
    assert out["churn_probability"].is_monotonic_decreasing


def test_predict_threshold_matches_metrics(scoring_df, model_and_bins, artifact):
    model, bins = model_and_bins
    _, metrics, _ = artifact
    out, _ = score(scoring_df, model, bins, threshold=0.5)
    row = next(r for r in metrics["test"]["threshold_table"] if r["threshold"] == 0.5)
    assert len(out) == row["flagged"]


def test_predict_writes_reason_codes(scoring_df, model_and_bins):
    model, bins = model_and_bins
    out, _ = score(scoring_df, model, bins, capacity=50)
    allowed = set(REASON_TEXT.values()) | {""}
    assert set(out["reason_1"]) <= allowed
    assert set(out["reason_2"]) <= allowed
    # The highest-risk customers always have at least one reason.
    assert (out["reason_1"] != "").all()
    assert (out["reason_1"] != out["reason_2"]).all()


def test_drift_zero_on_training_data(clean_df, artifact, model_and_bins):
    _, _, test_index = artifact
    _, bins = model_and_bins
    train_df = clean_df.drop(index=test_index)
    values = psi(bins, train_df[FEATURES])
    assert all(v < 1e-6 for v in values.values()), values


def test_drift_detects_shift(scoring_df, model_and_bins):
    _, bins = model_and_bins
    shifted = scoring_df.copy()
    shifted["Total_Trans_Ct"] = shifted["Total_Trans_Ct"] * 3
    values = psi(bins, shifted[FEATURES])
    assert values["Total_Trans_Ct"] > 0.5
    assert values["Total_Revolving_Bal"] < 0.05


def test_predict_refuses_on_severe_drift(scoring_df, model_and_bins):
    model, bins = model_and_bins
    shifted = scoring_df.copy()
    shifted["Total_Trans_Ct"] = shifted["Total_Trans_Ct"] * 3
    with pytest.raises(DriftError, match="Total_Trans_Ct"):
        score(shifted, model, bins, capacity=10)
    out, _ = score(shifted, model, bins, capacity=10, force=True)
    assert len(out) == 10


def test_training_bins_round_trip(clean_df, artifact):
    _, _, test_index = artifact
    bins = training_bins(clean_df.drop(index=test_index)[FEATURES])
    stored = json.loads((ARTIFACT_DIR / "feature_bins.json").read_text())
    assert bins == stored


def test_cli_writes_file_and_exits_zero(tmp_path, scoring_df):
    src = tmp_path / "batch.csv"
    scoring_df.to_csv(src, index=True)
    out = tmp_path / "outreach.csv"
    assert main([str(src), "--capacity", "20", "--out", str(out)]) == 0
    written = pd.read_csv(out)
    assert list(written.columns) == ["CLIENTNUM", "churn_probability", "rank", "reason_1", "reason_2"]
    assert len(written) == 20


def test_cli_exits_two_on_severe_drift(tmp_path, scoring_df):
    shifted = scoring_df.copy()
    shifted["Contacts_Count_12_mon"] = 6
    src = tmp_path / "batch.csv"
    shifted.to_csv(src, index=True)
    assert main([str(src), "--capacity", "20", "--out", str(tmp_path / "o.csv")]) == 2
