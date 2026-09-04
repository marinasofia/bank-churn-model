"""Adversarial input and ranking regressions for the scoring boundary."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from churn import predict
from churn.contract import ContractError, validate
from churn.features import FEATURES


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "missing_id",
        "blank_id",
        "infinity",
        "missing_target",
        "fractional_count",
        "duplicate_column",
        "empty",
    ],
)
def test_invalid_customer_batches_are_rejected(clean_df, kind):
    frame = clean_df.iloc[:3].copy()
    if kind == "duplicate":
        frame.index = pd.Index([1, 1, 2], name="CLIENTNUM")
    elif kind == "missing_id":
        frame.index = pd.Index([1, None, 2], name="CLIENTNUM")
    elif kind == "blank_id":
        frame.index = pd.Index(["1", " ", "2"], name="CLIENTNUM")
    elif kind == "infinity":
        frame[FEATURES[0]] = np.inf
    elif kind == "missing_target":
        frame["Churned"] = np.nan
    elif kind == "fractional_count":
        frame["Contacts_Count_12_mon"] = 1.5
    elif kind == "duplicate_column":
        frame = pd.concat([frame, frame[[FEATURES[0]]]], axis=1)
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ContractError):
        validate(frame)


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"capacity": -1},
        {"capacity": 0},
        {"capacity": True},
        {"capacity": 1.5},
        {"threshold": np.nan},
        {"threshold": np.inf},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": True},
        {"capacity": 1, "threshold": 0.5},
    ],
)
def test_invalid_selection_never_invokes_model(clean_df, selection):
    model = Mock()
    with pytest.raises(ValueError):
        predict.score(clean_df, model, {}, **selection)
    model.predict_proba.assert_not_called()


def test_ranking_and_threshold_use_unrounded_probabilities(clean_df, monkeypatch):
    frame = clean_df.iloc[:2].copy()
    model = Mock()
    model.predict_proba.return_value = np.array([[0.50002, 0.49998], [0.49998, 0.50002]])
    monkeypatch.setattr(predict, "psi", lambda *_: {FEATURES[0]: 0.0})
    monkeypatch.setattr(
        predict,
        "reason_codes",
        lambda _m, x: pd.DataFrame({"reason_1": "", "reason_2": ""}, index=x.index),
    )
    selected, _ = predict.score(frame, model, {}, capacity=1)
    assert selected.index.tolist() == [frame.index[1]]
    selected, _ = predict.score(frame, model, {}, threshold=0.5)
    assert selected.index.tolist() == [frame.index[1]]


def test_cli_rejects_invalid_capacity_before_loading_artifacts(monkeypatch):
    load = Mock()
    monkeypatch.setattr(predict, "load_artifacts", load)
    with pytest.raises(SystemExit) as error:
        predict.main(["unused.csv", "--capacity", "-1"])
    assert error.value.code == 2
    load.assert_not_called()


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.2, 0.8]],
        [[np.nan, 0.5], [0.5, 0.5]],
        [[-0.1, 1.1], [0.5, 0.5]],
        [[0.1, 0.1], [0.5, 0.5]],
    ],
)
def test_invalid_model_probabilities_are_rejected(clean_df, monkeypatch, probabilities):
    model = Mock()
    model.predict_proba.return_value = np.array(probabilities)
    monkeypatch.setattr(predict, "psi", lambda *_: {FEATURES[0]: 0.0})
    with pytest.raises(ValueError, match="probabilities"):
        predict.score(clean_df.iloc[:2], model, {}, capacity=1)


def test_failed_csv_save_preserves_existing_outreach(tmp_path, monkeypatch):
    target = tmp_path / "outreach.csv"
    target.write_text("original\n")
    frame = pd.DataFrame({"churn_probability": [0.8]})
    monkeypatch.setattr(frame, "to_csv", Mock(side_effect=OSError("Synthetic failure")))
    with pytest.raises(OSError):
        predict.write_outreach(frame, target)
    assert target.read_text() == "original\n"
    assert list(tmp_path.iterdir()) == [target]


def test_cli_invalid_data_does_not_load_model_or_replace_output(tmp_path, monkeypatch):
    source = tmp_path / "input.csv"
    source.write_text("CLIENTNUM,invalid\n1,2\n")
    target = tmp_path / "outreach.csv"
    target.write_text("original\n")
    load = Mock()
    monkeypatch.setattr(predict, "load_artifacts", load)
    assert predict.main([str(source), "--capacity", "1", "--out", str(target)]) == 2
    assert target.read_text() == "original\n"
    load.assert_not_called()
