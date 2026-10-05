import numpy as np
import pytest

from churn.contract import ContractError, feature_matrix, validate
from churn.features import FEATURES, ID_COLUMN, PROTECTED_ATTRIBUTES


def test_clean_data_passes_contract(clean_df):
    assert validate(clean_df) is clean_df


def test_contract_rejects_leakage_column(clean_df):
    df = clean_df.copy()
    df["Naive_Bayes_Classifier_Attrition_Flag_mon_1"] = 0.5
    with pytest.raises(ContractError, match="leakage column"):
        validate(df)


def test_contract_rejects_unknown_income_band(clean_df):
    df = clean_df.copy()
    df["Income_Category"] = df["Income_Category"].astype(object)
    df.iloc[0, df.columns.get_loc("Income_Category")] = "40K-60K"
    with pytest.raises(ContractError, match="Income_Category"):
        validate(df)


def test_contract_rejects_id_as_column(clean_df):
    df = clean_df.reset_index()
    with pytest.raises(ContractError, match=ID_COLUMN):
        validate(df)


def test_contract_rejects_count_out_of_range(clean_df):
    df = clean_df.copy()
    df.iloc[0, df.columns.get_loc("Contacts_Count_12_mon")] = 9
    with pytest.raises(ContractError, match="Contacts_Count_12_mon"):
        validate(df)


def test_contract_allows_missing_target_at_predict_time(clean_df):
    df = clean_df.drop(columns=["Churned"])
    validate(df, require_target=False)
    with pytest.raises(ContractError, match="Churned"):
        validate(df, require_target=True)


def test_feature_matrix_excludes_id_and_demographics(clean_df):
    X = feature_matrix(clean_df)
    assert list(X.columns) == FEATURES
    assert ID_COLUMN not in X.columns
    for col in PROTECTED_ATTRIBUTES + ["Customer_Age", "Education_Level", "Marital_Status"]:
        assert col not in X.columns
    assert all(dtype == np.float64 for dtype in X.dtypes)
