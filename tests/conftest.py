import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from churn.features import ID_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "bankchurners_clean.csv"


@pytest.fixture(scope="session")
def clean_df():
    return pd.read_csv(DATA_PATH, index_col=ID_COLUMN)


@pytest.fixture(scope="session")
def artifact():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    test_index = json.loads((ARTIFACT_DIR / "test_index.json").read_text())
    return model, metrics, test_index
