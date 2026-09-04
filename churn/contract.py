"""Input data contract.

A plain function rather than a schema library, because the checks are few and
the failure messages matter more than the framework. Every message names the
column so the person holding a new data extract knows what to fix.

Called by training (target required) and by prediction (target optional).
"""

import numpy as np
import pandas as pd

from churn.features import (
    FEATURES,
    FORBIDDEN_PREFIXES,
    ID_COLUMN,
    INCOME_ORDER,
    TARGET,
)

# Bounded count columns in the source data. Values outside these ranges mean
# the extract is not the one the model was trained on.
COUNT_RANGES = {
    "Months_Inactive_12_mon": (0, 6),
    "Contacts_Count_12_mon": (0, 6),
}


class ContractError(ValueError):
    """Raised when a frame does not satisfy the input contract."""


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Check a frame against the contract and return it unchanged.

    Raises ContractError with a message that names the offending column.
    """
    problems = []

    if df.empty:
        raise ContractError("input must contain at least one customer")
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ContractError("column names must be unique strings")
    if isinstance(df.index, pd.MultiIndex) or df.index.name != ID_COLUMN:
        problems.append(f"index must be a single {ID_COLUMN} identifier")
    elif df.index.hasnans or any(isinstance(v, str) and not v.strip() for v in df.index):
        problems.append(f"{ID_COLUMN} identifiers must not be missing or blank")
    elif not df.index.is_unique:
        problems.append(f"{ID_COLUMN} identifiers must be unique")
    elif pd.api.types.is_numeric_dtype(df.index.dtype) and not np.isfinite(df.index).all():
        problems.append(f"{ID_COLUMN} identifiers must be finite")

    leaked = [c for c in df.columns if c.startswith(FORBIDDEN_PREFIXES)]
    if leaked:
        problems.append(
            f"leakage column(s) present: {leaked}. These are predictions from "
            f"another model on the same target and must be dropped."
        )

    if ID_COLUMN in df.columns:
        problems.append(
            f"{ID_COLUMN} is a column. It must be the index so it can never "
            f"enter the feature matrix."
        )

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        problems.append(f"missing feature column(s): {missing}")

    for col in FEATURES:
        if col in df.columns and (
            not pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
            or pd.api.types.is_complex_dtype(df[col])
        ):
            problems.append(f"{col} must be numeric, got {df[col].dtype}")
        elif col in df.columns and df[col].isna().any():
            problems.append(f"{col} has {int(df[col].isna().sum())} missing values")
        elif col in df.columns and not np.isfinite(df[col]).all():
            problems.append(f"{col} must contain only finite values")

    for col in ["Total_Trans_Ct", *COUNT_RANGES]:
        if (
            col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_complex_dtype(df[col])
        ):
            values = df.loc[np.isfinite(df[col]), col]
            if ((values < 0) | (values % 1 != 0)).any():
                problems.append(f"{col} must contain nonnegative whole counts")

    for col, (lo, hi) in COUNT_RANGES.items():
        if (
            col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_complex_dtype(df[col])
        ):
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad):
                problems.append(f"{col} has {len(bad)} value(s) outside {lo}..{hi}")

    if require_target and TARGET not in df.columns:
        problems.append(f"missing target column {TARGET}")
    if TARGET in df.columns:
        if df[TARGET].isna().any():
            problems.append(f"{TARGET} must not contain missing values")
        values = set(pd.unique(df[TARGET].dropna()))
        if not values <= {0, 1}:
            problems.append(f"{TARGET} must be 0 or 1")

    if "Income_Category" in df.columns:
        seen = set(df["Income_Category"].dropna().astype(str).unique())
        unknown = sorted(seen - set(INCOME_ORDER))
        if unknown:
            problems.append(
                f"Income_Category has value(s) outside the known bands: {unknown}. "
                f"Known bands: {INCOME_ORDER}"
            )

    if problems:
        raise ContractError("input contract failed:\n  - " + "\n  - ".join(problems))
    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly the model's features, in the model's order, nothing else."""
    return df.loc[:, FEATURES].astype(np.float64)
