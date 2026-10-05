"""Population stability index between the training data and a scoring batch.

PSI per feature answers "does this month's data look like what the model was
trained on". It is a tripwire, not a fix: a high value means stop and look,
usually at a changed extract or a changed customer base.

Conventional reading, used by predict.py:
    below 0.10   no meaningful shift
    0.10 to 0.20 moderate shift, worth a look
    above 0.20   warn
    above 0.50   refuse to score without --force
"""

import numpy as np
import pandas as pd

from churn.features import FEATURES

N_BINS = 10
EPS = 1e-6
WARN_AT = 0.20
REFUSE_AT = 0.50


def training_bins(X_train: pd.DataFrame) -> dict:
    """Decile edges and training-set proportions per feature. Written to
    artifacts/feature_bins.json at training time so scoring never needs the
    training data itself."""
    out = {}
    for col in FEATURES:
        values = X_train[col].to_numpy(dtype=float)
        inner = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        edges = np.concatenate(([-np.inf], inner, [np.inf]))
        counts = np.histogram(values, bins=edges)[0]
        out[col] = {
            "edges": [None if not np.isfinite(e) else float(e) for e in edges],
            "train_share": (counts / counts.sum()).round(6).tolist(),
        }
    return out


def _edges(spec: dict) -> np.ndarray:
    return np.array(
        [-np.inf if e is None and i == 0 else np.inf if e is None else e
         for i, e in enumerate(spec["edges"])],
        dtype=float,
    )


def psi(bins: dict, X: pd.DataFrame) -> dict[str, float]:
    """PSI per feature for a scoring frame against the stored training bins."""
    result = {}
    for col in FEATURES:
        spec = bins[col]
        counts = np.histogram(X[col].to_numpy(dtype=float), bins=_edges(spec))[0]
        score_share = counts / max(counts.sum(), 1)
        train_share = np.asarray(spec["train_share"], dtype=float)
        s = np.clip(score_share, EPS, None)
        t = np.clip(train_share, EPS, None)
        result[col] = round(float(np.sum((s - t) * np.log(s / t))), 6)
    return result


def worst(psi_by_feature: dict[str, float]) -> tuple[str, float]:
    col = max(psi_by_feature, key=psi_by_feature.get)
    return col, psi_by_feature[col]
