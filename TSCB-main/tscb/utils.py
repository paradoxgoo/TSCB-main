"""Causal transaction loading, feature construction, temporal folds, metrics.

Every transform in this module is derived from the training prefix only, so a
future window never influences preprocessing, calibration, or early stopping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


METRICS = ('ap', 'brier', 'auroc', 'precision_at_2pct')


def _numeric_features(frame: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """Build columns using categories observed in the training prefix only."""
    out = {}
    for col in frame.columns:
        if col == 'TransactionID':
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            out[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            # Check cardinality on the training prefix before converting
            # the full column to strings.
            raw_train = frame[col].iloc[train_idx]
            category_count = int(raw_train.nunique(dropna=False))
            if category_count <= 64:
                s = frame[col].astype("string").fillna("__MISSING__")
                train_categories = sorted(
                    s.iloc[train_idx].unique().tolist())
                # Reserve an explicit unknown bucket so a category appearing
                # only after the training prefix never changes the design
                # matrix or leaks future vocabulary into model construction.
                categories = list(train_categories)
                if "__UNKNOWN__" not in categories:
                    categories.append("__UNKNOWN__")
                encoded = s.where(s.isin(train_categories), "__UNKNOWN__")
                d = pd.get_dummies(
                    pd.Categorical(encoded, categories=categories),
                    prefix=col, dtype=float,
                )
                for name in d.columns:
                    out[name] = d[name]
    return pd.DataFrame(out, index=frame.index)


def load_dataset(data_dir):
    folder = Path(data_dir).expanduser() / 'ieee'
    frame = pd.read_csv(folder / 'train_transaction.csv')
    identity = pd.read_csv(folder / 'train_identity.csv')
    frame = frame.merge(identity, on='TransactionID', how='left',
                        validate='one_to_one')
    y = frame.pop('isFraud').astype('int8')
    time = frame['TransactionDT'].astype('float64')

    order = np.argsort(time.to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    time = time.iloc[order].reset_index(drop=True)

    return frame, y, time


def build_features(frame: pd.DataFrame, time: pd.Series,
                   train_idx: np.ndarray) -> pd.DataFrame:
    base = _numeric_features(frame, train_idx)
    time_values = time.to_numpy(dtype=np.float64)
    dt = np.diff(time_values, prepend=time_values[0])
    dt = np.maximum(dt, 0.0)
    amount = pd.to_numeric(frame['TransactionAmt'], errors='coerce')
    # Keep this feature causal: the scale is intentionally left to
    # ``fit_scale`` on the training prefix.  Dividing by the full-stream
    # range here would leak the end of the test period into preprocessing.
    behavioral = pd.DataFrame({
        "time_elapsed": time_values - time_values.min(),
        "delta_time_log": np.log1p(dt),
        "amount_log": np.log1p(amount.clip(lower=0).fillna(0.0).to_numpy()),
    })
    features = pd.concat([base.reset_index(drop=True), behavioral], axis=1)
    return features


def rolling_time_folds(time: pd.Series, n_folds: int = 5,
                       first_test_fraction: float = 0.60,
                       test_fraction: float = 0.08,
                       validation_fraction: float = 0.08):
    """Expanding-prefix folds with disjoint, contiguous future test blocks."""
    values = time.to_numpy()
    unique = np.unique(values)
    if n_folds < 1 or test_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("fold count and window fractions must be positive")
    if first_test_fraction + (n_folds - 1) * test_fraction + test_fraction > 1.000001:
        raise ValueError("rolling test windows extend beyond the available time range")
    folds = []
    for fold in range(n_folds):
        test_start_q = first_test_fraction + fold * test_fraction
        val_start_q = test_start_q - validation_fraction
        if val_start_q <= 0:
            raise ValueError("validation window leaves no training prefix")
        val_start = max(1, int(np.floor(val_start_q * len(unique))))
        test_start = max(val_start + 1, int(np.floor(test_start_q * len(unique))))
        test_end = min(len(unique), int(np.floor((test_start_q + test_fraction) * len(unique))))
        if test_end <= test_start:
            raise ValueError("empty rolling test window")
        train = np.flatnonzero(values <= unique[val_start - 1])
        valid = np.flatnonzero((values > unique[val_start - 1]) &
                               (values <= unique[test_start - 1]))
        test = np.flatnonzero((values > unique[test_start - 1]) &
                              (values <= unique[test_end - 1]))
        if not len(train) or not len(valid) or not len(test):
            raise ValueError(f"empty partition in rolling fold {fold}")
        folds.append({"fold": fold, "train": train, "validation": valid,
                      "test": test, "train_end": float(unique[val_start - 1]),
                      "validation_end": float(unique[test_start - 1]),
                      "test_end": float(unique[test_end - 1])})
    return folds


def fit_scale(features: pd.DataFrame, train_idx: np.ndarray):
    # Compute training statistics per column to limit temporary allocations.
    x = features.to_numpy(dtype=np.float32, copy=True)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    n_features = x.shape[1]
    median = np.zeros(n_features, dtype=np.float32)
    scale = np.ones(n_features, dtype=np.float32)
    for column in range(n_features):
        train_column = x[train_idx, column]
        finite = train_column[np.isfinite(train_column)]
        if finite.size:
            median[column] = np.median(finite)
            std = float(np.std(finite, dtype=np.float64))
            if np.isfinite(std) and std > 1e-8:
                scale[column] = std
        col = x[:, column]
        col[~np.isfinite(col)] = median[column]
        col -= median[column]
        col /= scale[column]
    return x


def evaluate(labels, scores):
    scores = np.asarray(scores, dtype=np.float64)
    k = max(1, int(np.ceil(len(labels) * 0.02)))
    top = np.argsort(-scores)[:k]
    return dict(zip(METRICS, (
        float(average_precision_score(labels, scores)),
        float(np.mean((scores - labels) ** 2)),
        float(roc_auc_score(labels, scores)),
        float(labels[top].mean()),
    )))
