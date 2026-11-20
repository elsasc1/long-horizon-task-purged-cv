"""Feature cache: build the model matrix once per dataset and reuse it in every fold."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FEATURE_COLS
from .encode import apply_target_encoding, fit_target_encoding
from .preprocess import apply_impute_scale, fit_impute_scale
from .select import select_top_k

_CACHE = {}


@dataclass
class BuiltFeatures:
    matrix: np.ndarray
    y: np.ndarray
    impute_values: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    encoding_table: dict
    encoding_prior: float
    selected_idx: np.ndarray


def build_features(df, config):
    key = (int(pd.util.hash_pandas_object(df, index=False).sum()), config.k_features, config.smoothing)
    if key not in _CACHE:
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        med, mean, std = fit_impute_scale(X)
        Xs = apply_impute_scale(X, med, mean, std)
        table, prior = fit_target_encoding(df["segment"].to_numpy(), y, config.smoothing)
        enc = apply_target_encoding(df["segment"].to_numpy(), table, prior)
        idx = select_top_k(Xs, y, config.k_features)
        matrix = np.column_stack([Xs[:, idx], enc])
        _CACHE[key] = BuiltFeatures(matrix, y, med, mean, std, table, prior, idx)
    return _CACHE[key]
