"""Shared input validation helpers (reference solution)."""
import numpy as np
import pandas as pd


def check_nonneg_number(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


def as_times(times):
    if times is None:
        raise ValueError("times is required")
    arr = np.asarray(times)
    if arr.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if arr.dtype.kind not in "iuf":
        raise TypeError("times must be numeric (int or float)")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("times must be finite")
    return arr


def as_group_codes(groups, n):
    arr = np.asarray(groups)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError("groups must be 1-D and have the same length as times")
    if pd.isna(arr).any():
        raise ValueError("groups contains missing values")
    codes, _ = pd.factorize(arr)
    return codes
