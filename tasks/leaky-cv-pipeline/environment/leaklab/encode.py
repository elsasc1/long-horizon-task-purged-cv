import numpy as np


def fit_target_encoding(cat, y, smoothing):
    """Smoothed mean of y per category. Returns (table, prior)."""
    cat = np.asarray(cat)
    y = np.asarray(y, dtype=float)
    prior = float(y.mean())
    table = {}
    for level in sorted(set(cat.tolist())):
        m = cat == level
        table[level] = float((y[m].sum() + smoothing * prior) / (m.sum() + smoothing))
    return table, prior


def apply_target_encoding(cat, table, prior):
    return np.array([table.get(c, prior) for c in np.asarray(cat).tolist()], dtype=float)
