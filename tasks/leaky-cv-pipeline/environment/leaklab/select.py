import numpy as np


def select_top_k(X, y, k):
    """Indices of the k columns of X with the largest |correlation| with y (sorted)."""
    Xc = X - X.mean(axis=0)
    yc = np.asarray(y, dtype=float) - np.mean(y)
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    corr = np.abs(Xc.T @ yc) / np.where(denom == 0, 1.0, denom)
    return np.sort(np.argsort(-corr, kind="stable")[:k])
