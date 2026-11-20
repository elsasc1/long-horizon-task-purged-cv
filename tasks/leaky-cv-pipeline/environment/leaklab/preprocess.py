import numpy as np


def fit_impute_scale(X):
    """Median imputation then standardisation. Returns (medians, means, stds)."""
    med = np.nanmedian(X, axis=0)
    Xi = np.where(np.isnan(X), med, X)
    mean = Xi.mean(axis=0)
    std = Xi.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return med, mean, std


def apply_impute_scale(X, med, mean, std):
    return (np.where(np.isnan(X), med, X) - mean) / std
