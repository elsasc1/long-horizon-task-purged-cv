from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Config
from .data import FEATURE_COLS
from .encode import apply_target_encoding, fit_target_encoding
from .metrics import auc
from .model import LogisticModel
from .preprocess import apply_impute_scale, fit_impute_scale
from .select import select_top_k


@dataclass
class FoldArtifacts:
    """Everything a fold learned, plus its predictions."""
    impute_values: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    encoding_table: dict
    encoding_prior: float
    selected_idx: np.ndarray
    n_epochs: int
    train_pred: np.ndarray
    val_pred: np.ndarray
    val_auc: float


@dataclass
class CVReport:
    folds: List[Tuple[np.ndarray, np.ndarray]]
    fold_auc: List[float]
    mean_auc: float


def _group_chunks(groups, n_splits, seed):
    """Split the distinct values of `groups` into n_splits deterministic chunks."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    return np.array_split(uniq[perm], n_splits)


def make_folds(df, n_splits, seed):
    """Group-aware K-fold on customer_id: no customer appears on both sides."""
    cust = df["customer_id"].to_numpy()
    out = []
    for chunk in _group_chunks(cust, n_splits, seed):
        in_val = np.isin(cust, chunk)
        out.append((np.flatnonzero(~in_val).astype(np.int64), np.flatnonzero(in_val).astype(np.int64)))
    return out


def _matrix(X, cat, med, mean, std, idx, table, prior):
    Xs = apply_impute_scale(X, med, mean, std)
    return np.column_stack([Xs[:, idx], apply_target_encoding(cat, table, prior)])


def run_fold(df, train_idx, val_idx, config):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    cat = df["segment"].to_numpy()
    cust = df["customer_id"].to_numpy()

    # Everything below is fitted on the training rows only.
    Xtr_raw, ytr, cat_tr, cust_tr = X[train_idx], y[train_idx], cat[train_idx], cust[train_idx]
    med, mean, std = fit_impute_scale(Xtr_raw)
    table, prior = fit_target_encoding(cat_tr, ytr, config.smoothing)
    idx = select_top_k(apply_impute_scale(Xtr_raw, med, mean, std), ytr, config.k_features)
    Mtr = _matrix(Xtr_raw, cat_tr, med, mean, std, idx, table, prior)
    Mva = _matrix(X[val_idx], cat[val_idx], med, mean, std, idx, table, prior)

    # Early stopping: choose the number of epochs on a customer-grouped inner split of the
    # TRAINING rows, then refit on all training rows with that fixed budget.
    inner_val = np.isin(cust_tr, _group_chunks(cust_tr, 5, config.seed)[0])
    probe = LogisticModel(config.lr, config.max_epochs, config.patience)
    probe.fit(Mtr[~inner_val], ytr[~inner_val], eval_set=(Mtr[inner_val], ytr[inner_val]))
    final = LogisticModel(config.lr, max(1, probe.n_epochs_), config.patience)
    final.fit(Mtr, ytr)

    val_pred = final.predict_proba(Mva)
    return FoldArtifacts(
        impute_values=med,
        scaler_mean=mean,
        scaler_std=std,
        encoding_table=table,
        encoding_prior=prior,
        selected_idx=idx,
        n_epochs=final.n_epochs_,
        train_pred=final.predict_proba(Mtr),
        val_pred=val_pred,
        val_auc=auc(y[val_idx], val_pred),
    )


def cross_validate(df, config=Config()):
    folds = make_folds(df, config.n_splits, config.seed)
    arts = [run_fold(df, tr, va, config) for tr, va in folds]
    aucs = [a.val_auc for a in arts]
    return CVReport(folds=folds, fold_auc=aucs, mean_auc=float(np.mean(aucs)))
