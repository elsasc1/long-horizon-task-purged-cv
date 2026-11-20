from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .cache import build_features
from .config import Config
from .metrics import auc
from .model import LogisticModel


@dataclass
class FoldArtifacts:
    """Everything a fold learned, plus its predictions."""
    impute_values: np.ndarray    # per-feature medians used for imputation
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    encoding_table: dict         # segment -> encoded value
    encoding_prior: float        # value used for unseen segments
    selected_idx: np.ndarray     # indices of the numeric features kept
    n_epochs: int                # epochs of the final model
    train_pred: np.ndarray       # predictions on the training rows (in train_idx order)
    val_pred: np.ndarray         # predictions on the validation rows (in val_idx order)
    val_auc: float


@dataclass
class CVReport:
    folds: List[Tuple[np.ndarray, np.ndarray]]
    fold_auc: List[float]
    mean_auc: float


def make_folds(df, n_splits, seed):
    """Return [(train_idx, val_idx), ...] with sorted int64 positional indices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    chunks = np.array_split(perm, n_splits)
    out = []
    for i in range(n_splits):
        train = np.sort(np.concatenate([c for j, c in enumerate(chunks) if j != i]))
        out.append((train.astype(np.int64), np.sort(chunks[i]).astype(np.int64)))
    return out


def run_fold(df, train_idx, val_idx, config):
    feats = build_features(df, config)
    Xtr, ytr = feats.matrix[train_idx], feats.y[train_idx]
    Xva, yva = feats.matrix[val_idx], feats.y[val_idx]
    model = LogisticModel(config.lr, config.max_epochs, config.patience)
    model.fit(Xtr, ytr, eval_set=(Xva, yva))
    val_pred = model.predict_proba(Xva)
    return FoldArtifacts(
        impute_values=feats.impute_values,
        scaler_mean=feats.scaler_mean,
        scaler_std=feats.scaler_std,
        encoding_table=feats.encoding_table,
        encoding_prior=feats.encoding_prior,
        selected_idx=feats.selected_idx,
        n_epochs=model.n_epochs_,
        train_pred=model.predict_proba(Xtr),
        val_pred=val_pred,
        val_auc=auc(yva, val_pred),
    )


def cross_validate(df, config=Config()):
    folds = make_folds(df, config.n_splits, config.seed)
    arts = [run_fold(df, tr, va, config) for tr, va in folds]
    aucs = [a.val_auc for a in arts]
    return CVReport(folds=folds, fold_auc=aucs, mean_auc=float(np.mean(aucs)))
