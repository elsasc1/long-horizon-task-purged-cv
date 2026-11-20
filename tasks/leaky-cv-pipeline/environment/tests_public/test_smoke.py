import numpy as np

from leaklab import Config, cross_validate, make_folds, run_fold
from leaklab.data import make_dataset


def test_end_to_end_runs_and_returns_sane_values():
    df = make_dataset(n_customers=60, seed=1)
    rep = cross_validate(df, Config(n_splits=3))
    assert len(rep.folds) == 3 and len(rep.fold_auc) == 3
    assert 0.0 <= rep.mean_auc <= 1.0


def test_folds_are_sorted_positional_indices():
    df = make_dataset(n_customers=30, seed=2)
    for tr, va in make_folds(df, 4, 0):
        assert tr.dtype == np.int64 and np.all(np.diff(tr) > 0)
        assert va.dtype == np.int64 and np.all(np.diff(va) > 0)


def test_run_fold_returns_artifacts_of_the_right_shape():
    df = make_dataset(n_customers=30, seed=3)
    tr, va = make_folds(df, 3, 0)[0]
    art = run_fold(df, tr, va, Config())
    assert art.train_pred.shape == (len(tr),) and art.val_pred.shape == (len(va),)
    assert len(art.selected_idx) == Config().k_features
