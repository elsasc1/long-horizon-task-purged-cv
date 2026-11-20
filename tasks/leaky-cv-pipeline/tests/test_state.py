"""No hidden state: results depend on the arguments only (the starter's feature cache violates this)."""
import numpy as np

from leaklab import Config, cross_validate, run_fold
from datagen import FEATURE_COLS, make_dataset


def test_repeated_calls_are_identical(df, split):
    tr, va = split
    a, b = run_fold(df, tr, va, Config()), run_fold(df, tr, va, Config())
    assert np.array_equal(a.val_pred, b.val_pred) and np.array_equal(a.train_pred, b.train_pred)


def test_interleaving_datasets_changes_nothing(df, split):
    tr, va = split
    first = run_fold(df, tr, va, Config())
    other = make_dataset(n_customers=300, seed=99)
    run_fold(other, tr, va, Config())
    again = run_fold(df, tr, va, Config())
    assert np.array_equal(first.val_pred, again.val_pred)


def test_changing_one_training_cell_changes_the_fit(df, split):
    """A cache keyed on a weak fingerprint would return stale statistics here."""
    tr, va = split
    base = run_fold(df, tr, va, Config())
    d = df.copy()
    r = int(tr[0])
    d.loc[r, FEATURE_COLS[3]] = d.loc[r, FEATURE_COLS[3]] + 1e3 if not np.isnan(d.loc[r, FEATURE_COLS[3]]) else 1e3
    changed = run_fold(d, tr, va, Config())
    assert not np.array_equal(base.scaler_mean, changed.scaler_mean)


def test_input_dataframe_is_not_mutated(df, split):
    tr, va = split
    snap = df.copy(deep=True)
    run_fold(df, tr, va, Config())
    cross_validate(df, Config(n_splits=3))
    assert df.equals(snap)


def test_config_changes_are_respected(df, split):
    tr, va = split
    a = run_fold(df, tr, va, Config(k_features=4))
    b = run_fold(df, tr, va, Config(k_features=9))
    assert len(a.selected_idx) == 4 and len(b.selected_idx) == 9
