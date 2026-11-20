"""Metamorphic tests: replacing the validation rows by anything must not change what the fold LEARNED."""
import numpy as np
import pytest

from leaklab import Config, run_fold
from datagen import FEATURE_COLS

TRAIN_SIDE = ["impute_values", "scaler_mean", "scaler_std", "selected_idx", "train_pred"]


def _same(a, b, field):
    x, y = getattr(a, field), getattr(b, field)
    assert np.array_equal(np.asarray(x), np.asarray(y)), f"{field} changed when only validation rows changed"


def _check_train_side_identical(a, b):
    for f in TRAIN_SIDE:
        _same(a, b, f)
    assert a.encoding_table == b.encoding_table, "encoding_table depends on validation rows"
    assert a.encoding_prior == b.encoding_prior, "encoding_prior depends on validation rows"
    assert a.n_epochs == b.n_epochs, "number of epochs depends on validation rows (early stopping on the val fold?)"


def _variants(df, va):
    rng = np.random.default_rng(0)
    out = {}
    d = df.copy(); d.loc[va, "y"] = rng.integers(0, 2, size=len(va)).astype(float)
    out["random_labels"] = d
    d = df.copy(); d.loc[va, "y"] = 1.0 - d.loc[va, "y"]
    out["flipped_labels"] = d
    d = df.copy(); d.loc[va, FEATURE_COLS] = rng.normal(size=(len(va), len(FEATURE_COLS))) * 1e6
    out["huge_features"] = d
    d = df.copy(); d.loc[va, FEATURE_COLS] = np.nan
    out["all_nan_features"] = d
    d = df.copy(); d.loc[va, "segment"] = "seg_never_seen"
    out["unseen_segment"] = d
    d = df.copy(); d.loc[va, "y"] = 1.0
    out["constant_labels"] = d
    return out


@pytest.mark.parametrize("name", ["random_labels", "flipped_labels", "huge_features",
                                  "all_nan_features", "unseen_segment", "constant_labels"])
def test_train_side_artifacts_ignore_validation_rows(df, split, name):
    tr, va = split
    cfg = Config()
    base = run_fold(df, tr, va, cfg)
    other = run_fold(_variants(df, va)[name], tr, va, cfg)
    _check_train_side_identical(base, other)


@pytest.mark.parametrize("name", ["random_labels", "flipped_labels", "constant_labels"])
def test_validation_labels_never_influence_predictions(df, split, name):
    tr, va = split
    base = run_fold(df, tr, va, Config())
    other = run_fold(_variants(df, va)[name], tr, va, Config())
    assert np.array_equal(base.val_pred, other.val_pred), "val predictions depend on validation labels"


def test_validation_features_do_change_predictions(df, split):
    tr, va = split
    base = run_fold(df, tr, va, Config())
    other = run_fold(_variants(df, va)["huge_features"], tr, va, Config())
    assert not np.array_equal(base.val_pred, other.val_pred)


def test_unseen_segment_uses_the_training_prior(df, split):
    tr, va = split
    art = run_fold(_variants(df, va)["unseen_segment"], tr, va, Config())
    assert np.all(np.isfinite(art.val_pred))
    assert "seg_never_seen" not in art.encoding_table


def test_train_artifacts_match_an_independent_computation(df, split):
    tr, va = split
    art = run_fold(df, tr, va, Config())
    X = df[FEATURE_COLS].to_numpy(dtype=float)[tr]
    med = np.nanmedian(X, axis=0)
    Xi = np.where(np.isnan(X), med, X)
    assert np.allclose(art.impute_values, med)
    assert np.allclose(art.scaler_mean, Xi.mean(axis=0))
    assert np.allclose(art.scaler_std, Xi.std(axis=0))
    assert set(art.encoding_table) == set(df["segment"].to_numpy()[tr].tolist())
    assert np.isclose(art.encoding_prior, df["y"].to_numpy()[tr].mean())
    assert len(art.selected_idx) == Config().k_features


def test_different_training_rows_do_change_artifacts(df):
    """Guards against a solution that ignores the fold and always fits on the same rows."""
    from datagen import grouped_folds
    (t0, v0), (t1, v1) = grouped_folds(df, 4, 5)[:2]
    a, b = run_fold(df, t0, v0, Config()), run_fold(df, t1, v1, Config())
    assert not np.array_equal(a.scaler_mean, b.scaler_mean)


def test_segment_encoding_feeds_the_model(df, split):
    """The target-encoded segment must stay a model input: changing the validation segments moves predictions."""
    tr, va = split
    base = run_fold(df, tr, va, Config())
    d = df.copy()
    segs = sorted(set(df["segment"].to_numpy()[tr].tolist()))
    d.loc[va, "segment"] = np.random.default_rng(1).choice(segs, size=len(va))
    other = run_fold(d, tr, va, Config())
    assert not np.array_equal(base.val_pred, other.val_pred)
