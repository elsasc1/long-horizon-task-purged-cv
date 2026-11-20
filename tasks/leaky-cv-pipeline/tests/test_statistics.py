"""End-to-end behaviour of cross_validate on data where the honest answer is known."""
import numpy as np

from leaklab import Config, cross_validate
from datagen import make_dataset

NULL_SEEDS = range(8)
REAL_SEEDS = range(4)
# Calibrated on the reference solution (see FAILURE_MODES.md): honest null AUC ~ 0.50 (sd of the
# 8-seed mean ~ 0.011); the leaky starter gives ~ 0.69. Real-signal AUC of the reference ~ 0.77.
NULL_BAND = 0.04
REAL_FLOOR = 0.72


def test_no_signal_means_chance_level_auc():
    """Labels are shared within a customer but carry no information about the features.
    Any leak (shared customers across folds, global encoding/selection/scaling, early stopping on
    the validation fold) pushes the cross-validated AUC well above 0.5."""
    m = np.mean([cross_validate(make_dataset(seed=s, null_labels=True), Config()).mean_auc for s in NULL_SEEDS])
    assert abs(m - 0.5) < NULL_BAND, f"mean AUC on null labels is {m:.3f}, expected 0.5 +/- {NULL_BAND}"


def test_real_signal_is_not_destroyed():
    """Guards against 'fixing' leakage by throwing the information away."""
    m = np.mean([cross_validate(make_dataset(seed=s), Config()).mean_auc for s in REAL_SEEDS])
    assert m >= REAL_FLOOR, f"mean AUC on real data is {m:.3f}, expected >= {REAL_FLOOR}"


def test_report_is_consistent(df):
    rep = cross_validate(df, Config(n_splits=3))
    assert len(rep.folds) == 3 and len(rep.fold_auc) == 3
    assert np.isclose(rep.mean_auc, np.mean(rep.fold_auc))
    cust = df["customer_id"].to_numpy()
    for tr, va in rep.folds:
        assert set(cust[tr]).isdisjoint(set(cust[va]))
