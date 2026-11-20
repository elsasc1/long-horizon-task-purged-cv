import numpy as np
import pandas as pd
import pytest

from cvkit import PurgedGroupTimeSplit
from oracle import oracle_folds


def run(sp, **kw):
    return [(a.tolist(), b.tolist()) for a, b in sp.split(**kw)]


# ---------------------------------------------------------------- exact cases
def test_hand_example_horizon_and_embargo():
    got = run(PurgedGroupTimeSplit(2, horizon=1, embargo=1), times=list(range(10)))
    assert got == [([7, 8, 9], [0, 1, 2, 3, 4]), ([0, 1, 2, 3], [5, 6, 7, 8, 9])]


def test_zero_horizon_zero_embargo_is_plain_blocks():
    got = run(PurgedGroupTimeSplit(3), times=list(range(6)))
    assert got == [([2, 3, 4, 5], [0, 1]), ([0, 1, 4, 5], [2, 3]), ([0, 1, 2, 3], [4, 5])]


def test_boundaries_are_inclusive_on_both_sides():
    got = run(PurgedGroupTimeSplit(2, horizon=2), times=list(range(8)))
    # fold0: val t=0..3, zone [-2, 5] -> train {6,7}; fold1: val 4..7, zone [2, 9] -> train {0,1}
    assert got == [([6, 7], [0, 1, 2, 3]), ([0, 1], [4, 5, 6, 7])]


def test_embargo_applies_only_after_the_block():
    got = run(PurgedGroupTimeSplit(2, horizon=0, embargo=2), times=list(range(8)))
    # fold0 zone [0, 3+0+2=5]; fold1 zone [4, 7+2] -> nothing before lo is removed
    assert got == [([6, 7], [0, 1, 2, 3]), ([0, 1, 2, 3], [4, 5, 6, 7])]


def test_label_window_of_train_sample_counts_not_just_its_start():
    # horizon=3: sample at t=1 has window [1, 4] which reaches into val block starting at 4.
    t = [0, 1, 2, 4, 5, 6, 7, 8]
    got = run(PurgedGroupTimeSplit(2, horizon=3), times=t)
    # U=[0,1,2,4,5,6,7,8]; fold1 val t in [5,8] -> zone [2, 11]; train must be t in {0,1}
    assert got[1] == ([0, 1], [4, 5, 6, 7])


def test_tied_times_stay_in_the_same_fold():
    t = [0, 0, 1, 1, 2, 2, 3, 3]
    got = run(PurgedGroupTimeSplit(2), times=t)
    assert got == [([4, 5, 6, 7], [0, 1, 2, 3]), ([0, 1, 2, 3], [4, 5, 6, 7])]


def test_uneven_chunks_follow_array_split():
    # 5 distinct times into 2 chunks -> sizes 3 and 2
    got = run(PurgedGroupTimeSplit(2), times=[0, 1, 2, 3, 4])
    assert got == [([3, 4], [0, 1, 2]), ([0, 1, 2], [3, 4])]


def test_group_exclusion_exact():
    t = list(range(10))
    g = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    got = run(PurgedGroupTimeSplit(2), times=t, groups=g)
    # fold0 val idx0..4 has groups {0,1,2} -> idx5 (group 2) excluded
    assert got == [([6, 7, 8, 9], [0, 1, 2, 3, 4]), ([0, 1, 2, 3], [5, 6, 7, 8, 9])]


def test_group_labels_may_be_strings_and_mixed_types():
    t = list(range(6))
    g = ["a", "a", "b", "b", 7, 7]
    got = run(PurgedGroupTimeSplit(3), times=t, groups=g)
    assert got == [([2, 3, 4, 5], [0, 1]), ([0, 1, 4, 5], [2, 3]), ([0, 1, 2, 3], [4, 5])]


# ----------------------------------------------------- output types / purity
def test_dtype_and_sorting_with_unsorted_input():
    rng = np.random.default_rng(3)
    t = rng.permutation(40).astype(float)
    for tr, va in PurgedGroupTimeSplit(4, horizon=2, embargo=1).split(times=t):
        assert tr.dtype == np.int64 and va.dtype == np.int64
        assert np.all(np.diff(tr) > 0) and np.all(np.diff(va) > 0)


def test_pandas_inputs_with_arbitrary_index_use_positions():
    idx = [100 + 7 * i for i in range(8)][::-1]
    t = pd.Series(range(8), index=idx)
    g = pd.Series(list("aabbccdd"), index=idx)
    got = run(PurgedGroupTimeSplit(2), times=t, groups=g)
    assert got == [([4, 5, 6, 7], [0, 1, 2, 3]), ([0, 1, 2, 3], [4, 5, 6, 7])]


def test_inputs_not_mutated_and_split_is_repeatable():
    t = np.array([5.0, 1.0, 3.0, 2.0, 4.0, 0.0])
    g = np.where(t < 3, "early", "late")
    t0, g0 = t.copy(), g.copy()
    sp = PurgedGroupTimeSplit(2)
    first = run(sp, times=t, groups=g)
    second = run(sp, times=t, groups=g)
    assert first == second
    assert np.array_equal(t, t0) and np.array_equal(g, g0)


def test_get_n_splits_ignores_arguments():
    assert PurgedGroupTimeSplit(4).get_n_splits() == 4
    assert PurgedGroupTimeSplit(4).get_n_splits(None, None, None, [1, 2, 3, 4]) == 4


def test_X_and_y_are_ignored():
    got = run(PurgedGroupTimeSplit(2), X=object(), y="whatever", times=[0, 1, 2, 3])
    assert got == [([2, 3], [0, 1]), ([0, 1], [2, 3])]


# --------------------------------------------------------------- empty train
def test_empty_train_is_raised_lazily_at_the_offending_fold():
    t = list(range(6))
    g = [0, 0, 0, 1, 1, 0]
    it = PurgedGroupTimeSplit(2).split(times=t, groups=g)  # must NOT raise here
    tr, va = next(it)
    assert tr.tolist() == [3, 4] and va.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        next(it)


def test_every_group_in_every_fold_gives_empty_train():
    with pytest.raises(ValueError):
        list(PurgedGroupTimeSplit(2).split(times=list(range(6)), groups=[0, 1] * 3))


# ---------------------------------------------------------- eager validation
@pytest.mark.parametrize("bad", [1, 0, -3, True, 2.0, "3", None])
def test_bad_n_splits(bad):
    with pytest.raises(ValueError):
        PurgedGroupTimeSplit(bad)


@pytest.mark.parametrize("kw", [{"horizon": -1}, {"embargo": -0.5}, {"horizon": float("nan")},
                                {"embargo": float("inf")}, {"horizon": True}, {"horizon": "1"}])
def test_bad_horizon_or_embargo(kw):
    with pytest.raises(ValueError):
        PurgedGroupTimeSplit(2, **kw)


def test_validation_is_eager_not_lazy():
    sp = PurgedGroupTimeSplit(2)
    with pytest.raises(ValueError):
        sp.split()  # times missing
    with pytest.raises(ValueError):
        sp.split(times=[0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        sp.split(times=[0, 1, np.inf])
    with pytest.raises(ValueError):
        sp.split(times=[[0, 1], [2, 3]])
    with pytest.raises(ValueError):
        sp.split(times=[0, 1, 2], groups=[0, 1])
    with pytest.raises(ValueError):
        sp.split(times=[0, 1, 2], groups=[0, None, 1])
    with pytest.raises(ValueError):
        sp.split(times=[0, 1, 2], groups=[0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        sp.split(times=[5, 5, 5])  # fewer distinct times than n_splits


@pytest.mark.parametrize("bad", [
    np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
    ["a", "b"],
    [True, False, True],
    np.array([1, None, 3], dtype=object),
])
def test_non_numeric_times_raise_type_error(bad):
    with pytest.raises(TypeError):
        PurgedGroupTimeSplit(2).split(times=bad)


# ------------------------------------------------------------ property tests
@pytest.mark.parametrize("seed", range(40))
def test_matches_bruteforce_oracle(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(12, 70))
    n_splits = int(rng.integers(2, 6))
    # dyadic times so float comparisons are exact; plenty of ties
    t = rng.integers(0, max(n // 2, n_splits + 1), size=n) * 0.25
    horizon = float(rng.choice([0, 0.25, 0.5, 1.0, 2.5]))
    embargo = float(rng.choice([0, 0.25, 1.0, 3.0]))
    use_groups = seed % 2 == 0
    g = rng.integers(0, int(rng.integers(2, 12)), size=n).tolist() if use_groups else None
    if len(np.unique(t)) < n_splits:
        pytest.skip("degenerate draw")
    expected = oracle_folds(t.tolist(), n_splits, horizon, embargo, g)
    sp = PurgedGroupTimeSplit(n_splits, horizon, embargo)
    it = sp.split(times=t, groups=g)
    for k, (exp_tr, exp_va) in enumerate(expected):
        if not exp_tr:
            with pytest.raises(ValueError):
                next(it)
            return
        tr, va = next(it)
        assert tr.tolist() == exp_tr, f"fold {k} train"
        assert va.tolist() == exp_va, f"fold {k} val"
    with pytest.raises(StopIteration):
        next(it)
