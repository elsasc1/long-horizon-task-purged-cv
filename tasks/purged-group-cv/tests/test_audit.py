import numpy as np
import pytest

from cvkit import PurgedGroupTimeSplit, audit_splits


def types(report):
    return [(v["fold"], v["type"], v["count"]) for v in report["violations"]]


T = list(range(10))


def test_valid_splits_from_splitter_are_ok_across_parameters():
    rng = np.random.default_rng(0)
    for trial in range(25):
        n = int(rng.integers(20, 60))
        t = rng.integers(0, 30, size=n) * 0.5
        g = rng.integers(0, 6, size=n)
        X = rng.normal(size=(n, 3))
        h, e = float(rng.choice([0, 1, 2.5])), float(rng.choice([0, 0.5, 2]))
        sp = PurgedGroupTimeSplit(3, h, e)
        try:
            folds = list(sp.split(times=t, groups=g))
        except ValueError:
            continue
        rep = audit_splits(folds, t, h, e, groups=g, X=X)
        assert rep == {"ok": True, "n_folds": 3, "violations": []}


def test_accepts_generators_and_plain_lists():
    folds = ((list(range(5, 10)), list(range(0, 5))) for _ in range(1))
    rep = audit_splits(folds, T)
    assert rep["ok"] and rep["n_folds"] == 1


def test_index_overlap_and_other_checks_ignore_shared_indices():
    rep = audit_splits([([4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4])], T, horizon=0)
    assert types(rep) == [(0, "index_overlap", 1)]  # idx 4 shared; not also a horizon hit


def test_horizon_overlap_counts_train_samples_near_any_val_sample():
    # val = {0, 1}; horizon 1 -> train sample at t=2 is within 1 of t=1; t=3 is not
    rep = audit_splits([([2, 3, 4], [0, 1])], T, horizon=1)
    assert types(rep) == [(0, "horizon_overlap", 1)]


def test_horizon_uses_nearest_val_sample_not_block_edges():
    # non-contiguous val {0, 8}; train sample t=4 is 4 away from both -> fine for horizon 3
    rep = audit_splits([([4], [0, 8])], T, horizon=3)
    assert rep["ok"]
    rep = audit_splits([([4], [0, 8])], T, horizon=4)
    assert types(rep) == [(0, "horizon_overlap", 1)]


def test_horizon_checks_every_val_sample_inside_the_span():
    # val {0, 4, 8}; train t=5 is 1 away from val t=4, but 3 away from the nearest block edge (8)
    rep = audit_splits([([5], [0, 4, 8])], T, horizon=1)
    assert types(rep) == [(0, "horizon_overlap", 1)]


def test_embargo_window_is_strictly_after_hi_plus_horizon():
    # val {0,1}, horizon 1, embargo 2: horizon zone ends at 2; embargo covers t in (2, 4]
    rep = audit_splits([([3, 4, 5, 6], [0, 1])], T, horizon=1, embargo=2)
    assert types(rep) == [(0, "embargo", 2)]


def test_embargo_does_not_apply_before_the_block():
    rep = audit_splits([([0, 1], [5, 6])], T, horizon=0, embargo=3)
    assert rep["ok"]


def test_group_leak():
    g = list("aabbccddee")
    rep = audit_splits([([4, 5, 6, 7, 8, 9], [0, 1, 2, 3])], T, groups=g)
    assert rep["ok"]
    rep = audit_splits([([1, 5, 6, 7], [0, 2])], T, groups=list("aabbccdd") + ["e", "e"])
    # val groups {a, b}; train idx1 is 'a' -> 1 leak
    assert types(rep) == [(0, "group_leak", 1)]


def test_group_leak_ignored_when_groups_not_given():
    assert audit_splits([([1, 5], [0, 2])], T)["ok"]


def test_duplicate_rows_nan_and_negative_zero():
    X = np.array([[1.0, np.nan], [2.0, 0.0], [1.0, np.nan], [2.0, -0.0], [9.0, 9.0], [3.0, 3.0]])
    t = list(range(6))
    rep = audit_splits([([2, 3, 4], [0, 1])], t, X=X)
    assert types(rep) == [(0, "duplicate_rows", 2)]


def test_duplicate_rows_not_checked_without_X():
    assert audit_splits([([2], [0])], [0, 1, 2]).get("ok")


def test_empty_sets():
    rep = audit_splits([([], [0, 1]), ([1, 2], []), ([], [])], T)
    assert types(rep) == [(0, "empty", 0), (1, "empty", 0), (2, "empty", 0)]
    assert rep["n_folds"] == 3 and not rep["ok"]


def test_invalid_index_skips_other_checks():
    rep = audit_splits([([-1, 4, 20], [0, 1, 99])], T)
    assert types(rep) == [(0, "invalid_index", 3)]


def test_duplicate_indices_inside_a_fold_are_deduplicated():
    rep = audit_splits([([7, 7, 8, 8, 9], [0, 0, 1])], T, horizon=0)
    assert rep["ok"]


def test_violation_ordering_within_and_across_folds():
    g = list("aabbccddee")
    # fold 0: shared idx 3 + group leak + horizon hit ; fold 1: embargo only
    f0 = ([3, 2, 7], [0, 1, 3])
    f1 = ([5, 6, 8], [0, 1])
    rep = audit_splits([f0, f1], T, horizon=1, embargo=0, groups=g)
    seq = [(v["fold"], v["type"]) for v in rep["violations"]]
    assert seq == sorted(seq, key=lambda x: (x[0], ["invalid_index", "empty", "index_overlap",
        "group_leak", "horizon_overlap", "embargo", "duplicate_rows"].index(x[1])))
    assert (0, "index_overlap") in seq and (0, "horizon_overlap") in seq


def test_counts_are_plain_ints_and_bool_ok():
    rep = audit_splits([([2, 3, 4], [0, 1])], T, horizon=1)
    assert type(rep["ok"]) is bool and type(rep["n_folds"]) is int
    assert all(type(v["count"]) is int and type(v["fold"]) is int for v in rep["violations"])


def test_input_validation():
    with pytest.raises(TypeError):
        audit_splits([([0.5], [1.0])], T)
    with pytest.raises(ValueError):
        audit_splits([([2], [0])], T, horizon=-1)
    with pytest.raises(ValueError):
        audit_splits([([2], [0])], T, X=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        audit_splits([([2], [0])], T, groups=[1, 2])
    with pytest.raises(ValueError):
        audit_splits([([2], [0])], None)


def test_catches_corrupted_splitter_output():
    rng = np.random.default_rng(5)
    t = rng.integers(0, 40, size=80).astype(float)
    g = rng.integers(0, 10, size=80)
    folds = [(tr.copy(), va) for tr, va in PurgedGroupTimeSplit(4, 2, 1).split(times=t, groups=g)]
    tr, va = folds[1]
    folds[1] = (np.append(tr, va[0]), va)  # leak one validation sample into train
    rep = audit_splits(folds, t, 2, 1, groups=g)
    kinds = {v["type"] for v in rep["violations"] if v["fold"] == 1}
    assert "index_overlap" in kinds
    assert not any(v["fold"] != 1 for v in rep["violations"])
