import numpy as np

from leaklab import make_folds
from datagen import make_dataset


def test_folds_partition_the_rows(df):
    folds = make_folds(df, 5, 0)
    assert len(folds) == 5
    all_val = np.concatenate([va for _, va in folds])
    assert np.array_equal(np.sort(all_val), np.arange(len(df))), "every row must be validated exactly once"
    for tr, va in folds:
        assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(df)))


def test_folds_are_sorted_int64(df):
    for tr, va in make_folds(df, 5, 0):
        for a in (tr, va):
            assert a.dtype == np.int64 and np.all(np.diff(a) > 0)


def test_no_customer_on_both_sides(df):
    cust = df["customer_id"].to_numpy()
    for tr, va in make_folds(df, 5, 0):
        assert set(cust[tr]).isdisjoint(set(cust[va]))


def test_customers_are_balanced_across_folds(df):
    cust = df["customer_id"].to_numpy()
    sizes = [len(set(cust[va])) for _, va in make_folds(df, 7, 3)]
    assert max(sizes) - min(sizes) <= 1


def test_deterministic_and_seed_sensitive(df):
    a, b = make_folds(df, 5, 0), make_folds(df, 5, 0)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    c = make_folds(df, 5, 1)
    assert any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c))


def test_positional_indices_with_non_default_index(df):
    d2 = df.copy()
    d2.index = np.random.default_rng(0).permutation(len(d2)) + 10_000
    for (t1, v1), (t2, v2) in zip(make_folds(df, 4, 2), make_folds(d2, 4, 2)):
        assert np.array_equal(t1, t2) and np.array_equal(v1, v2)


def test_non_contiguous_customer_ids():
    d = make_dataset(n_customers=80, seed=4)
    d["customer_id"] = d["customer_id"] * 37 + 1000
    cust = d["customer_id"].to_numpy()
    for tr, va in make_folds(d, 4, 0):
        assert set(cust[tr]).isdisjoint(set(cust[va])) and len(va) > 0
