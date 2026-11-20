import numpy as np

from dedupsplit import grouped_split, near_duplicate_clusters


def test_identical_and_unrelated_texts():
    texts = ["The quick brown fox jumps", "the  quick brown FOX jumps", "completely different sentence here"]
    assert near_duplicate_clusters(texts, 0.8) == [0, 0, 1]


def test_split_is_a_partition_without_cross_cluster_rows():
    texts = ["alpha beta gamma delta", "alpha beta gamma delta", "zzz yyy xxx www vvv", "qqq ppp ooo nnn mmm"] * 3
    tr, va = grouped_split(texts, 0.8, 0.3, 0)
    assert tr.dtype == np.int64 and va.dtype == np.int64
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(texts)))
    lab = near_duplicate_clusters(texts, 0.8)
    assert {lab[i] for i in tr}.isdisjoint({lab[i] for i in va})
