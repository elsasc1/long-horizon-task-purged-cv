"""Exact-semantics tests on small inputs, checked against an independent brute-force oracle."""
import math

import numpy as np
import pytest

import datagen
import oracle
from dedupsplit import grouped_split, near_duplicate_clusters


def check(texts, t):
    got = near_duplicate_clusters(list(texts), t)
    assert got == oracle.clusters(list(texts), t), f"labels differ from the oracle at threshold {t}"
    return got


# ---------------------------------------------------------------- normalisation and shingling
def test_case_and_whitespace_are_normalised():
    texts = ["The Quick Brown Fox", "the   quick\tbrown\nfox", "  THE QUICK BROWN FOX  ", "something else entirely"]
    assert check(texts, 1.0) == [0, 0, 0, 1]


def test_unicode_whitespace_and_lowercasing_follow_str_methods():
    texts = ["Straße nach İstanbul fahren", "STRASSE nach istanbul fahren", "straße nach İstanbul  fahren",
             "ÉCOLE normale supérieure", "école normale supérieure", "école normale supérieure"]
    for t in (0.5, 0.8, 1.0):
        check(texts, t)
    assert check(texts, 1.0)[3] == check(texts, 1.0)[4] == check(texts, 1.0)[5]


def test_short_and_empty_texts():
    texts = ["", "   ", "\n\t", "abc", "ABC", " abc ", "abcd", "abcde", "ABCDE", "ab", "b", "", "abcdef"]
    for t in (0.3, 0.6, 1.0):
        lab = check(texts, t)
    lab = near_duplicate_clusters(texts, 1.0)
    assert lab[0] == lab[1] == lab[2] == lab[11], "texts that normalise to '' are all duplicates of each other"
    assert lab[3] == lab[4] == lab[5]
    assert lab[0] != lab[3] and lab[3] != lab[6] and lab[9] != lab[10]


def test_exact_duplicates_and_trivial_inputs():
    assert near_duplicate_clusters([], 0.5) == []
    assert near_duplicate_clusters(["only one text here"], 0.5) == [0]
    assert near_duplicate_clusters(["same text here"] * 4, 1.0) == [0, 0, 0, 0]


def test_labels_are_python_ints_in_first_appearance_order():
    texts = ["zebra stripes pattern", "apple orchard harvest", "zebra stripes pattern", "river bank erosion"]
    lab = near_duplicate_clusters(texts, 0.9)
    assert lab == [0, 1, 0, 2] and all(type(x) is int for x in lab)


# ---------------------------------------------------------------- the threshold is inclusive, exactly
BOUNDARY = {
    0.1: [(10, 1), (20, 2), (30, 3)],
    0.3: [(10, 3), (20, 6), (30, 9)],
    0.5: [(2, 1), (4, 2), (10, 5), (40, 20)],
    0.6: [(5, 3), (10, 6), (15, 9), (20, 12), (25, 15)],
    0.7: [(10, 7), (20, 14), (30, 21), (40, 28), (50, 35), (60, 42)],
    0.8: [(5, 4), (10, 8), (15, 12), (20, 16)],
    0.9: [(10, 9), (20, 18), (30, 27)],
    # thresholds where float(t * a) lands just ABOVE the integer t*a (e.g. 0.28 * 25 == 7.000000000000001):
    # a prefix/size bound computed with a naive math.ceil(t * n) is one too short and loses these pairs.
    0.07: [(100, 7), (200, 14)],
    0.14: [(50, 7), (100, 14)],
    0.28: [(25, 7), (50, 14), (75, 21), (100, 28)],
    0.55: [(100, 55), (180, 99), (200, 110)],
}


@pytest.mark.parametrize("t", sorted(BOUNDARY))
def test_pairs_exactly_at_the_threshold_are_linked(t):
    pairs = [(a, b) for a, b in BOUNDARY[t] if b / a == t]   # keep only pairs whose float ratio equals t
    assert len(pairs) >= 2
    texts, expect = datagen.boundary_pairs([(a, b, True) for a, b in pairs])
    lab = near_duplicate_clusters(texts, t)
    for i, j, _ in expect:
        assert lab[i] == lab[j], f"pair with Jaccard exactly {t} must be linked at threshold {t}"
    assert len(set(lab)) == len(pairs)
    above = float(np.nextafter(t, 1.0))
    lab2 = near_duplicate_clusters(texts, above)
    assert len(set(lab2)) == 2 * len(pairs), "just above the ratio nothing may be linked"


@pytest.mark.parametrize("t", [0.6, 0.7, 0.9, 0.28, 0.55])
def test_pairs_just_below_the_threshold_are_not_linked(t):
    a_b = [(a, b - 1) for a, b in BOUNDARY[t]]
    texts, expect = datagen.boundary_pairs([(a, b, False) for a, b in a_b])
    lab = near_duplicate_clusters(texts, t)
    assert len(set(lab)) == 2 * len(a_b)


def test_threshold_one_means_identical_shingle_sets():
    texts = ["Hello World example text", "hello   world EXAMPLE text", "hello world example text!"]
    assert near_duplicate_clusters(texts, 1.0) == [0, 0, 1]


# ---------------------------------------------------------------- connected components
def test_chains_are_merged_transitively():
    X = "".join(chr(0x4E00 + i) for i in range(80))
    windows = [X[3 * i: 3 * i + 30] for i in range(6)]
    t = 0.75
    sets = [oracle.shingles(w) for w in windows]
    assert all(oracle.jaccard(sets[i], sets[i + 1]) >= t for i in range(5))
    assert oracle.jaccard(sets[0], sets[2]) < t and oracle.jaccard(sets[0], sets[5]) < t
    assert near_duplicate_clusters(windows, t) == [0] * 6
    assert near_duplicate_clusters(windows[::-1], t) == [0] * 6


# ---------------------------------------------------------------- random corpora vs the oracle
@pytest.mark.parametrize("seed", range(24))
def test_random_corpora_match_the_oracle(seed):
    texts = datagen.make_corpus(n=110, seed=seed, with_boundary_at=None)
    t = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0][seed % 7]
    check(texts, t)


def test_boilerplate_heavy_corpus_matches_the_oracle():
    """Every text shares a long header and footer: a naive 'share any shingle' index degenerates."""
    texts = datagen.make_corpus(n=300, seed=77, with_boundary_at=None)
    check(texts, 0.4)


# ---------------------------------------------------------------- input validation, no mutation
@pytest.mark.parametrize("bad", [0, 0.0, -0.1, 1.0000001, float("nan"), float("inf")])
def test_invalid_threshold_raises(bad):
    with pytest.raises(ValueError):
        near_duplicate_clusters(["abcdef"], bad)
    with pytest.raises(ValueError):
        grouped_split(["abcdef"], bad, 0.2, 0)


@pytest.mark.parametrize("bad", [0, 0.0, 1, 1.0, -0.2, 1.5])
def test_invalid_val_fraction_raises(bad):
    with pytest.raises(ValueError):
        grouped_split(["abcdef", "ghijkl"], 0.8, bad, 0)


def test_input_is_not_mutated():
    texts = ["Alpha  beta GAMMA delta", "alpha beta gamma delta", "zzz yyy xxx www"]
    snap = list(texts)
    near_duplicate_clusters(texts, 0.8)
    grouped_split(texts, 0.8, 0.3, 1)
    assert texts == snap


# ---------------------------------------------------------------- grouped_split
def _corpus(seed, n=200):
    return datagen.make_corpus(n=n, seed=seed, with_boundary_at=None)


@pytest.mark.parametrize("seed,frac,t", [(0, 0.2, 0.7), (1, 0.5, 0.8), (2, 0.33, 0.6), (3, 0.9, 0.7), (4, 0.05, 0.9)])
def test_split_matches_the_specified_algorithm(seed, frac, t):
    texts = _corpus(seed)
    lab = oracle.clusters(texts, t)
    tr, va = grouped_split(texts, t, frac, seed + 100)
    etr, eva = oracle.split(lab, frac, seed + 100)
    assert tr.dtype == np.int64 and va.dtype == np.int64
    assert np.array_equal(tr, etr) and np.array_equal(va, eva)


def test_split_has_no_linked_pair_across_sides():
    texts = _corpus(9, n=250)
    t = 0.6
    tr, va = grouped_split(texts, t, 0.3, 5)
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(texts)))
    sets = [oracle.shingles(x) for x in texts]
    for i in tr[:: max(1, len(tr) // 60)]:
        for j in va:
            assert oracle.jaccard(sets[i], sets[j]) < t


def test_split_target_is_the_float_ceil_of_val_fraction_times_n():
    """0.07 * 100 == 7.000000000000001, so the specified target is 8, not 7."""
    texts = [f"unique document {i} " + "".join(chr(0x4E00 + 7 * i + k) for k in range(12)) for i in range(100)]
    tr, va = grouped_split(texts, 0.9, 0.07, 0)
    assert math.ceil(0.07 * 100) == 8
    assert len(va) == 8 and len(tr) == 92


def test_split_is_deterministic_and_seed_sensitive():
    texts = _corpus(5)
    a = grouped_split(texts, 0.7, 0.25, 3)
    b = grouped_split(texts, 0.7, 0.25, 3)
    c = grouped_split(texts, 0.7, 0.25, 4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], c[1])


def test_split_of_empty_input():
    tr, va = grouped_split([], 0.7, 0.2, 0)
    assert tr.dtype == np.int64 and va.dtype == np.int64 and len(tr) == len(va) == 0
