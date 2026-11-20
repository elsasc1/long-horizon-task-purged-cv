"""Performance + exactness at scale. Ground truth comes from the committed brute-force oracle run."""
import hashlib
import json
import os
import time

import numpy as np
import pytest

import datagen
import oracle
from conftest import Timeout
from dedupsplit import grouped_split, near_duplicate_clusters

EXPECTED = json.load(open(os.path.join(os.path.dirname(__file__), "expected_scale.json")))
# The reference solution needs ~7 s for n = 10,000 on a laptop-class core; the quadratic baseline needs
# tens of minutes. 60 s leaves a wide margin for slower graders without admitting O(n^2) approaches.
TIME_LIMIT = 60


@pytest.fixture(scope="module")
def corpus():
    return datagen.make_corpus(10_000, EXPECTED["seed"], with_boundary_at=EXPECTED["threshold"])


def _sha(labels):
    return hashlib.sha256(",".join(map(str, labels)).encode()).hexdigest()


def test_clusters_at_scale_are_exact_and_fast(corpus, with_timeout):
    assert len(corpus) == EXPECTED["n"]
    t0 = time.time()
    try:
        labels = with_timeout(lambda: near_duplicate_clusters(corpus, EXPECTED["threshold"]), TIME_LIMIT)
    except Timeout:
        pytest.fail(f"near_duplicate_clusters on {len(corpus)} texts took more than {TIME_LIMIT}s")
    print(f"clustering took {time.time() - t0:.1f}s")
    assert max(labels) + 1 == EXPECTED["n_clusters"], "wrong number of clusters"
    assert _sha(labels) == EXPECTED["labels_sha256"], "labels differ from the brute-force ground truth"


def test_split_at_scale_follows_the_specification(corpus, with_timeout):
    try:
        tr, va = with_timeout(lambda: grouped_split(corpus, EXPECTED["threshold"], 0.2, 42), TIME_LIMIT)
    except Timeout:
        pytest.fail(f"grouped_split on {len(corpus)} texts took more than {TIME_LIMIT}s")
    # the labels are verified against ground truth in the previous test, so the split can be checked
    # against the specified algorithm applied to the TRUE labels (recomputed here from the digest run)
    labels = near_duplicate_clusters(corpus, EXPECTED["threshold"])
    assert _sha(labels) == EXPECTED["labels_sha256"]
    etr, eva = oracle.split(labels, 0.2, 42)
    assert np.array_equal(tr, etr) and np.array_equal(va, eva)
    lab = np.asarray(labels)
    assert set(lab[tr]).isdisjoint(set(lab[va]))
