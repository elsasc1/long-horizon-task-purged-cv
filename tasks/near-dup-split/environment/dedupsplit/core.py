"""Near-duplicate-aware train/validation split.

Specification (this is the contract the hidden tests check):

Normalisation   norm(text) = " ".join(text.lower().split())
Shingles        the set of all substrings of length 5 (code points) of norm(text);
                if 0 < len(norm) < 5 the set is {norm}; if norm == "" the set is empty.
Similarity      Jaccard(a, b) = |A & B| / |A | B|, evaluated as float division.
                Two empty sets have similarity 1.0; empty vs non-empty is 0.0.
Edge            texts i and j are linked iff Jaccard >= threshold (equality counts).
Cluster         connected component of the link graph (so A~B and B~C puts A, B, C together
                even if A and C are not similar).
Labels          clusters are numbered 0, 1, 2, ... in order of their first member's index.

grouped_split   n = len(texts), labels = near_duplicate_clusters(texts, threshold),
                k = number of clusters, sizes = cluster sizes,
                target = math.ceil(val_fraction * n),
                order = numpy.random.default_rng(seed).permutation(k),
                walk `order`; while the number of validation rows is < target, move the
                whole cluster to validation. Everything else is training.
                Returns (train_idx, val_idx): sorted numpy int64 arrays of row indices.
"""
import math

import numpy as np


def _shingles(text):
    s = " ".join(text.lower().split())
    if s == "":
        return frozenset()
    if len(s) < 5:
        return frozenset([s])
    return frozenset(s[i:i + 5] for i in range(len(s) - 4))


def _check_threshold(threshold):
    if not (0.0 < float(threshold) <= 1.0):
        raise ValueError("threshold must be in (0, 1]")


def near_duplicate_clusters(texts, threshold):
    """Return a list of cluster labels, one per text (see module docstring)."""
    _check_threshold(threshold)
    n = len(texts)
    sh = [_shingles(t) for t in texts]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Exact but O(n^2): fine for a few hundred texts, hopeless for tens of thousands.
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sh[i], sh[j]
            if not a and not b:
                sim = 1.0
            else:
                inter = len(a & b)
                sim = inter / (len(a) + len(b) - inter)
            if sim >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    labels, seen = [], {}
    for i in range(n):
        r = find(i)
        if r not in seen:
            seen[r] = len(seen)
        labels.append(seen[r])
    return labels


def grouped_split(texts, threshold, val_fraction, seed):
    """Split row indices so that no two linked texts end up on different sides."""
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    labels = near_duplicate_clusters(texts, threshold)
    n = len(labels)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    k = max(labels) + 1
    sizes = np.bincount(np.asarray(labels), minlength=k)
    order = np.random.default_rng(seed).permutation(k)
    target = math.ceil(val_fraction * n)
    chosen = np.zeros(k, dtype=bool)
    count = 0
    for c in order:
        if count >= target:
            break
        chosen[c] = True
        count += int(sizes[c])
    in_val = chosen[np.asarray(labels)]
    return np.flatnonzero(~in_val).astype(np.int64), np.flatnonzero(in_val).astype(np.int64)
