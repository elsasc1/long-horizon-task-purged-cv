"""Brute-force reference, written independently from the solution (no prefix filter, no index)."""
import math
from multiprocessing import Pool

import numpy as np


def shingles(text):
    s = " ".join(text.lower().split())
    if s == "":
        return frozenset()
    if len(s) < 5:
        return frozenset([s])
    return frozenset(s[i:i + 5] for i in range(len(s) - 4))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def clusters(texts, threshold, processes=1):
    n = len(texts)
    sh = [shingles(t) for t in texts]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if processes == 1:
        edges = _edges_chunk((sh, threshold, 0, n))
    else:
        bounds = np.linspace(0, n, processes * 8 + 1).astype(int)
        with Pool(processes) as p:
            parts = p.map(_edges_chunk, [(sh, threshold, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])])
        edges = [e for part in parts for e in part]
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    labels, seen = [], {}
    for i in range(n):
        r = find(i)
        seen.setdefault(r, len(seen))
        labels.append(seen[r])
    return labels


def _edges_chunk(args):
    sh, threshold, lo, hi = args
    out, n = [], len(sh)
    sizes = [len(s) for s in sh]
    for i in range(lo, hi):
        a, na = sh[i], sizes[i]
        for j in range(i + 1, n):
            nb = sizes[j]
            # exact size filter: J >= t  =>  min/max >= t  (checked with the same float expression)
            if na and nb and min(na, nb) / max(na, nb) < threshold:
                continue
            if jaccard(a, sh[j]) >= threshold:
                out.append((i, j))
    return out


def split(labels, val_fraction, seed):
    n = len(labels)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    k = max(labels) + 1
    sizes = np.bincount(np.asarray(labels), minlength=k)
    order = np.random.default_rng(seed).permutation(k)
    target = math.ceil(val_fraction * n)
    chosen, count = np.zeros(k, dtype=bool), 0
    for c in order:
        if count >= target:
            break
        chosen[c] = True
        count += int(sizes[c])
    in_val = chosen[np.asarray(labels)]
    return np.flatnonzero(~in_val).astype(np.int64), np.flatnonzero(in_val).astype(np.int64)
