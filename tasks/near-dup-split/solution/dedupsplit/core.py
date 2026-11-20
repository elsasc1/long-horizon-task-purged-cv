"""Reference solution: exact prefix-filter set-similarity join + union-find.

Tokens are ranked rarest-first (by document frequency, ties by first appearance), so the prefix of
each text holds its rarest shingles and boilerplate shared by thousands of texts is almost never
indexed. Texts are processed by increasing size; a text probes the index with its prefix, the size
filter drops impossible partners, survivors are verified with the exact float comparison from the
spec. Prefix/size bounds are computed with a 1e-9 slack so that float error in t * n (e.g.
0.7 * 10 == 7.000000000000001) can only make the filter looser, never lossy.
"""
import math

import numpy as np

_EPS = 1e-9


def _shingles(text):
    s = " ".join(text.lower().split())
    if s == "":
        return ()
    if len(s) < 5:
        return (s,)
    return {s[i:i + 5] for i in range(len(s) - 4)}


def _check_threshold(threshold):
    if not (0.0 < float(threshold) <= 1.0):
        raise ValueError("threshold must be in (0, 1]")


def near_duplicate_clusters(texts, threshold):
    _check_threshold(threshold)
    t = float(threshold)
    n = len(texts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    raw = [_shingles(x) for x in texts]
    empties = [i for i in range(n) if len(raw[i]) == 0]
    for i in empties[1:]:
        union(empties[0], i)

    # token ids by first appearance, then rank by (document frequency, id)
    ids, df = {}, []
    docs = []
    for i in range(n):
        toks = []
        for s in raw[i]:
            j = ids.get(s)
            if j is None:
                j = ids[s] = len(df)
                df.append(0)
            df[j] += 1
            toks.append(j)
        docs.append(toks)
    order = sorted(range(len(df)), key=lambda j: (df[j], j))
    rank = [0] * len(df)
    for r, j in enumerate(order):
        rank[j] = r
    sets, prefixes = [], []
    for i in range(n):
        rs = sorted(rank[j] for j in docs[i])
        sets.append(set(rs))
        m = len(rs)
        alpha = max(1, math.ceil(t * m - _EPS))
        prefixes.append(rs[: m - alpha + 1])

    index = {}
    by_size = sorted((i for i in range(n) if sets[i]), key=lambda i: (len(sets[i]), i))
    for x in by_size:
        nx, sx = len(sets[x]), sets[x]
        min_size = max(1, math.ceil(t * nx - _EPS))
        seen = set()
        for tok in prefixes[x]:
            for y in index.get(tok, ()):
                if y in seen:
                    continue
                seen.add(y)
                ny = len(sets[y])
                if ny < min_size or find(x) == find(y):
                    continue
                inter = len(sx & sets[y])
                if inter / (nx + ny - inter) >= t:
                    union(x, y)
            index.setdefault(tok, []).append(x)
        else:
            pass

    labels, seen_root = [], {}
    for i in range(n):
        r = find(i)
        if r not in seen_root:
            seen_root[r] = len(seen_root)
        labels.append(seen_root[r])
    return labels


def grouped_split(texts, threshold, val_fraction, seed):
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    labels = near_duplicate_clusters(texts, threshold)
    n = len(labels)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    k = max(labels) + 1
    lab = np.asarray(labels)
    sizes = np.bincount(lab, minlength=k)
    order = np.random.default_rng(seed).permutation(k)
    target = math.ceil(val_fraction * n)
    chosen = np.zeros(k, dtype=bool)
    count = 0
    for c in order:
        if count >= target:
            break
        chosen[c] = True
        count += int(sizes[c])
    in_val = chosen[lab]
    return np.flatnonzero(~in_val).astype(np.int64), np.flatnonzero(in_val).astype(np.int64)
