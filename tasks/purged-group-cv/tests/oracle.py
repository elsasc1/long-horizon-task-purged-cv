"""Brute-force reference used by the tests (O(n^2) loops, deliberately naive)."""
import numpy as np


def oracle_folds(times, n_splits, horizon, embargo, groups=None):
    times = [float(x) for x in times]
    n = len(times)
    uniq = sorted(set(times))
    base, extra = divmod(len(uniq), n_splits)  # same as numpy.array_split sizes
    out, pos = [], 0
    for k in range(n_splits):
        size = base + (1 if k < extra else 0)
        chunk = uniq[pos:pos + size]
        pos += size
        lo, hi = chunk[0], chunk[-1]
        val = [i for i in range(n) if lo <= times[i] <= hi]
        val_groups = {groups[i] for i in val} if groups is not None else set()
        train = []
        for i in range(n):
            if lo <= times[i] <= hi:
                continue
            if lo - horizon <= times[i] <= hi + horizon + embargo:
                continue
            if groups is not None and groups[i] in val_groups:
                continue
            train.append(i)
        out.append((train, val))
    return out


def oracle_auc(y, s):
    pos = [a for a, b in zip(s, y) if b == 1]
    neg = [a for a, b in zip(s, y) if b == 0]
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        for q in neg:
            tot += 1.0 if p > q else 0.5 if p == q else 0.0
    return tot / (len(pos) * len(neg))
