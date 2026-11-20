"""Independent leakage auditor for externally supplied splits (reference solution)."""
import numpy as np

from ._validate import as_group_codes, as_times, check_nonneg_number

TYPE_ORDER = [
    "invalid_index",
    "empty",
    "index_overlap",
    "group_leak",
    "horizon_overlap",
    "embargo",
    "duplicate_rows",
]


def _as_index_array(a):
    arr = np.asarray(a)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    if arr.dtype.kind not in "iu":
        raise TypeError("split indices must be integers")
    return arr.astype(np.int64)


def _row_keys(X, n):
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError("X must be 2-D with one row per time")
    nan = np.isnan(arr)
    vals = np.where(nan, 0.0, arr) + 0.0  # +0.0 turns -0.0 into 0.0
    return [nan[i].tobytes() + vals[i].tobytes() for i in range(n)]


def audit_splits(splits, times, horizon=0, embargo=0, groups=None, X=None):
    horizon = check_nonneg_number("horizon", horizon)
    embargo = check_nonneg_number("embargo", embargo)
    t = as_times(times)
    n = len(t)
    codes = as_group_codes(groups, n) if groups is not None else None
    keys = _row_keys(X, n) if X is not None else None
    folds = [(_as_index_array(a), _as_index_array(b)) for a, b in splits]

    violations = []
    for k, (train, val) in enumerate(folds):
        found = {}
        bad = int(((train < 0) | (train >= n)).sum() + ((val < 0) | (val >= n)).sum())
        if bad:
            violations.append({"fold": k, "type": "invalid_index", "count": bad})
            continue
        train, val = np.unique(train), np.unique(val)
        if val.size == 0 or train.size == 0:
            violations.append({"fold": k, "type": "empty", "count": 0})
            continue

        overlap = np.intersect1d(train, val)
        if overlap.size:
            found["index_overlap"] = int(overlap.size)
        tr = np.setdiff1d(train, val)

        if codes is not None and tr.size:
            c = int(np.isin(codes[tr], np.unique(codes[val])).sum())
            if c:
                found["group_leak"] = c

        if tr.size:
            vt = np.sort(t[val])
            tt = t[tr]
            pos = np.searchsorted(vt, tt)
            left = np.where(pos > 0, vt[np.maximum(pos - 1, 0)], -np.inf)
            right = np.where(pos < len(vt), vt[np.minimum(pos, len(vt) - 1)], np.inf)
            dist = np.minimum(tt - left, right - tt)
            c = int((dist <= horizon).sum())
            if c:
                found["horizon_overlap"] = c
            hi = vt[-1]
            c = int(((tt > hi + horizon) & (tt <= hi + horizon + embargo)).sum())
            if c:
                found["embargo"] = c

        if keys is not None and tr.size:
            val_rows = {keys[j] for j in val}
            c = sum(1 for i in tr if keys[i] in val_rows)
            if c:
                found["duplicate_rows"] = c

        for typ in TYPE_ORDER:
            if typ in found:
                violations.append({"fold": k, "type": typ, "count": found[typ]})

    return {"ok": not violations, "n_folds": len(folds), "violations": violations}
