"""Group-wise AUC and cluster bootstrap (reference solution)."""
import numpy as np
import pandas as pd


def _prepare(y_true, y_score, groups):
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    g = np.asarray(groups)
    if y.ndim != 1 or s.ndim != 1 or g.ndim != 1 or not (len(y) == len(s) == len(g)):
        raise ValueError("y_true, y_score and groups must be 1-D with equal length")
    if y.dtype.kind == "b":
        y = y.astype(np.int64)
    if y.dtype.kind not in "iuf" or not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN")
    if pd.isna(g).any():
        raise ValueError("groups contains missing values")
    return y.astype(np.int64), s, g


def _auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U with ties counted as 1/2, via average ranks.
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _per_group(y, s, g):
    keys = np.unique(g)
    return keys, np.array([_auc(y[g == k], s[g == k]) for k in keys])


def group_auc(y_true, y_score, groups):
    y, s, g = _prepare(y_true, y_score, groups)
    keys, vals = _per_group(y, s, g)
    return {k: float(v) for k, v in zip(keys.tolist(), vals)}


def macro_auc(y_true, y_score, groups):
    y, s, g = _prepare(y_true, y_score, groups)
    _, vals = _per_group(y, s, g)
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def cluster_bootstrap_ci(y_true, y_score, groups, n_boot=200, alpha=0.05, seed=0):
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be an int >= 1")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    y, s, g = _prepare(y_true, y_score, groups)
    _, vals = _per_group(y, s, g)
    G = len(vals)
    ok = vals[~np.isnan(vals)]
    point = float(ok.mean()) if ok.size else float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_boot)):
        draw = rng.integers(0, G, size=G)
        v = vals[draw]
        v = v[~np.isnan(v)]
        stats.append(float(v.mean()) if v.size else float("nan"))
    stats = np.array(stats)
    stats = stats[~np.isnan(stats)]
    if stats.size:
        lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
        lo, hi = float(lo), float(hi)
    else:
        lo = hi = float("nan")
    return {"point": point, "lo": lo, "hi": hi, "n_boot_valid": int(stats.size)}
