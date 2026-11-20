import math

import numpy as np
import pytest

from cvkit import cluster_bootstrap_ci, group_auc, macro_auc
from oracle import oracle_auc


def test_auc_with_ties_exact():
    out = group_auc([0, 1, 0, 1], [0.5, 0.5, 0.1, 0.9], ["g"] * 4)
    assert out == {"g": 0.875}


def test_perfect_and_inverted():
    y, s = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    assert group_auc(y, s, [1] * 4) == {1: 1.0}
    assert group_auc(y, [-x for x in s], [1] * 4) == {1: 0.0}


def test_single_class_group_is_nan_and_keys_are_native_sorted():
    y = [0, 1, 0, 0, 1, 0]
    s = [0.1, 0.9, 0.3, 0.2, 0.7, 0.4]
    g = ["b", "b", "a", "a", "c", "c"]
    out = group_auc(y, s, g)
    assert list(out) == ["a", "b", "c"]
    assert math.isnan(out["a"]) and out["b"] == 1.0 and out["c"] == 1.0
    assert all(type(k) is str for k in out) and all(type(v) is float for v in out.values())


def test_integer_group_keys_are_python_ints():
    out = group_auc([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.5], [3, 3, 7, 7])
    assert list(out) == [3, 7] and all(type(k) is int for k in out)


def test_bool_labels_and_float_labels_accepted():
    assert group_auc([False, True], [0.1, 0.9], [0, 0]) == {0: 1.0}
    assert group_auc([0.0, 1.0], [0.1, 0.9], [0, 0]) == {0: 1.0}


def test_macro_ignores_nan_groups_and_is_not_pooled():
    y = [0, 1, 0, 1, 1, 1]
    s = [0.2, 0.8, 0.9, 0.1, 0.5, 0.6]  # g0 AUC 1.0, g1 AUC 0.0, g2 single class
    g = [0, 0, 1, 1, 2, 2]
    assert macro_auc(y, s, g) == 0.5
    pooled = oracle_auc(y, s)
    assert pooled != 0.5  # makes sure a pooled implementation is caught


def test_macro_all_nan_is_nan():
    assert math.isnan(macro_auc([1, 1, 1], [0.1, 0.2, 0.3], [0, 0, 1]))


@pytest.mark.parametrize("seed", range(15))
def test_auc_matches_pair_counting(seed):
    rng = np.random.default_rng(seed)
    n = 60
    y = rng.integers(0, 2, size=n)
    s = rng.integers(0, 6, size=n) / 4.0  # heavy ties
    g = rng.integers(0, 4, size=n)
    out = group_auc(y, s, g)
    for k, v in out.items():
        exp = oracle_auc(y[g == k].tolist(), s[g == k].tolist())
        assert (math.isnan(v) and math.isnan(exp)) or abs(v - exp) < 1e-12


@pytest.mark.parametrize("bad", [
    dict(y_true=[0, 2], y_score=[0.1, 0.2], groups=[0, 0]),
    dict(y_true=[0, 1], y_score=[0.1, np.nan], groups=[0, 0]),
    dict(y_true=[0, 1], y_score=[0.1], groups=[0, 0]),
    dict(y_true=[0, 1], y_score=[0.1, 0.2], groups=[0, None]),
    dict(y_true=[[0, 1]], y_score=[[0.1, 0.2]], groups=[[0, 0]]),
])
def test_validation(bad):
    for fn in (group_auc, macro_auc, cluster_bootstrap_ci):
        with pytest.raises(ValueError):
            fn(**bad)


def _reference_ci(y, s, g, n_boot, alpha, seed):
    keys = sorted(set(g))
    a = np.array([oracle_auc([yy for yy, gg in zip(y, g) if gg == k],
                             [ss for ss, gg in zip(s, g) if gg == k]) for k in keys])
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(a), size=len(a))
        v = a[draw]
        v = v[~np.isnan(v)]
        stats.append(v.mean() if v.size else np.nan)
    stats = np.array(stats)
    stats = stats[~np.isnan(stats)]
    point = a[~np.isnan(a)].mean()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, lo, hi, stats.size


@pytest.mark.parametrize("seed", range(6))
def test_bootstrap_matches_exact_algorithm(seed):
    rng = np.random.default_rng(100 + seed)
    n = 120
    g = rng.integers(0, 9, size=n)
    y = rng.integers(0, 2, size=n)
    s = rng.normal(size=n) + 0.8 * y
    n_boot, alpha = int(rng.choice([50, 200, 333])), float(rng.choice([0.05, 0.1, 0.2]))
    res = cluster_bootstrap_ci(y, s, g, n_boot=n_boot, alpha=alpha, seed=seed)
    point, lo, hi, nv = _reference_ci(y.tolist(), s.tolist(), g.tolist(), n_boot, alpha, seed)
    assert res["point"] == pytest.approx(point, abs=1e-12)
    assert res["lo"] == pytest.approx(lo, abs=1e-12)
    assert res["hi"] == pytest.approx(hi, abs=1e-12)
    assert res["n_boot_valid"] == nv and type(res["n_boot_valid"]) is int
    assert all(type(res[k]) is float for k in ("point", "lo", "hi"))


def test_bootstrap_default_args_and_determinism():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 6, size=80)
    y = rng.integers(0, 2, size=80)
    s = rng.normal(size=80)
    a = cluster_bootstrap_ci(y, s, g)
    b = cluster_bootstrap_ci(y, s, g, n_boot=200, alpha=0.05, seed=0)
    assert a == b
    assert cluster_bootstrap_ci(y, s, g, seed=1) != a


def test_bootstrap_all_groups_single_class():
    res = cluster_bootstrap_ci([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
    assert math.isnan(res["point"]) and math.isnan(res["lo"]) and math.isnan(res["hi"])
    assert res["n_boot_valid"] == 0


@pytest.mark.parametrize("kw", [dict(n_boot=0), dict(n_boot=-1), dict(n_boot=True), dict(n_boot=2.5),
                                dict(alpha=0), dict(alpha=1), dict(alpha=-0.1)])
def test_bootstrap_param_validation(kw):
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], [0, 0, 1, 1], **kw)
