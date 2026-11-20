# Build `cvkit`: leakage-safe cross-validation, auditing and group-aware metrics

You are working in `/app`, a Python package called `cvkit` whose modules are
currently stubs that raise `NotImplementedError`. Implement the package so that it
meets the specification below. A hidden test suite checks the behaviour; the
specification is complete, so follow it literally (including edge cases).

Constraints: Python 3.12, `numpy` and `pandas` only (no scikit-learn, no network).
Do not change the public names or signatures that are already in the stubs.

Conventions used throughout: all indices are **positional** (0-based, relative to the
order of the inputs, never pandas labels); index arrays are `numpy.int64`, sorted
ascending; times are numeric (int or float).

---

## 1. `cvkit.PurgedGroupTimeSplit(n_splits=5, horizon=0, embargo=0)`

A cross-validation splitter for samples whose *label* at time `t` depends on data in
the closed window `[t, t + horizon]`. It builds contiguous validation blocks in time and
removes ("purges") training samples that could leak information into them.

**Constructor.** `n_splits` must be an `int` (not `bool`) `>= 2`; `horizon` and
`embargo` must be finite real numbers `>= 0` (not `bool`). Otherwise raise `ValueError`.

`get_n_splits(...)` returns `n_splits` and accepts (and ignores) the same arguments as `split`.

**`split(X=None, y=None, groups=None, times=None)`** returns an iterator of
`(train_idx, val_idx)` pairs, one per fold, in fold order.

*Eager validation*: the following errors are raised **when `split(...)` is called**,
before any iteration:
- `times is None` → `ValueError`
- `times` not 1-D → `ValueError`; dtype not integer/unsigned/float (e.g. `datetime64`,
  strings, bool, object) → `TypeError`; contains NaN or ±inf → `ValueError`
- `groups` given but not 1-D, or its length differs from `times` → `ValueError`;
  `groups` containing missing values (`None`, `NaN`) → `ValueError`
- fewer distinct values in `times` than `n_splits` → `ValueError`

`times` and `groups` may be lists, numpy arrays or pandas Series (with any index).
`X` and `y` are ignored. Inputs must not be mutated.

**Fold construction.** Let `U = numpy.unique(times)` (sorted distinct times). Split `U`
into `n_splits` contiguous chunks with `numpy.array_split(U, n_splits)`. For fold `k`
let `lo = chunk[0]` and `hi = chunk[-1]`.

- **Validation set** = all samples with `lo <= t <= hi` (so samples with equal times are
  always in the same fold).
- **Time purge/embargo zone** = closed interval `[lo - horizon, hi + horizon + embargo]`
  (evaluated in float64, `hi + horizon + embargo` left to right). Every sample whose
  time lies in this zone is excluded from training.
- **Group exclusion** (only if `groups` is given): every sample whose group appears
  among the validation samples is also excluded from training.
- **Training set** = all remaining samples.

Folds are produced lazily. If a fold's training set is empty, raise `ValueError` when
iteration reaches that fold (earlier folds have already been yielded).

---

## 2. `cvkit.audit_splits(splits, times, horizon=0, embargo=0, groups=None, X=None)`

An *independent* checker for splits produced by anyone (not necessarily by
`PurgedGroupTimeSplit`). `splits` is any iterable of `(train_idx, val_idx)` pairs of
integer sequences/arrays (it may be a generator). `horizon`/`embargo` are validated as in §1,
`times`/`groups` as in §1 (same exceptions). `X`, if given, must be 2-D with one row per
sample (else `ValueError`) and is converted to float64.

Returns a dict:

```python
{"ok": bool, "n_folds": int, "violations": [ {"fold": int, "type": str, "count": int}, ... ]}
```

`ok` is `True` iff `violations` is empty. Violations are ordered by fold, then by the
fixed type order below. At most one entry per (fold, type). Index arrays are
de-duplicated before checking (`numpy.unique`). Integer-typed index arrays are required;
a non-integer, non-empty array raises `TypeError`. For each fold:

1. `invalid_index`: `count` = number of entries (train + val) that are `< 0` or `>= n`.
   If present, report only this and skip all other checks for the fold.
2. `empty`: if the validation set or the training set is empty, `count = 0`; skip the remaining checks.
3. `index_overlap`: number of indices present in both train and val.

   *For checks 4–7, ignore train indices that are also in val.*
4. `group_leak` (only if `groups` given): number of train samples whose group occurs in val.
5. `horizon_overlap`: number of train samples `i` for which some val sample `j` has
   `|t_i - t_j| <= horizon`.
6. `embargo`: with `hi = max(t of val samples)`, number of train samples with
   `hi + horizon < t_i <= hi + horizon + embargo`.
7. `duplicate_rows` (only if `X` given): number of train samples whose feature row is
   identical to the row of at least one val sample. NaN equals NaN; `0.0` equals `-0.0`.

A correct `PurgedGroupTimeSplit` run with the same parameters must always audit as `ok`.

---

## 3. `cvkit.metrics`

`group_auc(y_true, y_score, groups)` → `dict` mapping each group (as a native Python
value, in `numpy.unique` order) to the AUC of that group, as a Python `float`.
AUC = P(score_pos > score_neg) + 0.5·P(score_pos == score_neg). A group with only one
class has AUC `nan`. Errors → `ValueError`: lengths differ or not 1-D; `y_true` not in
`{0, 1}` (bool is accepted as 0/1); `y_score` contains NaN; `groups` contains missing values.

`macro_auc(y_true, y_score, groups)` → `float`: mean of the non-`nan` per-group AUCs
(`nan` if there are none). Same validation.

`cluster_bootstrap_ci(y_true, y_score, groups, n_boot=200, alpha=0.05, seed=0)` →
`{"point": float, "lo": float, "hi": float, "n_boot_valid": int}`.
`n_boot` must be an `int >= 1`, `0 < alpha < 1`, else `ValueError`. Algorithm (exactly):

1. `a` = per-group AUCs as an array in `numpy.unique(groups)` order; `G = len(a)`;
   `point` = macro AUC as above.
2. `rng = numpy.random.default_rng(seed)`.
3. For each of `n_boot` repetitions: `draw = rng.integers(0, G, size=G)`; the statistic is the
   mean of `a[draw]` ignoring `nan` entries (a group drawn twice counts twice), or `nan`
   if all entries are `nan`.
4. Discard `nan` statistics; `n_boot_valid` is how many remain.
   `lo, hi = numpy.quantile(valid, [alpha/2, 1 - alpha/2])` (default linear interpolation);
   `nan, nan` if none remain.

---

## 4. Command line: `python -m cvkit <command> ...` (run from `/app`)

All output goes to **stdout as a single line of JSON**. Any non-finite float is written
as `null`. On success exit code `0` (except `audit`, see below). On any error (missing
file or column, `ValueError`, `TypeError`) print `error: <message>` to **stderr**, print
nothing to stdout, and exit with code `2`. Data files are CSVs with a header.

- `split --data CSV --time COL [--group COL] --n-splits N [--horizon H] [--embargo E]`
  → `{"folds": [{"fold": 0, "train": [...], "val": [...]}, ...]}` (the splitter's output).
- `audit --data CSV --time COL --splits JSON [--group COL] [--features C1,C2,...] [--horizon H] [--embargo E]`
  → the dict returned by `audit_splits`. `--splits` is a JSON file in the format printed
  by `split`. Exit code `0` if `ok`, `1` if there are violations.
- `ci --data CSV --y COL --score COL --group COL [--n-boot N] [--alpha A] [--seed S]`
  → the dict returned by `cluster_bootstrap_ci` (defaults as in §3).

`H`, `E`, `A` are floats; `N`, `S` are integers.
