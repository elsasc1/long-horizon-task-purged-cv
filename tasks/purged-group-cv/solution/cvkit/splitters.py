"""Purged, embargoed, group-aware cross-validation splitter (reference solution)."""
import numpy as np

from ._validate import as_group_codes, as_times, check_nonneg_number


class PurgedGroupTimeSplit:
    def __init__(self, n_splits=5, horizon=0, embargo=0):
        if isinstance(n_splits, (bool, np.bool_)) or not isinstance(
            n_splits, (int, np.integer)
        ):
            raise ValueError("n_splits must be an int >= 2")
        if n_splits < 2:
            raise ValueError("n_splits must be an int >= 2")
        self.n_splits = int(n_splits)
        self.horizon = check_nonneg_number("horizon", horizon)
        self.embargo = check_nonneg_number("embargo", embargo)

    def get_n_splits(self, X=None, y=None, groups=None, times=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None, times=None):
        # Eager validation: errors surface at call time, before iteration.
        t = as_times(times)
        n = len(t)
        codes = as_group_codes(groups, n) if groups is not None else None
        uniq = np.unique(t)
        if len(uniq) < self.n_splits:
            raise ValueError("fewer distinct times than n_splits")
        return self._generate(t, codes, uniq)

    def _generate(self, t, codes, uniq):
        for k, chunk in enumerate(np.array_split(uniq, self.n_splits)):
            lo, hi = chunk[0], chunk[-1]
            val = np.flatnonzero((t >= lo) & (t <= hi))
            upper = hi + self.horizon + self.embargo
            excluded = (t >= lo - self.horizon) & (t <= upper)
            keep = ~excluded
            if codes is not None:
                keep &= ~np.isin(codes, np.unique(codes[val]))
            train = np.flatnonzero(keep)
            if train.size == 0:
                raise ValueError(f"fold {k} has an empty training set")
            yield train.astype(np.int64), val.astype(np.int64)
