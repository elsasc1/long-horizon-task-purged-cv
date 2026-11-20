"""Purged, embargoed, group-aware cross-validation splitter. TODO: implement (see instruction)."""


class PurgedGroupTimeSplit:
    def __init__(self, n_splits=5, horizon=0, embargo=0):
        raise NotImplementedError

    def get_n_splits(self, X=None, y=None, groups=None, times=None):
        raise NotImplementedError

    def split(self, X=None, y=None, groups=None, times=None):
        raise NotImplementedError
