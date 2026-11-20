# Idea 3 (design note only, NOT implemented, NOT verified): a scikit-learn-compatible purged splitter

Status: this is a sketch of a possible third task. No code, no tests and no reference solution
exist for it, and nothing in it has been run. It is kept here as a design note so that the
difference between the two finished tasks and a possible third one is explicit.

## Goal

Ship `PurgedGroupTimeSeriesSplit(n_splits, horizon, embargo)` as a `sklearn.model_selection`
compatible splitter: it must work inside `cross_val_score`, `GridSearchCV` and `Pipeline`
without the caller knowing about purging. This is `purged-group-cv` plus the scikit-learn contract.

## What the agent would have to get right

* The splitter protocol: `split(X, y, groups)` as a generator of `(train, test)` index arrays,
  `get_n_splits(X, y, groups)`, `groups=None` handling, `__repr__`/`get_params` behaviour expected
  by `sklearn.base.clone`.
* Pass-through of `groups` through `cross_val_score(..., groups=...)` and `GridSearchCV.fit(..., groups=...)`.
* Index types: positional integer arrays, sorted, no overlap between train and test.
* Interaction with `Pipeline`: fitting imputers/scalers on the training indices only (already
  guaranteed by sklearn when used correctly, which is the point of the metamorphic test below).

## How it could be verified

* Reuse the brute-force oracle and invariance tests of `purged-group-cv` on the splits it yields.
* Metamorphic test through the public sklearn API: replace test rows by arbitrary data and require
  identical fitted parameters of the training side.
* Contract tests against a few scikit-learn utilities.

## Why it is not built here

* It needs `scikit-learn` pinned in the image; its behaviour changes between minor versions, so the
  tests would have to be re-validated on every pin.
* The Docker image has never been built in the environment where this repository was written, and
  scikit-learn was not used to validate anything there. Writing tests for an API I could not run
  would produce unverified claims, which is what the other tasks deliberately avoid.
* The core difficulty (purge zone, groups, eager validation) is already covered by `purged-group-cv`.
