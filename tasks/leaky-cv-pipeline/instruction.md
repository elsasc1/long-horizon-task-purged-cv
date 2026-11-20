# Fix the leaky cross-validation pipeline

`/app/leaklab` is a small tabular-ML package that cross-validates a logistic model on customer
data (`customer_id`, `segment`, 30 numeric features `x0..x29`, binary target `y`; several rows per
customer). It reports a mean AUC of about 0.78 on the bundled synthetic dataset
(`leaklab.data.make_dataset`).

That number is **not trustworthy**. When the labels are replaced by values that carry no information
about the features (`make_dataset(null_labels=True)`), the package still reports an AUC far above
chance (about 0.7 instead of 0.5). Several independent mechanisms inflate the score.

## Your job

Make `cross_validate` an honest estimate of how the pipeline performs **on customers it has never
seen**, without changing what the pipeline does when it is used correctly:

* Everything a fold learns (imputation, scaling, target encoding, feature selection, number of
  training epochs, model weights) must be derived from that fold's **training rows only**.
  Nothing in the fold's training-side results may depend on the validation rows' features or labels,
  not even through a cache, a module-level variable or a hash of the whole dataset.
* Validation must be grouped by `customer_id`: a customer must never appear on both sides of a split.
* The model must still use the same kind of inputs (imputed and scaled numeric features, a top-`k`
  selection with `Config.k_features`, and the smoothed target encoding of `segment`) and must still
  learn real signal when there is some.
* The validation predictions must not depend on validation labels. A segment never seen in training
  must be encoded with the training prior.

## Contract (keep these)

* Public API of `leaklab` (`Config`, `CVReport`, `FoldArtifacts`, `cross_validate`, `make_folds`,
  `run_fold`) and the field names of `FoldArtifacts` / `CVReport` stay as they are.
* `make_folds(df, n_splits, seed)` returns `[(train_idx, val_idx), ...]`: sorted `int64` *positional*
  indices, every row validated exactly once, deterministic given `seed`, customers spread evenly
  over the folds.
* `run_fold(df, train_idx, val_idx, config)` must give the same result whenever it is called with
  the same arguments, whatever was called before. The input `DataFrame` must not be mutated.
* The environment is offline; only `numpy`, `pandas` and the standard library are available.
* The public smoke tests in `/app/tests_public` must keep passing. They are far from sufficient.

You may edit or delete any module of `leaklab` except `Config`'s field names and the helper
functions' semantics in `preprocess.py`, `encode.py`, `select.py`, `metrics.py` and `model.py`
(you may call them as often as you like).

## Grading

A hidden test-suite checks the points above, including how results react when the validation rows
are replaced by arbitrary data. Reward is 1 only if all hidden tests pass.
