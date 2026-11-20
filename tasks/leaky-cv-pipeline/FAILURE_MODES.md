# Leaky CV pipeline: what is planted, how it is checked

Author's design notes. Difficulty claims are hypotheses: nothing here was calibrated against a
model run, and the Docker image was never built in the environment where this was written (the
suite was run with the pinned versions in a venv instead).

## The five planted leaks (all in the starter)

| # | Leak | Where | Why it is easy to miss |
|---|------|-------|------------------------|
| 1 | Imputer + scaler fitted on all rows | `cache.build_features` | Looks like harmless preprocessing |
| 2 | Target encoding of `segment` computed with every label (each customer sees its own label) | `cache.build_features` | Smoothing makes it look regularised |
| 3 | Top-`k` feature selection by correlation with `y` on all rows | `cache.build_features` | 30 noise features, `k=6`: classic selection bias |
| 4 | Early stopping on the validation fold | `cv.run_fold` | A "tuning detail", not obviously a label leak |
| 5 | Row-level folds although a customer has several rows (labels are customer-level) | `cv.make_folds` | `customer_id` is not a feature, so it feels irrelevant |

Plus an amplifier: the feature matrix is cached by a hash of the whole dataframe, so even a
correct `run_fold` that still calls `build_features` stays leaky. Moving fits inside the fold
requires removing the cache indirection, not just editing one function.

## How the hidden tests catch them

* **Metamorphic invariance** (`test_invariance.py`): replace the validation rows by random labels,
  flipped labels, constant labels, 1e6-scale features, all-NaN features or an unseen segment;
  every training-side artifact (medians, mean, std, selected columns, encoding table and prior,
  number of epochs, training predictions) must stay bit-identical, and validation predictions
  must not move with validation labels. This catches leaks 1-4 regardless of how they are coded.
* **Independent recomputation**: medians, mean, std, encoding prior and segment set are recomputed
  from the training rows in the test.
* **Folds** (`test_folds.py`): partition, customer-disjointness, balance, determinism, positional
  indices, non-contiguous ids. Leak 5.
* **Hidden state** (`test_state.py`): repeated/interleaved calls, a one-cell change in the training
  data, no mutation of the input.
* **Statistics** (`test_statistics.py`): AUC on null labels over 8 seeds must be 0.5 +/- 0.04
  (starter: ~0.69, reference: ~0.50); AUC on real data over 4 seeds must stay >= 0.72 (reference
  ~0.77), so "fixing" leakage by deleting the signal fails.

Thresholds are deterministic (fixed seeds, single-threaded BLAS); the null band is about 3.5
standard errors of the reference's 8-seed mean.

## Measured on the reference solution

| | starter | reference |
|---|---|---|
| hidden tests | 17 failed / 12 passed | 29 passed |
| mean AUC, real data (6 seeds) | 0.777 | 0.767 |
| mean AUC, null labels (12 seeds) | 0.690 | 0.504 |

Note how small the gap is on real data: the leak is invisible in the headline metric, which is
exactly why a null-label check or an invariance check is needed.

## Mutation check

Twelve mutants of the reference solution, each re-introducing one defect. 11 are caught
(scaler/imputer on all rows, encoding on all rows, selection on all rows, early stopping on the
validation fold, row-level folds, prior from all labels, constant predictions, fold ignored,
validation-label-dependent epoch count, median from all rows, dropping the encoding column).
One survives on purpose: an inner early-stopping split that is row-level instead of
customer-grouped. It never touches the outer validation fold, so it is a quality choice, not a leak.

## Known limits

* Hours, not days, of work for a strong engineer; the difficulty is in finding all leaks and
  in the cache indirection, not in the code volume.
* Float determinism relies on the single-threaded BLAS set in `conftest.py`.
* Not calibrated against any model.
