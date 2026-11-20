# Make the near-duplicate-aware split fast, without changing a single answer

Near-duplicate texts (copies with a changed word, different case or spacing) that land on both sides of a
train/validation split inflate validation scores. `/app/dedupsplit/core.py` already implements an
**exact** group-aware split, but it compares every pair of texts, so it is unusable beyond a few
thousand rows.

Your job: make `near_duplicate_clusters` and `grouped_split` fast on large inputs (about 10,000 texts
of roughly 350 characters, many sharing long boilerplate headers and footers) while returning
**exactly** what the current code returns on every input. This is not an approximate-dedup task:
no MinHash/LSH sampling, no "probably the same".

The specification is the docstring at the top of `core.py` (normalisation, 5-character shingles,
Jaccard similarity, inclusive threshold, connected components, canonical labels, the exact split
procedure). Summary of the points that are easy to get wrong:

* The edge test is `inter / union >= threshold` with float division; pairs whose similarity is
  *exactly* the threshold are linked.
* Clusters are connected components, so a chain A~B~C is one cluster even if A and C are not similar.
* Labels are numbered by first appearance and must not depend on Python's hash randomisation
  (`PYTHONHASHSEED`).
* `grouped_split` must follow the stated procedure exactly (including `math.ceil(val_fraction * n)`
  evaluated in floating point, and the seeded `numpy` permutation of the cluster ids).
* Invalid `threshold` / `val_fraction` raise `ValueError`; inputs must not be mutated; empty input works.

## Rules

* Public API stays the same: `dedupsplit.near_duplicate_clusters(texts, threshold) -> list[int]` and
  `dedupsplit.grouped_split(texts, threshold, val_fraction, seed) -> (train_idx, val_idx)`.
* Pure Python + `numpy` / `pandas` / standard library only; the environment is offline. Single CPU.
* Edit any file under `/app/dedupsplit`. The public tests in `/app/tests_public` must keep passing;
  they are tiny.

## Grading

A hidden test-suite compares your output with an independent brute-force implementation on many small
inputs (including thresholds at which floating-point rounding bites), checks determinism across
`PYTHONHASHSEED` values, and runs a 10,000-text corpus against a precomputed ground truth under a
wall-clock limit. Reward is 1 only if all hidden tests pass.
