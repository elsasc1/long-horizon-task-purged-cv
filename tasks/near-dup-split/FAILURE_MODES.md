# Near-duplicate split: design notes

Author's notes. Difficulty is a hypothesis: nothing here was calibrated against a model run, and the
Docker image was never built where this was written (tests were run with the pinned package versions
in a virtualenv instead).

## What makes it harder than it looks

The starter is already *correct*; the task is to keep it exactly correct while making it fast. The
traps are all in the speed-up:

1. **Boilerplate shingles.** Every text shares a ~100-character header (5 variants) and a footer. A
   naive inverted index on "share any shingle" makes ~n^2 candidate pairs. Prefix filtering with a
   rarest-first token order (or equivalent) is needed.
2. **Floating-point bounds.** The standard prefix/size filters use `ceil(t * n)`. For some
   thresholds `t * n` evaluates slightly above the true integer (`0.28 * 25 == 7.000000000000001`,
   `0.07 * 100`, `0.55 * 100`, ...), so a naive `ceil` makes the filter one token too tight and
   silently drops pairs whose similarity is exactly `t`. The tests include such pairs at these
   thresholds (constructed from distinct CJK characters so the exact Jaccard is known: `b / a`).
3. **Transitive clusters**: chains whose ends are not similar must still merge.
4. **Canonical labels + hash randomisation**: labels are first-appearance order; Python's `set`
   iteration order of strings changes between runs.
5. **Exact split procedure**: `math.ceil(0.07 * 100) == 8`, not 7; seeded permutation of clusters.
6. **Unicode**: `str.lower()` (not `casefold`), `str.split()` on any Unicode whitespace.

## Verification design

* `tests/oracle.py`: independent brute-force clustering (no prefix filter, no index; only the
  provably exact size filter `min/max >= t` when run at scale).
* `test_spec.py`: 70 tests - normalisation, short/empty texts, exact-threshold and just-below pairs
  at 11 thresholds, chains, 24 random boilerplate-heavy corpora vs the oracle, validation errors,
  split procedure.
* `test_determinism.py`: same labels under four `PYTHONHASHSEED` values.
* `test_scale.py`: 10,020 texts (10,000 generated + 20 boundary texts) at threshold 0.7; the
  ground truth (4,313 clusters, SHA-256 of the label list) was computed once with the brute-force
  oracle (about 5 minutes on 2 cores, `tests/make_expected.py`) and is committed as
  `expected_scale.json`. Wall-clock limit 60 s per call (`signal.setitimer`), the reference solution
  takes about 7 s; the quadratic starter would need tens of minutes.

## Mutation check (reference solution)

16 mutants, all caught by `test_spec.py` / `test_determinism.py`: naive float `ceil` in the prefix
length, naive float `ceil` in the size filter, `>` instead of `>=`, empty texts not merged,
non-canonical labels, no `lower()`, `casefold`, short texts without shingles, 4-character
shingles, size filter one too strict, no verification, split target `int()` instead of `ceil`,
split ignoring the seed, row-level split, `split(" ")`, prefix one token too short.

Honest note: no mutant was caught *only* by `test_determinism.py`; with canonical first-appearance
labels a correct algorithm is hash-order independent, and sloppy ones also fail the oracle
comparisons. It stays as a cheap guard for implementations that number clusters by set iteration.

## Known limits

* An engineer who knows prefix filtering (AllPairs/PPJoin) solves it in well under an hour; the
  float-rounding trap and boilerplate are what a hurried solution misses. Hours, not days.
* The scale ground truth depends on the generator's output under NumPy 2.2.6 (pinned in the image).
* Timing limit assumes a core no more than ~8x slower than the author's.
* Not calibrated against any model.
