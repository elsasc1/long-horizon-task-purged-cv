# Failure modes this task is designed to probe

> **Status: hypotheses, not measurements.** This task has not been run against any model
> yet, so nothing below is evidence that a model fails. The goal of this file is to
> make the *intended* difficulty explicit and checkable (e.g. with Harbor trials), and to
> separate real engineering gaps from stylistic preferences.

## Intended gaps (each maps to at least one test)

| # | Subtle requirement | Typical wrong implementation | Test that catches it |
|---|---|---|---|
| 1 | Purge on the **label window** `[t, t+horizon]` of *train* samples, so the zone starts at `lo - horizon` | Only drop train samples whose *timestamp* falls in the validation block | `test_label_window_of_train_sample_counts_not_just_its_start`, mutation "no purge before block" |
| 2 | Inclusive boundaries on both ends | `<` instead of `<=` (or the reverse) | `test_boundaries_are_inclusive_on_both_sides` + 40 oracle cases |
| 3 | Embargo only **after** `hi + horizon` | Symmetric embargo | `test_embargo_applies_only_after_the_block` |
| 4 | Equal timestamps stay in the same fold (chunks are built on **distinct** times) | Split by sample position | `test_tied_times_stay_in_the_same_fold` |
| 5 | Group exclusion composed with the time purge | Applying only one of them | `test_group_exclusion_exact`, oracle cases with groups |
| 6 | Positional indices, even with a shuffled pandas index | Returning index labels | `test_pandas_inputs_with_arbitrary_index_use_positions` |
| 7 | **Eager** validation in `split()` but **lazy** empty-train error | A generator function (validation runs on first `next`) or an upfront crash | `test_validation_is_eager_not_lazy`, `test_empty_train_is_raised_lazily_at_the_offending_fold` |
| 8 | AUC ties count 1/2 (average ranks) | `argsort` ranks without tie handling | `test_auc_with_ties_exact`, 15 pair-counting cases |
| 9 | **Macro** AUC over groups, single-class groups skipped | Pooled AUC | `test_macro_ignores_nan_groups_and_is_not_pooled` |
| 10 | Exact bootstrap stream (`default_rng(seed).integers`) so results are reproducible | `np.random.seed`, `rng.choice` on rows, resampling samples instead of groups | `test_bootstrap_matches_exact_algorithm` |
| 11 | Auditor is independent: checks *every* validation sample, ignores shared indices for checks 4–7, `NaN == NaN`, `0.0 == -0.0` | Re-using the splitter's block logic, `X` equality via `==` | `test_horizon_checks_every_val_sample_inside_the_span`, `test_duplicate_rows_nan_and_negative_zero` |
| 12 | Strict JSON (`NaN` → `null`), exit codes `0/1/2`, empty stdout on error | `json.dumps` defaults, tracebacks on stdout | `test_ci_command_matches_library_and_nan_becomes_null`, error-path tests |

## Not a "behaviour gap" (deliberately not tested)

- Code style, naming, helper structure, type hints, docstrings.
- Performance beyond what the test sizes need.
- Any choice the instruction leaves free (private helper functions, error message wording after `error:`).

## Fairness checklist

- [x] The instruction states every behaviour the tests check (conventions, eager vs lazy errors, JSON format).
- [x] A reference solution passes all tests (`scripts/check_task.sh`).
- [x] The untouched stubs fail all tests.
- [x] Float comparisons in oracle tests use dyadic values, so results do not depend on rounding order.
- [x] Mutation check: 15 deliberately broken variants of the reference solution were run; the tests
      reject every one that changes the specified behaviour (one variant, `rng.choice(G, size=G)`,
      produces the same stream as `rng.integers` in NumPy and is therefore *behaviourally equivalent*).
- [ ] Calibrated against real agent runs (pass rate, failure analysis). **Still to do.**

## Reward-hacking considerations

- Tests are behavioural and oracle-based (brute-force O(n²) reference in `tests/oracle.py`);
  there is no fixed output that can be hard-coded to pass all 144 checks.
- Hidden tests are never inside the image (Harbor copies `/tests` only at verification time).
- The agent has no network access, so it cannot fetch a reference implementation.
- The package forbids scikit-learn (not installed), removing the shortcut of wrapping
  `GroupKFold`/`roc_auc_score`.
