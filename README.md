# Long-horizon task examples: leakage-free validation

Three **original** software-engineering tasks in the [Harbor](https://harborframework.com/docs/tasks)
task format, all about one theme: validation scores that look better than they should because of
data leakage. Each task gives an agent a starter package and hidden tests; a reference solution
proves the task is solvable and a mutation check shows the tests catch the intended mistakes.

I built them to practise the skills this kind of work requires: a clear specification, a fair
verifier, and an honest account of where a model is *expected* to fail.

> **Status.** Every task: the reference solution passes all hidden tests and the starter fails.
> None has been calibrated against a model and the Docker images have not been built (tests were
> run in a virtualenv with the pinned versions). Difficulty ratings are the author's estimates,
> not measurements. Run `scripts/check_task.sh` to reproduce the numbers below.

| Task | What the agent does | Hidden tests | Starter / reference | Mutants caught |
|---|---|---|---|---|
| [`purged-group-cv`](tasks/purged-group-cv) | Implement a purged / embargoed / group-aware time-series splitter, an independent leakage auditor, group-wise metrics with a cluster bootstrap and a CLI, from stubs | 144 | 144 fail / 144 pass | 15 variants checked, see its notes |
| [`leaky-cv-pipeline`](tasks/leaky-cv-pipeline) | Find and fix five planted leaks (global scaler, global target encoding, global feature selection, early stopping on the validation fold, row-level folds) behind a feature cache | 29 | 17 fail / 29 pass | 11 of 12 (the survivor is not a leak) |
| [`near-dup-split`](tasks/near-dup-split) | Make an exact near-duplicate-aware split fast (10,000 texts) without changing any answer; includes a float-rounding trap in the usual prefix filter | 73 | 71 pass, 2 time out / 73 pass | 16 of 16 |

Each task folder contains `instruction.md` (the prompt), `environment/` (Dockerfile + starter),
`solution/` (reference + `solve.sh`), `tests/` (verifier) and `FAILURE_MODES.md` (design notes,
measured numbers and known limits).

A design note for a possible fourth piece, **not implemented**, is in
[`docs/IDEA_sklearn_purged_splitter.md`](docs/IDEA_sklearn_purged_splitter.md).

## Run the sanity check locally

```bash
pip install numpy==2.2.6 pandas==2.3.0 pytest==8.3.5
scripts/check_task.sh                 # all tasks (the near-dup-split starter run takes ~2.5 minutes)
STARTER=0 scripts/check_task.sh       # skip the starter runs
scripts/check_task.sh leaky-cv-pipeline
```

With Harbor installed (needs Docker): `harbor run -p tasks/<task> -a oracle`.

## Why this topic

Data leakage in validation (overlapping label windows, shared groups, duplicated rows, statistics
fitted on the whole dataset) silently inflates scores. These tasks make "does it really work?"
testable rather than a matter of opinion: by replacing validation rows with arbitrary data and
requiring identical training-side results, by null-label experiments, and by brute-force oracles.

## Notes

- Everything here is original work; it is not derived from any employer, platform or client task.
- License: MIT.
