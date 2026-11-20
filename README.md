# Long-horizon task example: leakage-safe cross-validation toolkit

An **original** long-horizon software-engineering task, written in the
[Harbor](https://harborframework.com/docs/tasks) task format. An agent receives a
Python package (`cvkit`) made of stubs and must implement a purged / embargoed /
group-aware time-series splitter, an independent leakage auditor, group-wise metrics with
a cluster bootstrap, and a small CLI — all verified by hidden tests.

I built it to practise the skill that this kind of work requires: writing a clear
specification, a fair verifier, and an honest account of where a model is *expected* to fail.

> **Status:** reference solution passes all 144 tests, the starter fails all 144, and 15
> mutation variants were checked (details in [`FAILURE_MODES.md`](tasks/purged-group-cv/FAILURE_MODES.md)).
> The task has **not** been calibrated against any model yet and the Docker image has
> not been built in CI. The difficulty rating is therefore an estimate, not a measurement.

## Layout

```
tasks/purged-group-cv/
├── task.toml                 # Harbor config (timeouts, resources, metadata)
├── instruction.md            # the prompt shown to the agent (complete specification)
├── environment/              # Dockerfile + starter package with stubs
├── solution/                 # reference implementation + solve.sh (oracle)
├── tests/                    # test.sh + pytest suite + brute-force oracle
└── FAILURE_MODES.md          # intended failure modes, fairness checklist
scripts/check_task.sh         # starter must fail, solution must pass (no Docker needed)
```

## Run the sanity check locally

```bash
pip install numpy pandas pytest
scripts/check_task.sh
# starter  : 144 failed
# solution : 144 passed
```

With Harbor installed (needs Docker): `harbor run -p tasks/purged-group-cv -a oracle`.

## Why this topic

Data leakage in validation (overlapping label windows, shared groups, duplicated rows)
silently inflates scores. The task asks for the checks to be *proved* by an independent
auditor, which makes "does it really work?" testable rather than a matter of opinion.

## Notes

- Everything here is original work; it is not derived from any employer, platform or client task.
- License: MIT.
