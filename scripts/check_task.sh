#!/bin/bash
# Local sanity check of the task, without Docker:
#   1. the starter stubs must FAIL the hidden tests
#   2. the reference solution must PASS them
# Usage: scripts/check_task.sh   (needs python3, numpy, pandas, pytest)
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TASK="$HERE/tasks/purged-group-cv"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

run() {  # $1 = folder containing a cvkit/ package
  (cd "$WORK" && CVKIT_ROOT="$1" PYTHONPATH="$TASK/tests" PYTHONDONTWRITEBYTECODE=1 \
     python3 -I -m pytest "$TASK/tests" -q -p no:cacheprovider 2>&1 | tail -1)
}

mkdir -p "$WORK/starter" "$WORK/solution"
cp -r "$TASK/environment/cvkit" "$WORK/starter/cvkit"
cp -r "$TASK/solution/cvkit" "$WORK/solution/cvkit"

echo "starter  : $(run "$WORK/starter")"
echo "solution : $(run "$WORK/solution")"
