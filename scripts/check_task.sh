#!/bin/bash
# Local sanity check of every task, without Docker:
#   1. the starter must FAIL the hidden tests
#   2. the starter + reference solution (solve.sh, run for real) must PASS them
#      (and the small public tests shipped with the starter)
# Usage:  scripts/check_task.sh [task-name ...]      default: all tasks under tasks/
#         STARTER=0 scripts/check_task.sh            skip the starter runs (near-dup-split's takes ~2.5 min)
# Needs python3 with numpy, pandas, pytest (pinned versions: see the Dockerfiles).
# Exit code 0 only if every starter fails and every solution passes.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ $# -gt 0 ]; then TASKS=("$@"); else TASKS=($(cd "$HERE/tasks" && ls)); fi
STATUS=0

run_hidden() {  # $1 = app dir, $2 = task dir  -> last line of the pytest summary
  (cd "$WORK" && APP_ROOT="$1" CVKIT_ROOT="$1" PYTHONPATH="$2/tests" PYTHONDONTWRITEBYTECODE=1 \
     python3 -I -m pytest "$2/tests" -q -p no:cacheprovider 2>&1 | tail -1)
}

for name in "${TASKS[@]}"; do
  TASK="$HERE/tasks/$name"
  echo "== $name"

  prepare() {  # $1 = destination app dir: environment minus the Dockerfile
    mkdir -p "$1"
    cp -r "$TASK/environment/." "$1/"
    rm -f "$1/Dockerfile"
  }

  if [ "${STARTER:-1}" != "0" ]; then
    prepare "$WORK/$name/starter"
    out="$(run_hidden "$WORK/$name/starter" "$TASK")"
    echo "starter  : $out"
    echo "$out" | grep -q "failed" || { echo "  !! the starter should fail"; STATUS=1; }
  fi

  prepare "$WORK/$name/solution"
  # run the real solve.sh with /app and /solution redirected (tokens first, so substitutions cannot collide)
  sed "s#/solution/#@SOL@/#g; s#/app#@APP@#g; s#@SOL@#$TASK/solution#g; s#@APP@#$WORK/$name/solution#g" \
    "$TASK/solution/solve.sh" > "$WORK/$name/solve.sh"
  bash "$WORK/$name/solve.sh" || { echo "  !! solve.sh failed"; STATUS=1; }
  out="$(run_hidden "$WORK/$name/solution" "$TASK")"
  echo "solution : $out"
  echo "$out" | grep -q "passed" && ! echo "$out" | grep -q "failed" || { echo "  !! the solution should pass"; STATUS=1; }
  if [ -d "$WORK/$name/solution/tests_public" ]; then
    pub="$(cd "$WORK/$name/solution" && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests_public -q -p no:cacheprovider 2>&1 | tail -1)"
    echo "public   : $pub"
    echo "$pub" | grep -q "failed" && { echo "  !! public tests fail on the solution"; STATUS=1; }
  fi
done
exit $STATUS
