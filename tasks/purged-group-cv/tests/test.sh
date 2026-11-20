#!/bin/bash
# Harbor verifier: run the hidden test-suite and write a binary reward.
mkdir -p /logs/verifier
cd /app
export PYTHONPATH=/app:/tests
python -m pytest /tests -q -p no:cacheprovider -x -rf
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
