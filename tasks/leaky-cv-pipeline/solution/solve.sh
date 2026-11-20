#!/bin/bash
# Reference solution: replace the leaky cv module and drop the whole-dataset feature cache.
set -e
cp /solution/leaklab/cv.py /app/leaklab/cv.py
rm -f /app/leaklab/cache.py
