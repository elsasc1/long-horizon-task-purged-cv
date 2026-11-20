import os
import sys

# Single-threaded BLAS: makes float results bit-reproducible across calls (set BEFORE numpy loads).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

APP_ROOT = os.environ.get("APP_ROOT", "/app")
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pytest  # noqa: E402

from datagen import make_dataset  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return make_dataset(n_customers=300, seed=11)


@pytest.fixture(scope="session")
def split(df):
    from datagen import grouped_folds
    return grouped_folds(df, 4, 5)[0]
