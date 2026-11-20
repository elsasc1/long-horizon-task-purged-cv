import os
import signal
import sys

APP_ROOT = os.environ.get("APP_ROOT", "/app")
HERE = os.path.dirname(__file__)
for p in (APP_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest  # noqa: E402


class Timeout(Exception):
    pass


def run_with_timeout(fn, seconds):
    """Run fn() and abort it with Timeout after `seconds` (main thread only, POSIX)."""
    def handler(signum, frame):
        raise Timeout(f"exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture(scope="session")
def with_timeout():
    return run_with_timeout
