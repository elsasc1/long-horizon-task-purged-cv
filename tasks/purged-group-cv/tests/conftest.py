import os
import sys

ROOT = os.environ.get("CVKIT_ROOT", "/app")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
