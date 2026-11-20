"""Results must not depend on Python's per-process string-hash randomisation."""
import hashlib
import os
import subprocess
import sys

import datagen
import oracle

HERE = os.path.dirname(__file__)
APP_ROOT = os.environ.get("APP_ROOT", "/app")
SCRIPT = (
    "import sys, hashlib; sys.path[:0] = [sys.argv[1], sys.argv[2]]\n"
    "import datagen; from dedupsplit import near_duplicate_clusters\n"
    "texts = datagen.make_corpus(700, 5, with_boundary_at=None)\n"
    "print(hashlib.sha256(','.join(map(str, near_duplicate_clusters(texts, 0.6))).encode()).hexdigest())\n"
)


def test_labels_do_not_depend_on_pythonhashseed():
    texts = datagen.make_corpus(700, 5, with_boundary_at=None)
    expected = hashlib.sha256(",".join(map(str, oracle.clusters(texts, 0.6))).encode()).hexdigest()
    for seed in ("0", "1", "2", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", SCRIPT, APP_ROOT, HERE], env=env, capture_output=True,
                             text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-500:]
        assert out.stdout.strip() == expected, f"labels change with PYTHONHASHSEED={seed}"
