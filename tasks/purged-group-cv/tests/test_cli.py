import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cvkit import PurgedGroupTimeSplit, audit_splits, cluster_bootstrap_ci

ROOT = os.environ.get("CVKIT_ROOT", "/app")


def cli(*args):
    p = subprocess.run([sys.executable, "-m", "cvkit", *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


@pytest.fixture()
def data(tmp_path):
    rng = np.random.default_rng(11)
    n = 60
    df = pd.DataFrame({
        "t": rng.integers(0, 24, size=n) * 0.5,
        "g": 0,
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "y": rng.integers(0, 2, size=n),
    })
    df["g"] = (df["t"] // 2).astype(int)  # groups drift with time, so folds keep some train data
    df["s"] = rng.normal(size=n) + df["y"]
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return df, str(path), tmp_path


def test_split_command_matches_library(data):
    df, path, _ = data
    code, out, err = cli("split", "--data", path, "--time", "t", "--group", "g",
                         "--n-splits", 3, "--horizon", 1, "--embargo", 0.5)
    assert code == 0 and err == ""
    assert out.count("\n") == 1
    got = json.loads(out)
    exp = [{"fold": k, "train": a.tolist(), "val": b.tolist()}
           for k, (a, b) in enumerate(PurgedGroupTimeSplit(3, 1, 0.5).split(times=df.t, groups=df.g))]
    assert got == {"folds": exp}


def test_audit_ok_and_violation_exit_codes(data):
    df, path, tmp = data
    code, out, _ = cli("split", "--data", path, "--time", "t", "--group", "g",
                       "--n-splits", 3, "--horizon", 1, "--embargo", 0.5)
    splits = json.loads(out)
    sp = tmp / "s.json"
    sp.write_text(out)
    args = ["audit", "--data", path, "--time", "t", "--splits", str(sp), "--group", "g",
            "--features", "f1,f2", "--horizon", 1, "--embargo", 0.5]
    code, out2, err = cli(*args)
    assert code == 0 and json.loads(out2) == {"ok": True, "n_folds": 3, "violations": []}

    splits["folds"][1]["train"].append(splits["folds"][1]["val"][0])
    sp.write_text(json.dumps(splits))
    code, out3, _ = cli(*args)
    rep = json.loads(out3)
    assert code == 1 and not rep["ok"]
    assert any(v["fold"] == 1 and v["type"] == "index_overlap" for v in rep["violations"])
    exp = audit_splits([(f["train"], f["val"]) for f in splits["folds"]], df.t, 1, 0.5,
                       groups=df.g, X=df[["f1", "f2"]].to_numpy())
    assert rep == exp


def test_ci_command_matches_library_and_nan_becomes_null(data, tmp_path):
    df, path, _ = data
    code, out, err = cli("ci", "--data", path, "--y", "y", "--score", "s", "--group", "g",
                         "--n-boot", 120, "--alpha", 0.1, "--seed", 4)
    assert code == 0 and err == ""
    assert json.loads(out) == pytest.approx(
        cluster_bootstrap_ci(df.y, df.s, df.g, 120, 0.1, 4))

    bad = pd.DataFrame({"y": [1, 1, 1, 1], "s": [0.1, 0.2, 0.3, 0.4], "g": [0, 0, 1, 1]})
    p = tmp_path / "single.csv"
    bad.to_csv(p, index=False)
    code, out, _ = cli("ci", "--data", p, "--y", "y", "--score", "s", "--group", "g")
    res = json.loads(out)  # must be strict JSON: NaN literal would not parse in other tools
    assert "NaN" not in out
    assert res == {"point": None, "lo": None, "hi": None, "n_boot_valid": 0}


@pytest.mark.parametrize("argv", [
    ["split", "--data", "{missing}", "--time", "t", "--n-splits", 3],
    ["split", "--data", "{csv}", "--time", "nope", "--n-splits", 3],
    ["split", "--data", "{csv}", "--time", "t", "--group", "nope", "--n-splits", 3],
    ["split", "--data", "{csv}", "--time", "t", "--n-splits", 1],
    ["split", "--data", "{csv}", "--time", "t", "--n-splits", 3, "--horizon", -1],
    ["ci", "--data", "{csv}", "--y", "y", "--score", "s", "--group", "g", "--alpha", 2],
])
def test_errors_exit_2_with_stderr_and_empty_stdout(data, argv):
    _, path, tmp = data
    argv = [a.format(csv=path, missing=str(tmp / "missing.csv")) if isinstance(a, str) else a
            for a in argv]
    code, out, err = cli(*argv)
    assert code == 2 and out == "" and err.startswith("error:")
