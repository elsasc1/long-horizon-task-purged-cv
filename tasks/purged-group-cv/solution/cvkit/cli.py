"""Command line interface (reference solution)."""
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from .audit import audit_splits
from .metrics import cluster_bootstrap_ci
from .splitters import PurgedGroupTimeSplit


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _emit(obj):
    sys.stdout.write(json.dumps(_clean(obj), allow_nan=False) + "\n")


def _read(path, columns):
    df = pd.read_csv(path)
    for c in columns:
        if c is not None and c not in df.columns:
            raise KeyError(f"column not found: {c}")
    return df


def _cmd_split(a):
    df = _read(a.data, [a.time, a.group])
    sp = PurgedGroupTimeSplit(a.n_splits, a.horizon, a.embargo)
    groups = df[a.group] if a.group else None
    folds = [
        {"fold": k, "train": tr, "val": va}
        for k, (tr, va) in enumerate(sp.split(times=df[a.time], groups=groups))
    ]
    _emit({"folds": folds})
    return 0


def _cmd_audit(a):
    feats = a.features.split(",") if a.features else []
    df = _read(a.data, [a.time, a.group] + feats)
    with open(a.splits) as f:
        folds = json.load(f)["folds"]
    report = audit_splits(
        [(x["train"], x["val"]) for x in folds],
        df[a.time],
        a.horizon,
        a.embargo,
        groups=df[a.group] if a.group else None,
        X=df[feats].to_numpy(dtype=float) if feats else None,
    )
    _emit(report)
    return 0 if report["ok"] else 1


def _cmd_ci(a):
    df = _read(a.data, [a.y, a.score, a.group])
    res = cluster_bootstrap_ci(
        df[a.y], df[a.score], df[a.group], a.n_boot, a.alpha, a.seed
    )
    _emit(res)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="cvkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split")
    s.add_argument("--data", required=True)
    s.add_argument("--time", required=True)
    s.add_argument("--group")
    s.add_argument("--n-splits", type=int, required=True)
    s.add_argument("--horizon", type=float, default=0.0)
    s.add_argument("--embargo", type=float, default=0.0)
    s.set_defaults(fn=_cmd_split)

    s = sub.add_parser("audit")
    s.add_argument("--data", required=True)
    s.add_argument("--time", required=True)
    s.add_argument("--splits", required=True)
    s.add_argument("--group")
    s.add_argument("--features")
    s.add_argument("--horizon", type=float, default=0.0)
    s.add_argument("--embargo", type=float, default=0.0)
    s.set_defaults(fn=_cmd_audit)

    s = sub.add_parser("ci")
    s.add_argument("--data", required=True)
    s.add_argument("--y", required=True)
    s.add_argument("--score", required=True)
    s.add_argument("--group", required=True)
    s.add_argument("--n-boot", type=int, default=200)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_ci)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
