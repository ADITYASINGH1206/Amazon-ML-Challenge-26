"""Shared CLI, logging and result-recording helpers for every experiment script."""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

T0 = time.time()


def base_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--data-dir", default="dataset", help="folder containing train/ and test/")
    ap.add_argument("--work", default="er_work", help="all caches, universes, features and results go here")
    ap.add_argument("--universe", default="U1", help="evaluation universe tag (built by 00_prepare.py)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="smaller settings for a fast first pass")
    return ap


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.__stdout__

    def write(self, s):
        self.f.write(s)
        try:
            self.out.write(s)
        except UnicodeEncodeError:
            enc = self.out.encoding or "ascii"
            self.out.write(s.encode(enc, "replace").decode(enc))

    def flush(self):
        self.f.flush()
        self.out.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):  # anything else (encoding, fileno, ...) comes from the real stdout
        return getattr(self.out, name)


def setup(args, name):
    for d in ("cache", "universes", "results", "logs", "models"):
        os.makedirs(os.path.join(args.work, d), exist_ok=True)
    sys.stdout = Tee(os.path.join(args.work, "logs", f"{name}.log"))
    print("\n" + "#" * 100)
    print(f"# {name}  started {time.strftime('%Y-%m-%d %H:%M:%S')}  python {platform.python_version()} "
          f"cpu={os.cpu_count()}")
    print(f"# args: {vars(args)}")
    print("#" * 100)
    args._name = name


def step(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _clean(v):
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def log_result(args, exp, variant, **metrics):
    """One row per (experiment, variant). 99_report.py keeps the latest row for each pair."""
    row = {"exp": exp, "variant": str(variant), "universe": getattr(args, "universe", None),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), **{k: _clean(v) for k, v in metrics.items()}}
    with open(os.path.join(args.work, "results", "results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    show = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                     if k not in ("exp", "variant", "universe", "time") and not isinstance(v, (dict, list)))
    print(f"  RESULT [{exp}] {variant}: {show}", flush=True)


def udir(args, tag=None):
    return os.path.join(args.work, "universes", tag or args.universe)
