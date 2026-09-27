#!/usr/bin/env python3
"""Business Entity Resolution — Entry point wrapper.

Forwards to src/run_pipeline.py so you can run directly from this directory:
    python run_pipeline.py
    python run_pipeline.py all
    python run_pipeline.py --stages predict
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_RUNNER = os.path.join(HERE, "src", "run_pipeline.py")

if __name__ == "__main__":
    args = sys.argv[1:]
    # Convert 'all' shorthand to '--stages all' if passed alone
    if len(args) == 1 and args[0] in ("all", "load", "dict", "norm", "finetune", "embed", "block", "feats", "train", "predict"):
        args = ["--stages", args[0]]
    cmd = [sys.executable, SRC_RUNNER] + args
    res = subprocess.run(cmd)
    sys.exit(res.returncode)
