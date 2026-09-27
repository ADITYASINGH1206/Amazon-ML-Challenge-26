#!/usr/bin/env python3
"""Business Entity Resolution — Instant threshold calibration wrapper.

Forwards to src/calibrate.py:
    python calibrate.py --threshold 0.70
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_CALIBRATE = os.path.join(HERE, "src", "calibrate.py")

if __name__ == "__main__":
    cmd = [sys.executable, SRC_CALIBRATE] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)
