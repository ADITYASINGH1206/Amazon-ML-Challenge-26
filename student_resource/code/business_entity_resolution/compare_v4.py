#!/usr/bin/env python3
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_SCRIPT = os.path.join(HERE, "src", "compare_v4.py")

if __name__ == "__main__":
    res = subprocess.run([sys.executable, SRC_SCRIPT] + sys.argv[1:])
    sys.exit(res.returncode)
