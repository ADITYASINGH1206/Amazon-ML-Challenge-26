import json
import os
import sys
import time

T0 = time.time()


class Tee:
    """Mirror stdout into a log file (UTF-8), flushing every line so progress is visible."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.__stdout__

    def write(self, s):
        self.f.write(s)
        self.f.flush()
        try:
            self.out.write(s)
        except UnicodeEncodeError:
            enc = self.out.encoding or "ascii"
            self.out.write(s.encode(enc, "replace").decode(enc))
        self.out.flush()

    def flush(self):
        self.f.flush()
        self.out.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.out, name)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def exists(*paths):
    return all(os.path.exists(p) for p in paths)
