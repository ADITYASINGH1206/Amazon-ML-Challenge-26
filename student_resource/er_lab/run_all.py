#!/usr/bin/env python3
"""Run the whole experiment lab in order, continuing past failures, then build SUMMARY.md.

    python run_all.py                       # CPU experiments on the real data (dataset/ -> er_work/)
    python run_all.py --gpu                 # + neural embeddings, fine-tuning, cross-encoder (before the matcher)
    python run_all.py --quick               # smaller universes / fewer rounds for a fast first pass
    python run_all.py --from 03             # resume from a step
    python run_all.py --only 01,02          # run selected steps
    python run_all.py --toy                 # generate a tiny synthetic dataset and run everything on it (2-3 min)
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work", default="er_work")
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="run 06_embeddings (+finetune) and 07_cross_encoder")
    ap.add_argument("--skip-france", action="store_true")
    ap.add_argument("--from", dest="start", default="00")
    ap.add_argument("--only", default="")
    ap.add_argument("--toy", action="store_true")
    a = ap.parse_args()

    common = ["--data-dir", a.data_dir, "--work", a.work, "--n-jobs", str(a.n_jobs)] + (["--quick"] if a.quick else [])
    prep = []
    if a.toy:
        a.data_dir, a.work = "toy_dataset", "toy_work"
        subprocess.run([sys.executable, os.path.join(HERE, "tools", "make_toy_data.py"), "--out", a.data_dir],
                       check=True)
        common = ["--data-dir", a.data_dir, "--work", a.work, "--n-jobs", str(a.n_jobs)]
        prep = ["--fracs", "0.1,0.25,0.5", "--main-frac", "0.5", "--aux-frac", "0.08", "--dict-pairs", "50000"]
    steps = [
        ("00", "00_prepare.py", prep),
        ("01", "01_normalization.py", []),
        ("02", "02_blocking.py", ["--scaling", "--also-u2"]),
        ("03", "03_features.py", []),
    ]
    if not a.quick:
        steps += [("03b", "03_features.py", ["--level", "2"])]
    if a.gpu:
        steps += [("06", "06_embeddings.py", ["--finetune"]), ("07", "07_cross_encoder.py", [])]
    steps += [("04", "04_matcher.py", [])]
    if not a.quick:
        steps += [("04b", "04_matcher.py", ["--feats", "feats_L2", "--only", "A", "--tag", "featsL2:"])]
    steps += [("05", "05_decision.py", [])]
    if not a.skip_france:
        steps += [("08", "08_france_probe.py", [])]
    steps += [("99", "99_report.py", [])]

    only = set(a.only.split(",")) if a.only else None
    started = False
    status = []
    for sid, script, extra in steps:
        if only is not None:
            if sid not in only:
                continue
        elif not started:
            if sid < a.start and not sid.startswith(a.start):
                continue
            started = True
        cmd = [sys.executable, os.path.join(HERE, script)] + common + extra
        print(f"\n>>> [{sid}] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        status.append((sid, script, r.returncode, time.time() - t0))
        if r.returncode != 0:
            print(f"!!! step {sid} ({script}) failed with code {r.returncode} -- continuing", flush=True)
    print("\n=== run summary ===")
    for sid, script, rc, t in status:
        print(f"  [{sid}] {script:<24} {'OK ' if rc == 0 else 'FAIL'}  {t / 60:6.1f} min")
    print(f"\nSend back: {os.path.join(a.work, 'SUMMARY.md')} and the folder {os.path.join(a.work, 'logs')}")


if __name__ == "__main__":
    main()
