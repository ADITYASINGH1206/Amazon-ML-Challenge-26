#!/usr/bin/env python3
"""
Convenience submission validator runner for ML Challenge 2026.

Auto-detects the dataset and output paths from src.config so you can run:
    python validate_submission.py
or
    python validate_submission.py --check-ids
from inside code/business_entity_resolution without needing to specify paths.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import src
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# Also ensure student_resource/utils can be imported
PROJECT_ROOT = THIS_DIR.parent.parent
STUDENT_UTILS = PROJECT_ROOT / "student_resource" / "utils"
if STUDENT_UTILS.exists():
    sys.path.insert(0, str(STUDENT_UTILS))

from src import config
from src.utils import log

# Import the core validate function
try:
    import importlib.util
    val_py = STUDENT_UTILS / "validate_submission.py"
    if val_py.exists():
        spec = importlib.util.spec_from_file_location("val_mod", str(val_py))
        val_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(val_mod)
        validate = val_mod.validate
    else:
        # Fallback to direct import if in python path
        from validate_submission import validate
except Exception as e:
    log.error(f"Could not load official validate function: {e}")
    sys.exit(1)


def run_validation(matching_path=None, candidate_path=None, test_dir=None, check_ids=False):
    """Run validation using configured paths."""
    matching_file = Path(matching_path) if matching_path else config.OUTPUT_DIR / "matching_results.tsv"
    cand_file = Path(candidate_path) if candidate_path else config.OUTPUT_DIR / "candidate_pairs.tsv"
    t_dir = Path(test_dir) if test_dir else config.TEST_DIR

    log.info("ML Challenge 2026 — Submission Validator")
    log.info(f"  matching:  {matching_file}")
    log.info(f"  candidate: {cand_file}")
    log.info(f"  test dir:  {t_dir}")
    log.info(f"  check IDs: {check_ids}")

    if not matching_file.exists():
        log.error(f"Matching file not found at: {matching_file}")
        return 1

    try:
        errors, warnings = validate(
            str(matching_file),
            str(cand_file) if cand_file.exists() else None,
            str(t_dir),
            check_ids=check_ids,
        )
    except Exception as exc:
        log.error(f"Validation encountered an error: {exc}")
        return 1

    print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"\nFAIL — {len(errors)} issue(s) to fix before submitting:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1

    print("\nPASS — no blocking issues found. Safe to submit. ✅")
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-configured submission validator.")
    parser.add_argument("--matching", "-m", default=None, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", "-c", default=None, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", "-t", default=None, help="Path to test dataset directory")
    parser.add_argument("--check-ids", action="store_true", help="Also check that every matched ID exists in test S2/S3")
    args = parser.parse_args()

    ret = run_validation(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
        check_ids=args.check_ids,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
