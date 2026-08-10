#!/usr/bin/env python3
"""Validate every EvalPort suite under benchmarks/ against the real SDK validator.

This is the CI gate for the benchmark hub: a suite that doesn't pass
`openeval.validate.validate_suite()` fails the build. No suite should ever
be merged that this script doesn't accept.

Usage:
    python3 validate_all.py                # validate every *.json suite under benchmarks/
    python3 validate_all.py --quiet         # only print failures + summary
    python3 validate_all.py path/to/one.json path/to/two.json   # validate specific files

Exit code is 0 if every suite is valid, 1 otherwise (suitable for CI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk" / "python"))

from openeval.validate import validate_suite  # noqa: E402

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent

# Files under benchmarks/ that are not suite JSON and should be skipped when
# globbing — none currently, but keeping this explicit avoids ever accidentally
# trying to validate e.g. a stray package.json or lockfile as an EvalPort suite.
_SKIP_NAMES = {"package.json", "package-lock.json"}


def _discover_suites() -> List[Path]:
    return sorted(
        p
        for p in BENCHMARKS_DIR.rglob("*.json")
        if p.name not in _SKIP_NAMES and "_tools" not in p.parts
    )


def main() -> int:
    args = sys.argv[1:]
    quiet = "--quiet" in args
    explicit_paths = [Path(a) for a in args if a != "--quiet"]

    paths = explicit_paths if explicit_paths else _discover_suites()

    if not paths:
        print("No suite files found under benchmarks/ — nothing to validate.", file=sys.stderr)
        return 1

    total_cases = 0
    failures = []

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                suite = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            failures.append((path, [f"could not read/parse file: {exc}"]))
            print(f"FAIL  {path}  (unreadable: {exc})")
            continue

        result = validate_suite(suite)
        n_cases = len(suite.get("test_cases", []))
        total_cases += n_cases

        if result.valid:
            if not quiet:
                print(f"OK    {path}  ({n_cases} cases)")
        else:
            failures.append((path, result.errors))
            print(f"FAIL  {path}")
            for err in result.errors:
                print(f"        - {err}")

    print()
    print(f"Validated {len(paths)} suite file(s), {total_cases} total test cases.")
    if failures:
        print(f"{len(failures)} suite(s) FAILED validation:")
        for path, errors in failures:
            print(f"  - {path}: {len(errors)} error(s)")
        return 1

    print("All suites valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
