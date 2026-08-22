#!/usr/bin/env python3
"""
Reference conformance-fixture runner for EvalPort (Discussion #9).

Every file in fixtures/*.json is a self-contained, portable test case: a
`type` (testcase | grader | suite | resultset), a `document` to validate, and
an `expect.valid` boolean this repo's own Python SDK is checked against here.
The point of this suite is NOT "does the Python SDK pass" -- the Python SDK's
own pytest suite (sdk/python/tests/) already covers that far more thoroughly.
The point is that these fixtures are a portable, language-agnostic artifact:
a conformance implementation in any language (Rust, Go, a browser-only JS
build, whatever) can load these same JSON files, run its OWN validator
against `document`, and check its own answer against `expect` -- without
needing this repo's Python code, or even a network connection.

Usage:
    python3 spec/conformance/run.py

Exits non-zero if any fixture's expectation doesn't hold, so this is CI-usable
directly (and IS wired into CI -- see .github/workflows/ci.yml).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FIXTURES_DIR = _HERE / "fixtures"
_SDK_DIR = _HERE.parent.parent / "sdk" / "python"

sys.path.insert(0, str(_SDK_DIR))

from openeval.validate import (  # noqa: E402
    validate_grader,
    validate_result_set,
    validate_suite,
    validate_test_case,
)

_VALIDATORS = {
    "testcase": validate_test_case,
    "grader": validate_grader,
    "suite": validate_suite,
    "resultset": validate_result_set,
}


def _load_fixtures():
    return sorted(_FIXTURES_DIR.glob("*.json"))


def main() -> int:
    fixture_paths = _load_fixtures()
    if not fixture_paths:
        print(f"No fixtures found in {_FIXTURES_DIR}", file=sys.stderr)
        return 1

    failures = []
    for path in fixture_paths:
        fixture = json.loads(path.read_text())
        doc_type = fixture["type"]
        document = fixture["document"]
        expected_valid = fixture["expect"]["valid"]
        expected_error_paths = fixture["expect"].get("error_paths")

        validator = _VALIDATORS.get(doc_type)
        if validator is None:
            failures.append(f"{path.name}: unknown type '{doc_type}'")
            continue

        result = validator(document)
        actual_valid = result.valid

        if actual_valid != expected_valid:
            failures.append(
                f"{path.name}: expected valid={expected_valid}, got valid={actual_valid} "
                f"(errors: {result.errors})"
            )
            continue

        if expected_error_paths is not None and not actual_valid:
            actual_paths = {e["path"] for e in result.errors}
            missing = [p for p in expected_error_paths if p not in actual_paths]
            if missing:
                failures.append(
                    f"{path.name}: expected error path(s) {missing} not present in "
                    f"actual errors {sorted(actual_paths)}"
                )
                continue

        print(f"[PASS] {path.name} -- {fixture['description'][:88]}")

    print()
    if failures:
        print(f"=== {len(failures)}/{len(fixture_paths)} FIXTURE(S) FAILED ===", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"=== ALL {len(fixture_paths)} CONFORMANCE FIXTURES PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
