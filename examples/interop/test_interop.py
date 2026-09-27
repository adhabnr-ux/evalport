"""Runs each interop demo as a subprocess and checks its exit code and report.

    pytest examples/interop/test_interop.py -v

Each demo exits non-zero on any validation failure or undocumented mismatch,
so exit code 0 is the main assertion; the output checks pin down the claims
the README makes about each demo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def run_demo(script: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(HERE / script)],
        cwd=HERE.parents[1], capture_output=True, text=True, timeout=300,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{script} exited {proc.returncode}:\n{output}"
    return proc.stdout


def lines(output: str) -> list:
    return [" ".join(line.split()) for line in output.splitlines()]


def test_dataset_portability():
    out = run_demo("1_dataset_portability.py")
    rows = lines(out)
    assert "RESULT: PASS" in out
    assert "UNEXPECTED" not in out
    assert "network connections attempted: 0" in out
    # Every test-case data field survives both DeepEval's LLMTestCase and
    # DSPy's dspy.Example byte for byte: string input stays a string,
    # context / expected_tools (including the empty "no tool" list) and
    # metadata (e.g. TruthfulQA's accepted-answer list) all come back...
    assert "tc.id lossless (24 cases) lossless (24 cases)" in rows
    assert "tc.input lossless (24 cases) lossless (24 cases)" in rows
    assert "tc.expected_output lossless (23 cases) lossless (23 cases)" in rows
    assert "tc.context lossless (1 cases) lossless (1 cases)" in rows
    assert "tc.expected_tools lossless (3 cases) lossless (3 cases)" in rows
    assert "tc.metadata lossless (23 cases) lossless (23 cases)" in rows
    # ...the DeepEval adapter adds only its own documented bookkeeping key...
    assert "tc.metadata.deepeval added to 24/24 cases -" in rows
    # ...and grader definitions still have no slot in either framework.
    assert "tc.graders lossy 24/24 cases lossy 24/24 cases" in rows
    for name in ("deepeval", "dspy"):
        assert (f"{name}: round-trips suite.id, tc.id, tc.input, tc.expected_output, "
                "tc.context, tc.expected_tools, tc.metadata") in out


def test_results_portability():
    out = run_demo("2_results_portability.py")
    rows = lines(out)
    assert "RESULT: PASS" in out
    assert "UNEXPECTED" not in out
    assert "network connections attempted: 0" in out
    assert "valid=True" in out and "valid=False" not in out
    assert sum(1 for r in rows if r.startswith("gsm8k_") and r.endswith(" yes")) == 10
    assert "gsm8k_2 70000 \"$70,000\" 0.0 0.0 0.0 0.0 yes" in rows
    assert ("pass rate: DeepEval ResultSet 0.70 | Haystack ResultSet 0.70 | "
            "Haystack aggregated_report 0.70") in rows
    assert "lossy: results[].grader_results[].reason (10 value(s))" in " ".join(rows)


def test_cross_framework_comparison():
    out = run_demo("3_cross_framework_comparison.py")
    rows = lines(out)
    assert "RESULT: PASS" in out
    assert "network connections attempted: 0" in out
    # the three framework-native ResultSet files (the two suite-grader
    # ResultSets further down are checked separately below)
    assert sum(1 for r in rows if r.startswith("wrote ") and "valid=True" in r) == 3
    assert "valid=False" not in out
    assert "group_id=gsm8k-exact-match-3-frameworks: 3 members joined on test_case_id" in rows
    assert "test case expected output deepeval dspy haystack verdict" in rows
    assert "gsm8k_2 70000 \"$70,000\" fail pass fail DISAGREE" in rows
    assert "gsm8k_3 540 \" 540\\n\" pass pass fail DISAGREE" in rows
    assert "gsm8k_6 260 \"280\" fail fail fail all fail" in rows
    assert "pass rate 0.70 0.90 0.60" in rows
    assert "3/10 test cases get different verdicts from different frameworks on identical outputs:" in rows
    # Opt-in: the suite's own exact_match grader, run inside DeepEval and DSPy
    # through the adapters' grader helpers, gives the same verdict everywhere.
    assert "deepeval valid=True grader_id=gr_exact_match" in rows
    assert "dspy valid=True grader_id=gr_exact_match" in rows
    assert "test case expected output deepeval dspy spec agree" in rows
    assert sum(1 for r in rows if r.startswith("gsm8k_") and r.endswith(" yes")) == 10
    assert "gsm8k_2 70000 \"$70,000\" fail fail fail yes" in rows
    assert "gsm8k_8 45 \"45.\" fail fail fail yes" in rows
    assert "pass rate 0.70 0.70 0.70" in rows


@pytest.mark.parametrize("script", [
    "1_dataset_portability.py", "2_results_portability.py", "3_cross_framework_comparison.py",
])
def test_demo_is_deterministic(script):
    assert run_demo(script) == run_demo(script)
