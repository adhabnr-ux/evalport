#!/usr/bin/env python3
"""Demo 1 -- dataset portability: EvalPort suite -> framework-native objects -> EvalPort.

Takes real suites shipped in this repo -- a 10-case slice of the GSM8K and
TruthfulQA benchmark suites (benchmarks/), plus the RAG and agent-tool example
suites (examples/), which exercise `context` and `expected_tools` -- exports
each one into two real frameworks' native dataset objects through the
adapters in adapters/, converts those objects back to EvalPort, and diffs the
result against the original, field by field:

  * DeepEval -- deepeval_openeval_adapter.from_openeval() -> real
                deepeval.test_case.LLMTestCase objects -> to_openeval()
  * DSPy     -- dspy_openeval_adapter.from_openeval() -> real dspy.Example
                objects -> to_openeval()

Nothing here calls a model or the network (a socket guard enforces that).

Exit status: 0 when every round-tripped suite validates AND every field that
changed is listed in KNOWN_LOSSY for that adapter (each entry says where the
loss comes from). Any other difference -- a field that silently changed and
that nobody documented -- exits 1.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
SLICE = 10  # test cases taken from the front of each benchmark suite
SUITES = [
    "benchmarks/gsm8k/gsm8k.json",
    "benchmarks/truthfulqa/truthfulqa.json",
    "examples/rag-eval-suite.json",
    "examples/agent-tools.json",
]

# --- offline guard -----------------------------------------------------------
# Set before any framework import: no telemetry, no remote model-cost map, and
# any socket connect / DNS lookup raises instead of reaching the network.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
NETWORK_ATTEMPTS: List[str] = []


def _blocked(*args: Any, **kwargs: Any) -> Any:
    NETWORK_ATTEMPTS.append(repr(args[1:] or args)[:80])
    raise OSError("network access is disabled in the EvalPort interop demos")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
socket.getaddrinfo = _blocked  # type: ignore[assignment]

# deepeval before dspy: dspy installs a lazy `openai` module proxy that breaks
# deepeval's own `openai.types` imports if dspy is imported first.
from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402
import dspy  # noqa: E402

import deepeval_openeval_adapter as de_adapter  # noqa: E402
import dspy_openeval_adapter as dspy_adapter  # noqa: E402
from openeval.validate import validate_suite  # noqa: E402

# Fields compared: every TestCase data field in spec/SPEC.md (1. TestCase) and
# the Suite-level fields.
TC_FIELDS = ["id", "input", "expected_output", "context", "retrieval_context",
             "tools_called", "expected_tools", "tags", "graders", "metadata"]
SUITE_FIELDS = ["id", "version", "name", "description", "graders", "config", "metadata"]
FIELDS = [f"suite.{f}" for f in SUITE_FIELDS] + [f"tc.{f}" for f in TC_FIELDS]

# Fields each adapter is KNOWN to change on an EvalPort -> framework -> EvalPort
# round trip, and why. Anything outside this list that changes fails the demo.
KNOWN_LOSSY: Dict[str, Dict[str, str]] = {
    "deepeval": {
        "suite.version": "to_openeval() stamps the SDK's current spec version (1.0.0-rc.5)",
        "suite.name": "a list of LLMTestCase has no suite name; the adapter synthesizes one",
        "suite.graders": "LLMTestCase has no grader slot; one placeholder `custom` grader "
                         "(`gr_deepeval_metrics`) replaces the suite's graders "
                         "(adapter README, 'Design decisions')",
        "suite.description": "a list of LLMTestCase has no suite description",
        "suite.config": "a list of LLMTestCase has no suite config (provider/model settings)",
        "suite.metadata": "a list of LLMTestCase has no suite-level metadata",
        "tc.expected_tools": "an EMPTY list (`[]`, i.e. 'no tool should be called') is dropped: "
                             "to_openeval() skips falsy lists, so the assertion silently becomes "
                             "'no expectation'. Non-empty lists round-trip -- NOT in the README",
        "tc.graders": "every case references the placeholder grader (same reason)",
        "tc.metadata": "from_openeval() never sets LLMTestCase.metadata, so the original "
                       "TestCase.metadata is dropped -- NOT listed in the adapter README",
    },
    "dspy": {
        "suite.version": "to_openeval() stamps the SDK's current spec version (1.0.0-rc.5)",
        "suite.name": "a list of dspy.Example has no suite name",
        "suite.description": "a list of dspy.Example has no suite description",
        "suite.graders": "dspy.Example has no grader slot; placeholder `dspy_metric` grader",
        "suite.config": "a list of dspy.Example has no suite config (provider/model settings)",
        "suite.metadata": "a list of dspy.Example has no suite-level metadata",
        "tc.input": "a string input becomes Example field `input_1`; to_openeval() "
                    "re-flattens it as ['input_1: <text>'] (adapter module docstring)",
        "tc.context": "from_openeval() only maps input + expected_output onto Example "
                      "fields; `context` is dropped",
        "tc.expected_tools": "same: `expected_tools` is dropped",
        "tc.graders": "every case references the placeholder grader",
        "tc.metadata": "replaced by the adapter's own metadata.dspy bookkeeping",
    },
}


def load(rel_path: str) -> Dict[str, Any]:
    suite = json.loads((REPO / rel_path).read_text(encoding="utf-8"))
    suite["test_cases"] = suite["test_cases"][:SLICE]
    assert validate_suite(suite).valid, f"{rel_path} does not validate"
    return suite


def via_deepeval(suite: Dict[str, Any]) -> Dict[str, Any]:
    cases = []
    for kwargs in de_adapter.from_openeval(suite):
        # from_openeval() returns tool *names*; LLMTestCase (deepeval 4.2.6)
        # rejects plain strings with TypeError, so wrap them in ToolCall as the
        # adapter README instructs.
        for key in ("tools_called", "expected_tools"):
            if key in kwargs:
                kwargs[key] = [ToolCall(name=name) for name in kwargs[key]]
        cases.append(LLMTestCase(**kwargs))
    assert all(type(c) is LLMTestCase for c in cases)
    # suite_id is a required argument: a list of LLMTestCase carries no suite id.
    return de_adapter.to_openeval(cases, suite_id=suite["id"])


def via_dspy(suite: Dict[str, Any]) -> Dict[str, Any]:
    examples = dspy_adapter.from_openeval(suite)
    assert all(isinstance(e, dspy.Example) for e in examples)
    ids = [getattr(e, "_openeval_test_case_id") for e in examples]
    return dspy_adapter.to_openeval(examples, input_keys=["input_1"],
                                    expected_key="expected_output", ids=ids,
                                    suite_id=suite["id"])


FRAMEWORKS: List[Tuple[str, str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = [
    ("deepeval", "LLMTestCase", via_deepeval),
    ("dspy", "dspy.Example", via_dspy),
]


def diff(original: Dict[str, Any], back: Dict[str, Any]) -> Dict[str, List[Tuple[str, Any, Any]]]:
    """{field: [(where, before, after), ...]} for every field whose value changed."""
    changes: Dict[str, List[Tuple[str, Any, Any]]] = {}
    for f in SUITE_FIELDS:
        if original.get(f) != back.get(f):
            changes.setdefault(f"suite.{f}", []).append(("suite", original.get(f), back.get(f)))
    back_by_id = {tc["id"]: tc for tc in back.get("test_cases", [])}
    if len(back_by_id) != len(original["test_cases"]):
        changes.setdefault("tc.count", []).append(
            ("suite", len(original["test_cases"]), len(back_by_id)))
    for tc in original["test_cases"]:
        other = back_by_id.get(tc["id"], {})
        for f in TC_FIELDS:
            if tc.get(f) != other.get(f):
                changes.setdefault(f"tc.{f}", []).append((tc["id"], tc.get(f), other.get(f)))
    return changes


def present(original: Dict[str, Any], field: str) -> bool:
    scope, name = field.split(".", 1)
    if scope == "suite":
        return name in original
    return any(name in tc for tc in original["test_cases"])


def short(value: Any, width: int = 58) -> str:
    text = "<absent>" if value is None else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> int:
    failures: List[str] = []
    examples: Dict[Tuple[str, str], Tuple[str, Any, Any]] = {}
    # (framework, field) -> [changed, total]; totals count suites (suite.*) or
    # test cases carrying the field (tc.*), across every suite in SUITES.
    tally: Dict[Tuple[str, str], List[int]] = {}
    print("EvalPort interop demo 1: dataset portability "
          "(EvalPort suite -> framework objects -> EvalPort suite)")
    for rel_path in SUITES:
        original = load(rel_path)
        n = len(original["test_cases"])
        print(f"  source: {rel_path:<40} {original['id']:<18} {n:>2} cases")
        for name, _, roundtrip in FRAMEWORKS:
            back = roundtrip(original)
            valid = validate_suite(back)
            if not valid.valid:
                failures.append(f"{name}/{original['id']}: round-tripped suite invalid: {valid.errors[:3]}")
            changes = diff(original, back)
            for field in FIELDS + ["tc.count"]:
                scope, fname = field.split(".", 1)
                if scope == "suite":
                    total = 1 if fname in original else 0
                else:
                    total = sum(1 for tc in original["test_cases"] if fname in tc)
                changed = changes.get(field, [])
                total = max(total, len(changed))
                if not total:
                    continue
                cell = tally.setdefault((name, field), [0, 0])
                cell[0] += len(changed)
                cell[1] += total
                if changed:
                    examples.setdefault((name, field), changed[0])
                    if field not in KNOWN_LOSSY[name]:
                        failures.append(f"{name}/{original['id']}: undocumented change to {field}")

    unit = {"suite": "suites", "tc": "cases"}
    print(f"\n  {'field':<20}" + "".join(f"{f'via {native}':<24}" for _, native, _ in FRAMEWORKS))
    for field in FIELDS + ["tc.count"]:
        if not any((name, field) in tally for name, _, _ in FRAMEWORKS):
            continue
        row = f"  {field:<20}"
        for name, _, _ in FRAMEWORKS:
            changed, total = tally.get((name, field), (0, 0))
            u = unit[field.split(".")[0]]
            if not total:
                cell = "-"
            elif not changed:
                cell = f"lossless ({total} {u})"
            else:
                label = "lossy" if field in KNOWN_LOSSY[name] else "UNEXPECTED"
                cell = f"{label} {changed}/{total} {u}"
            row += f"{cell:<24}"
        print(row)

    print("\nWhy each changed field changes (first occurrence):")
    for name, _, _ in FRAMEWORKS:
        for field in FIELDS + ["tc.count"]:
            if (name, field) not in examples:
                continue
            why = KNOWN_LOSSY[name].get(field, "UNEXPECTED -- not a documented lossy field")
            where, before, after = examples[(name, field)]
            print(f"  [{name}] {field}: {why}")
            print(f"      {where}: {short(before)} -> {short(after, 40)}")

    print(f"\nnetwork connections attempted: {len(NETWORK_ATTEMPTS)}")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    for name, _, _ in FRAMEWORKS:
        kept = [f for f in FIELDS if (name, f) in tally and not tally[(name, f)][0]]
        print(f"{name}: round-trips {', '.join(kept)}")
    print("RESULT: PASS -- every round-tripped suite validates and every changed field "
          "is a documented lossy field (see 'why' above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
