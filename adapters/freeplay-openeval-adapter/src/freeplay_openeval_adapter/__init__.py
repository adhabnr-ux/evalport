"""
freeplay-openeval-adapter

Convert Freeplay (https://freeplay.ai) Dataset/DatasetTestCase test-case
data, and CompletionTestCase/TraceTestCase test-run data carrying
per-test-case eval_results, to and from EvalPort's TestCase/Suite/ResultSet
schemas (https://github.com/adhabnr-ux/evalport).

Verified against the real installed `freeplay` package (0.6.0) --
freeplay.resources.test_cases.{Dataset, DatasetTestCase} for the suite
side, and freeplay.resources.test_suites.{TestSuiteRun, TestSuiteRunResults}
for the results side -- not the docs. See README.md's "Design notes" for
the two real wrinkles this handles (named-variable input flattening, and
why results_to_openeval() converts recording-time eval_results rather than
the retrieved, unschemad run-level summary).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

__all__ = [
    "to_openeval",
    "from_openeval",
    "results_to_openeval",
    "flatten_inputs",
    "clamp_score",
]

_SPEC_VERSION = "1.0.0"
_PREFERRED_INPUT_KEYS = ("input", "question", "query", "prompt", "text")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` off either a dict or a real Freeplay object's attribute --
    every public entry point accepts either, since Freeplay's own SDK
    objects (DatasetTestCase, CompletionTestCase, TraceTestCase, ...) are
    plain attribute-holders, not dicts, but tests and hand-built callers
    may reasonably want to pass dicts too."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def flatten_inputs(inputs: Mapping[str, Any]) -> str:
    """
    Flatten a Freeplay DatasetTestCase.inputs / CompletionTestCase.variables
    mapping (Mapping[str, str|int|bool|float|dict|list] -- a named-variable
    system, not a single string) into the plain string EvalPort's
    TestCase.input requires.

    Same strategy the parea/vellum/literalai/humanloop adapters in this
    repo use: prefer a recognizable single-input key, fall back to the
    first string-valued entry, then to a stable JSON dump of the whole
    mapping so no data is silently dropped. Raises ValueError only on a
    genuinely empty mapping -- there is nothing to represent.
    """
    if not inputs:
        raise ValueError(
            "flatten_inputs() received an empty inputs/variables mapping -- "
            "nothing to convert into a TestCase.input string"
        )

    for key in _PREFERRED_INPUT_KEYS:
        value = inputs.get(key)
        if isinstance(value, str) and value:
            return value

    for value in inputs.values():
        if isinstance(value, str) and value:
            return value

    return json.dumps(dict(inputs), sort_keys=True, default=str)


def clamp_score(value: Optional[Union[bool, float, int]]) -> Optional[float]:
    """
    Freeplay's per-test-case `eval_results` values passed into
    TestSuiteRun.record()/record_trace() are typed `Union[bool, float]`
    (freeplay/resources/test_suites.py), with no documented range on the
    float side. EvalPort's GraderResult.score must be `null` or in
    `[0, 1]`. Booleans map to the natural 1.0/0.0 pass/fail encoding;
    numbers are clamped into range. The caller-facing conversion functions
    in this module always preserve the raw, unclamped value under the
    spec's own reserved `metadata.openeval.raw_score` key, so clamping
    here never silently loses information.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    raise TypeError(
        f"eval_results value must be bool or float per Freeplay's own "
        f"TestSuiteRun.record() signature, got {type(value).__name__}"
    )


def _default_grader(grader_type: str) -> Dict[str, Any]:
    grader: Dict[str, Any] = {"id": "grader-1", "type": grader_type}
    if grader_type == "llm_judge":
        grader["params"] = {
            "model": "gpt-4o",
            "prompt": "Does the output correctly satisfy the input? "
            "Input: {input}\nOutput: {output}\nExpected: {expected}\n"
            "Answer yes or no.",
        }
    return grader


def _message_to_dict(message: Any) -> Dict[str, Any]:
    """Best-effort, lossless-enough conversion of a Freeplay
    UserMessage/SystemMessage/AssistantMessage (or a plain dict already
    shaped like one) into a plain dict for metadata storage. `content` on
    these classes is either a plain string or a list of typed content-block
    objects (TextBlock/ToolResultBlock/ToolCallBlock/MediaReferenceBlock);
    blocks are captured via their own attribute dict rather than dropped."""
    if isinstance(message, dict):
        return message
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = [
            vars(block) if hasattr(block, "__dict__") else str(block)
            for block in content
        ]
    return {"role": role, "content": content}


def to_openeval(
    dataset: Union[Any, Mapping[str, Any]],
    id: Optional[str] = None,
    name: Optional[str] = None,
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """
    Convert a Freeplay `Dataset` (or a plain dict with `dataset_id`/`id`
    and `test_cases`) into an EvalPort Suite.

    Each `DatasetTestCase.inputs` is a named-variable `Mapping`, not a
    plain string -- flattened via flatten_inputs(). The original mapping
    (and, when present, `history`) is preserved losslessly under
    `metadata.freeplay` so `from_openeval()` can restore it. `output`
    becomes `expected_output` directly (already a plain `Optional[str]`
    on `DatasetTestCase`, no flattening needed).

    Freeplay test cases don't carry their own grader/metric definition --
    metrics are configured on the dataset's downstream test suite, not
    visible on `DatasetTestCase` itself -- so, like literalai-openeval-adapter
    and vellum-openeval-adapter in this repo, a single default grader
    (`llm_judge` unless overridden via `grader_type`) is generated and
    referenced by every test case.
    """
    test_cases_raw = _get(dataset, "test_cases")
    if not test_cases_raw:
        raise ValueError(
            "to_openeval() requires a Dataset (or dict) with at least one "
            "test case in test_cases"
        )

    dataset_id = (
        id or _get(dataset, "dataset_id") or _get(dataset, "id") or str(uuid.uuid4())
    )
    grader = _default_grader(grader_type)

    test_cases: List[Dict[str, Any]] = []
    for tc in test_cases_raw:
        tc_id = _get(tc, "id") or str(uuid.uuid4())
        inputs = _get(tc, "inputs") or {}
        output = _get(tc, "output")
        history = _get(tc, "history")
        raw_metadata = dict(_get(tc, "metadata") or {})

        freeplay_meta: Dict[str, Any] = {"original_inputs": dict(inputs)}
        if history:
            freeplay_meta["history"] = [_message_to_dict(m) for m in history]
        raw_metadata["freeplay"] = freeplay_meta

        entry: Dict[str, Any] = {
            "id": str(tc_id),
            "input": flatten_inputs(inputs),
            "graders": [grader["id"]],
            "metadata": raw_metadata,
        }
        if output is not None:
            entry["expected_output"] = output
        test_cases.append(entry)

    return {
        "version": _SPEC_VERSION,
        "id": str(dataset_id),
        "name": name or f"freeplay-dataset-{dataset_id}",
        "test_cases": test_cases,
        "graders": [grader],
    }


def from_openeval(suite: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert an EvalPort Suite back into a list of dicts shaped as
    `DatasetTestCase(**kwargs)`-ready keyword arguments (`inputs`,
    `output`, `metadata`, `id`).

    When a test case's `metadata.freeplay.original_inputs` is present
    (i.e. it round-trips through this adapter), the original named-variable
    mapping is restored exactly. For a `TestCase` this adapter didn't
    produce, `inputs` falls back to `{"input": test_case["input"]}` --
    the same honest single-key fallback every adapter in this repo uses
    for data it didn't originate.
    """
    items: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        metadata = dict(tc.get("metadata") or {})
        freeplay_meta = metadata.pop("freeplay", {}) if isinstance(metadata.get("freeplay"), dict) else {}

        inputs = freeplay_meta.get("original_inputs")
        if not isinstance(inputs, dict):
            inputs = {"input": tc.get("input")}

        item: Dict[str, Any] = {
            "id": tc.get("id"),
            "inputs": inputs,
            "output": tc.get("expected_output"),
            "metadata": metadata or None,
        }
        if "history" in freeplay_meta:
            item["history"] = freeplay_meta["history"]
        items.append(item)
    return items


def results_to_openeval(
    suite_id: str,
    run_id: str,
    recorded: Iterable[Mapping[str, Any]],
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert per-test-case data recorded via Freeplay's
    `TestSuiteRun.record()` / `.record_trace()` into an EvalPort ResultSet.

    `recorded` is an iterable of dicts, one per test case, each shaped as:
      - "test_case_id": str (required) -- the CompletionTestCase/TraceTestCase id
      - "eval_results": Dict[str, Union[bool, float]] (required) -- exactly
        the shape `TestSuiteRun.record()`'s own `eval_results` parameter
        takes (freeplay/resources/test_suites.py); one entry per named
        evaluator, e.g. {"exact_match": True, "helpfulness": 0.82}.
      - "output": optional str -- the recorded completion/trace output,
        stored in metadata for reference (EvalPort's Result has no native
        "output" field; only pass/fail + grader_results).
      - "passed": optional bool override for the overall Result.passed;
        if omitted, defaults to all(grader "passed" values).

    Why this converts the *recording-time* eval_results and not the shape
    returned by `TestSuiteRun.get_results()`: `TestSuiteRunResults`, as
    actually parsed in freeplay/resources/test_suites.py's
    `_parse_run_results()`, only exposes an aggregate `summary_statistics`
    (`auto_evaluation`/`human_evaluation`/`client_evaluation`, each a bare
    `Dict[str, Any]` passed through with zero schema) plus a top-level
    `eval_results: Optional[Dict[str, Any]]` that's equally unvalidated --
    neither one documents a per-test-case breakdown anywhere in the SDK.
    Building a `Result` per test case from that shape would mean inventing
    a join the SDK doesn't actually provide, rather than converting real
    data. The `eval_results: Dict[str, Union[bool, float]]` passed into
    `record()`/`record_trace()` at recording time IS real, typed, and
    already keyed by the individual test case being recorded -- callers
    already have exactly this in hand at the point they call
    `record()`/`record_trace()` in a loop over a run, and can accumulate it
    into the `recorded` list this function expects. If Freeplay's retrieved
    run-results shape gains real per-test-case structure in a future SDK
    release, that's a natural, separate extension of this function --
    documented here rather than guessed at now.
    """
    recorded = list(recorded)
    if not recorded:
        raise ValueError(
            "results_to_openeval() requires at least one recorded test "
            "case -- EvalPort's ResultSet.results must be non-empty"
        )

    results: List[Dict[str, Any]] = []
    for item in recorded:
        test_case_id = item.get("test_case_id")
        if not test_case_id:
            raise ValueError(
                "results_to_openeval(): every recorded item needs a "
                "'test_case_id'"
            )
        eval_results = item.get("eval_results") or {}

        grader_results: List[Dict[str, Any]] = []
        for grader_name, raw_value in eval_results.items():
            score = clamp_score(raw_value)
            if isinstance(raw_value, bool):
                gr_passed = raw_value
            else:
                gr_passed = score is not None and score >= 0.5
            grader_results.append({
                "grader_id": grader_name,
                "type": "custom",
                "score": score,
                "passed": gr_passed,
                "metadata": {"openeval": {"raw_score": raw_value}},
            })

        passed = item.get("passed")
        if passed is None:
            passed = all(gr["passed"] for gr in grader_results) if grader_results else False

        result_metadata: Dict[str, Any] = {}
        if item.get("output") is not None:
            result_metadata["freeplay"] = {"output": item["output"]}

        results.append({
            "test_case_id": str(test_case_id),
            "passed": bool(passed),
            "grader_results": grader_results,
            "metadata": result_metadata,
        })

    result_set: Dict[str, Any] = {
        "version": _SPEC_VERSION,
        "suite_id": str(suite_id),
        "run_id": str(run_id),
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    return result_set
