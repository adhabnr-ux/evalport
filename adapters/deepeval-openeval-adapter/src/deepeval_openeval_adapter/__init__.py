"""DeepEval <-> EvalPort adapter.

Standalone converter between DeepEval's `LLMTestCase` / `TestResult` /
`MetricData` objects and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than an in-repo DeepEval
change: see https://github.com/confident-ai/deepeval/issues/3067, opened
after reading `LLMTestCase`/`TestResult`/`MetricData` in DeepEval's own
source (`deepeval/test_case/llm_test_case.py`,
`deepeval/evaluate/types.py`, `deepeval/test_run/api.py`) -- the same
"standalone package, zero footprint on the target framework" shape used
by the AutoGen, CrewAI, Giskard, and Guardrails adapters in this ecosystem.

Two independent conversions are provided, mirroring DeepEval's own
two-layer model (test cases you define, and the metric results DeepEval's
`evaluate()` produces from them):

- `to_openeval()` / `from_openeval()` -- `LLMTestCase` objects <-> an
  EvalPort `EvalSuite` (the test cases themselves, pre-run).
- `test_results_to_openeval()` -- `TestResult` objects (DeepEval's
  `evaluate()` output, each carrying a list of `MetricData`) -> an
  EvalPort `ResultSet`.

Mapping, verified against the real, installed `deepeval==4.1.10` source
(not the docs):

| DeepEval field (`LLMTestCase`)      | EvalPort `TestCase` field          |
|--------------------------------------|-------------------------------------|
| `input`                              | `input`                             |
| `expected_output`                    | `expected_output`                   |
| `context`                            | `context`                           |
| `retrieval_context`                  | `retrieval_context`                 |
| `tools_called` (`List[ToolCall]`)    | `tools_called` (tool *names* only)  |
| `expected_tools` (`List[ToolCall]`)  | `expected_tools` (tool *names* only)|
| `tags`                               | `tags`                              |
| `name`, `comments`, `token_cost`, `completion_time`, `flaky`, `multimodal`, full `ToolCall` objects, `metadata` | `metadata["deepeval"]` (no EvalPort `TestCase` field covers these) |

DeepEval's `TestCase` schema fields (`context`, `retrieval_context`,
`tools_called`, `expected_tools`, `tags`) line up with EvalPort's schema
almost one-to-one -- a closer natural fit than most adapters in this
ecosystem need to reach for, since EvalPort's `TestCase` was designed with
exactly this shape (RAG context + agent tool-calls + tags) in mind.

| DeepEval field (`MetricData`, via `TestResult.metrics_data`) | EvalPort `GraderResult` field |
|---|---|
| `name`                | `grader_id` (slug-normalized) |
| `score`                | `score` (clamped to `[0, 1]`) |
| `success`              | `passed`                      |
| `reason`               | `reason`                      |
| `threshold`, `strict_mode`, `evaluation_model`, `error`, `evaluation_cost`, `input_tokens`, `output_tokens` | `metadata` |
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "test_results_to_openeval", "__version__"]
__version__ = "0.1.0"

# LLMTestCase fields this adapter reads explicitly. Anything else present on
# the object (a genuinely new field added by a future deepeval release) is
# simply not seen -- there's no dict of "leftover" fields to fall back to
# since LLMTestCase is a real pydantic model, not a schema-less dict like
# Opik's DatasetItem. If deepeval adds a field this adapter should carry,
# that's a version-gated update to this list, not silent data loss of
# something this adapter never claimed to read.
_KNOWN_TESTCASE_FIELDS = (
    "input", "actual_output", "expected_output", "context", "retrieval_context",
    "metadata", "tools_called", "comments", "expected_tools", "token_cost",
    "completion_time", "flaky", "multimodal", "name", "tags",
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _clamp01(value: Optional[float]) -> Optional[float]:
    """Clamp a score into EvalPort's required [0, 1] range.

    DeepEval's built-in metrics (AnswerRelevancy, Faithfulness, GEval, ...)
    are documented as 0-1, but MetricData.score is a plain
    Optional[float] with no enforced bound -- a custom or community metric
    could return anything. EvalPort's schema requires score in [0, 1]
    (or null), so this is a real, documented lossy step for any metric
    that scores outside that range, not an assumption that all metrics do.
    """
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _slugify(name: str) -> str:
    """Normalize a DeepEval metric name ("Answer Relevancy") into a grader_id
    ("answer_relevancy"), matching the normalization every other adapter in
    this ecosystem applies to human-readable metric/feedback names."""
    return "_".join(name.strip().lower().replace("-", " ").split())


def _stringify_context_item(item: Any) -> str:
    """A `retrieval_context` entry is either a plain str or a
    `RetrievedContextData` (`context`, `source`). Its own `model_serializer`
    renders as `f"{source}: {context}"` -- matched here exactly so a value
    printed by DeepEval itself and a value round-tripped through this
    adapter read identically."""
    if isinstance(item, str):
        return item
    source = _get(item, "source")
    context = _get(item, "context")
    if source is not None and context is not None:
        return f"{source}: {context}"
    return str(item)


def _tool_call_name(tool_call: Any) -> Optional[str]:
    if isinstance(tool_call, str):
        return tool_call
    return _get(tool_call, "name")


def _tool_call_to_dict(tool_call: Any) -> Dict[str, Any]:
    """Serialize a full ToolCall (name, type, description, reasoning, output,
    input_parameters) for metadata preservation -- EvalPort's TestCase only
    has room for tool *names*, so the rest lives in metadata rather than
    being silently dropped."""
    if isinstance(tool_call, dict):
        return dict(tool_call)
    if hasattr(tool_call, "model_dump"):
        d = tool_call.model_dump()
        # ToolCallType is an Enum; model_dump's field_serializer already
        # renders it as a plain string, but guard defensively for any
        # Enum that slips through un-serialized.
        if "type" in d and hasattr(d["type"], "value"):
            d["type"] = d["type"].value
        return d
    return {"name": _get(tool_call, "name")}


def to_openeval(
    test_cases: Sequence[Any],
    *,
    suite_id: str,
    name: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    grader_id: str = "gr_deepeval_metrics",
    grader_handler: str = "deepeval:metrics",
) -> Dict[str, Any]:
    """Export DeepEval `LLMTestCase` objects to an EvalPort-shaped suite (dict).

    `test_cases` is any iterable of `deepeval.test_case.LLMTestCase`
    instances (or equivalent plain dicts with the same field names).

    DeepEval's `LLMTestCase` has no public unique-identifier field (only an
    optional, human-chosen `name` and a private `_identifier` UUID that
    DeepEval itself does not propagate into `TestResult` either -- verified
    by reading `deepeval/evaluate/types.py`). So test case IDs are, in
    order of precedence: an explicit `ids[i]` if you pass one, else
    `test_case.name` if set, else an auto `tc_{i}`. Pass the *same* `ids`
    list to `test_results_to_openeval()` (or rely on the same
    name/index-based defaulting) to keep results correlated to the suite
    that produced them -- DeepEval's own `evaluate()` correlates results to
    test cases positionally (via `TestResult.index`), which is exactly the
    fallback this adapter uses.

    DeepEval doesn't attach specific metrics to a `LLMTestCase` up front --
    which metrics run is chosen separately, at `evaluate()` time. So every
    test case in the exported suite references one placeholder `custom`
    grader (`grader_id`/`grader_handler`, overridable) rather than guessing
    which of DeepEval's dozens of metrics you intend to run; the real,
    per-metric grading shows up honestly in `test_results_to_openeval()`'s
    output instead, once metrics have actually executed.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance.
    """
    test_case_list = list(test_cases)
    resolved_ids: List[str] = []
    for i, tc in enumerate(test_case_list):
        if ids is not None:
            resolved_ids.append(ids[i])
        else:
            tc_name = _get(tc, "name")
            resolved_ids.append(tc_name if tc_name else f"tc_{i}")

    test_case_dicts: List[Dict[str, Any]] = []
    for i, (tc, tc_id) in enumerate(zip(test_case_list, resolved_ids)):
        input_value = _get(tc, "input")
        if not input_value:
            raise ValueError(
                f"LLMTestCase at index {i} (id={tc_id!r}) has no `input` -- "
                "EvalPort's TestCase.input is required and non-empty."
            )

        ec: Dict[str, Any] = {"id": tc_id, "input": input_value, "graders": [grader_id]}

        expected_output = _get(tc, "expected_output")
        if expected_output is not None:
            ec["expected_output"] = expected_output

        context = _get(tc, "context")
        if context:
            ec["context"] = [_stringify_context_item(c) for c in context]

        retrieval_context = _get(tc, "retrieval_context")
        if retrieval_context:
            ec["retrieval_context"] = [_stringify_context_item(c) for c in retrieval_context]

        tools_called = _get(tc, "tools_called")
        if tools_called:
            ec["tools_called"] = [n for n in (_tool_call_name(t) for t in tools_called) if n]

        expected_tools = _get(tc, "expected_tools")
        if expected_tools:
            ec["expected_tools"] = [n for n in (_tool_call_name(t) for t in expected_tools) if n]

        tags = _get(tc, "tags")
        if tags:
            ec["tags"] = list(tags)

        metadata: Dict[str, Any] = {}
        user_metadata = _get(tc, "metadata")
        if user_metadata:
            metadata.update(dict(user_metadata))

        deepeval_meta: Dict[str, Any] = {}
        for field in ("actual_output", "comments", "token_cost", "completion_time",
                      "flaky", "multimodal", "name"):
            value = _get(tc, field)
            if value not in (None, False):
                deepeval_meta[field] = value
        if tools_called:
            deepeval_meta["tools_called_full"] = [_tool_call_to_dict(t) for t in tools_called]
        if expected_tools:
            deepeval_meta["expected_tools_full"] = [_tool_call_to_dict(t) for t in expected_tools]
        identifier = _get(tc, "_identifier")
        if identifier:
            deepeval_meta["identifier"] = identifier
        if deepeval_meta:
            metadata["deepeval"] = deepeval_meta

        if metadata:
            ec["metadata"] = metadata

        test_case_dicts.append(ec)

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name or f"DeepEval test cases ({suite_id})",
        "test_cases": test_case_dicts,
        "graders": [{
            "id": grader_id,
            "type": "custom",
            "params": {"handler": grader_handler},
            "description": (
                "Placeholder: DeepEval metrics are chosen at evaluate() time, "
                "not attached to a test case up front. See test_results_to_openeval() "
                "for the real, per-metric grader results once metrics have run."
            ),
        }],
        "metadata": {"openeval": {"source": "deepeval"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of DeepEval-constructible test case dicts.

    Each returned dict is keyed exactly like `LLMTestCase`'s constructor
    kwargs (`input`, `expected_output`, `context`, `retrieval_context`,
    `tools_called`, `expected_tools`, `tags`, `name`) so callers can do
    `LLMTestCase(**d)` directly -- verified against the real constructor
    signature, not guessed. `tools_called`/`expected_tools` come back as
    plain tool-name strings (EvalPort doesn't carry full `ToolCall` detail
    at the suite level); construct real `ToolCall(name=...)` objects from
    them if `LLMTestCase` requires `ToolCall` instances rather than dicts in
    your installed deepeval version.

    Multi-turn `input` (a list of strings) is rejected outright: DeepEval's
    `LLMTestCase.input` is `str` only (multi-turn lives in a separate
    `ConversationalTestCase.turns`, out of scope for this adapter -- see the
    README for why).
    """
    test_cases: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        input_value = tc.get("input")
        if isinstance(input_value, list):
            raise ValueError(
                f"Test case {tc.get('id')!r} has multi-turn `input` (a list); "
                "DeepEval's LLMTestCase.input is single-turn (str) only. "
                "See ConversationalTestCase for DeepEval's multi-turn shape, "
                "which this adapter does not cover."
            )

        item: Dict[str, Any] = {"input": input_value}
        if "expected_output" in tc:
            item["expected_output"] = tc["expected_output"]
        if "context" in tc:
            item["context"] = list(tc["context"])
        if "retrieval_context" in tc:
            item["retrieval_context"] = list(tc["retrieval_context"])
        if "tools_called" in tc:
            item["tools_called"] = list(tc["tools_called"])
        if "expected_tools" in tc:
            item["expected_tools"] = list(tc["expected_tools"])
        if "tags" in tc:
            item["tags"] = list(tc["tags"])

        metadata = tc.get("metadata") or {}
        deepeval_meta = dict(metadata.get("deepeval") or {})
        if "name" in deepeval_meta:
            item["name"] = deepeval_meta["name"]
        elif tc.get("id"):
            # No original DeepEval `name` recorded (this suite wasn't
            # produced by to_openeval()) -- fall back to the EvalPort id so
            # round-tripped test cases stay identifiable, not anonymous.
            item["name"] = tc["id"]
        for field in ("comments", "token_cost", "completion_time", "flaky", "multimodal"):
            if field in deepeval_meta:
                item[field] = deepeval_meta[field]

        test_cases.append(item)
    return test_cases


def test_results_to_openeval(
    test_results: Any,
    *,
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    runner_name: str = "deepeval",
    runner_version: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Export DeepEval evaluation results to an EvalPort ResultSet.

    `test_results` is either a `deepeval.evaluate.types.EvaluationResult`
    (what `deepeval.evaluate()` returns -- this reads its `.test_results`
    attribute) or a plain iterable of `TestResult` objects/equivalent dicts
    directly.

    Each `TestResult.metrics_data` entry (a `MetricData` -- `name`, `score`,
    `success`, `reason`, `threshold`, ...) becomes one EvalPort
    `GraderResult`, `type="custom"` (DeepEval metrics are LLM-judge-backed,
    rubric-based, or arbitrary Python callables depending on which metric
    class ran -- there's no single EvalPort built-in type that's honest for
    all of them, the same reasoning the Giskard and Opik adapters use for
    their own framework-defined checks). `score` is clamped to EvalPort's
    required `[0, 1]`; `threshold`, `strict_mode`, `evaluation_model`,
    `error`, `evaluation_cost`, `input_tokens`, `output_tokens` are
    preserved under the grader result's `metadata` rather than dropped.

    `test_case_id` correlation: a `TestResult` carries `name` and `index`
    but no direct link back to whatever id `to_openeval()` assigned. Pass
    the *same* `ids` list you gave `to_openeval()` (matched positionally
    against `test_results`, mirroring how DeepEval's own `TestResult.index`
    already tracks position) to recover exact correlation; if omitted, this
    falls back to `TestResult.name` if set, else `tc_{index}` -- the same
    two-step fallback `to_openeval()` itself uses, so the defaults line up
    automatically when neither side passes explicit ids.

    A `TestResult` with no `metrics_data` at all (DeepEval logs this rather
    than raising when a metric errors out) produces a `runner_error`, not a
    silent empty pass.

    `started_at` defaults to the current UTC time in ISO 8601 if omitted.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass
    it to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    if started_at is None:
        from datetime import datetime, timezone
        started_at = datetime.now(timezone.utc).isoformat()

    raw_results = _get(test_results, "test_results", test_results)
    raw_results = list(raw_results)

    results: List[Dict[str, Any]] = []
    for i, tr in enumerate(raw_results):
        if ids is not None:
            test_case_id = ids[i]
        else:
            tr_name = _get(tr, "name")
            test_case_id = tr_name if tr_name else f"tc_{i}"

        result: Dict[str, Any] = {"test_case_id": str(test_case_id)}

        actual_output = _get(tr, "actual_output")
        if isinstance(actual_output, str):
            result["actual_output"] = actual_output
        elif actual_output is not None:
            # A multimodal actual_output is List[Union[str, MLLMImage]] --
            # EvalPort's actual_output is a plain string, so join the text
            # pieces and note the image placeholders rather than raising,
            # since this is a real (if unusual) DeepEval shape to support.
            result["actual_output"] = "".join(str(p) for p in actual_output)

        metrics_data = _get(tr, "metrics_data")
        if not metrics_data:
            result["grader_results"] = []
            result["passed"] = False
            result["error"] = {
                "type": "runner_error",
                "message": "TestResult has no metrics_data (metric evaluation produced no results).",
            }
            results.append(result)
            continue

        grader_results: List[Dict[str, Any]] = []
        for md in metrics_data:
            md_name = _get(md, "name") or "unknown_metric"
            score = _clamp01(_get(md, "score"))
            success = _get(md, "success")
            gr: Dict[str, Any] = {
                "grader_id": _slugify(md_name),
                "type": "custom",
                "score": score,
                "passed": bool(success) if success is not None else (score is not None and score >= 0.5),
            }
            reason = _get(md, "reason")
            if reason:
                gr["reason"] = reason

            gr_metadata: Dict[str, Any] = {"metric_name": md_name}
            for field in ("threshold", "strict_mode", "evaluation_model", "error",
                          "evaluation_cost", "input_tokens", "output_tokens"):
                value = _get(md, field)
                if value is not None:
                    gr_metadata[field] = value
            gr["metadata"] = gr_metadata

            grader_results.append(gr)

        result["grader_results"] = grader_results
        overall_success = _get(tr, "success")
        result["passed"] = bool(overall_success) if overall_success is not None else all(
            g["passed"] for g in grader_results
        )

        tr_metadata: Dict[str, Any] = {}
        for field in ("index", "conversational", "multimodal"):
            value = _get(tr, field)
            if value is not None:
                tr_metadata[field] = value
        extra_meta = _get(tr, "metadata")
        if extra_meta:
            tr_metadata["user_metadata"] = dict(extra_meta)
        if tr_metadata:
            result["metadata"] = {"deepeval": tr_metadata}

        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    scores = [
        g["score"]
        for r in results
        for g in r.get("grader_results", [])
        if g.get("score") is not None
    ]
    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped": 0,
        "pass_rate": (passed / total) if total else 0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0,
    }

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at or started_at,
        "runner": {"name": runner_name, "version": runner_version or __version__},
        "results": results,
        "summary": summary,
        "metadata": {"openeval": {"source": "deepeval"}},
    }
