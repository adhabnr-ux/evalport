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

Plus an opt-in grader mapping, `graders_to_deepeval_metrics()`, which turns
an EvalPort `exact_match` grader into a real DeepEval metric with the same
semantics (other grader types have no faithful DeepEval equivalent and are
refused with the reason -- see its docstring).

Mapping, verified against the real, installed `deepeval` source (4.1.10,
re-checked against 4.2.6), not the docs:

| DeepEval field (`LLMTestCase`)      | EvalPort `TestCase` field          |
|--------------------------------------|-------------------------------------|
| `input`                              | `input`                             |
| `expected_output`                    | `expected_output`                   |
| `context`                            | `context`                           |
| `retrieval_context`                  | `retrieval_context`                 |
| `tools_called` (`List[ToolCall]`)    | `tools_called` (tool *names* only)  |
| `expected_tools` (`List[ToolCall]`)  | `expected_tools` (tool *names* only)|
| `tags`                               | `tags`                              |
| `metadata`                           | `metadata` (key for key; the `"deepeval"` key is reserved) |
| `name`, `comments`, `token_cost`, `completion_time`, `flaky`, `multimodal`, full `ToolCall` objects | `metadata["deepeval"]` (no EvalPort `TestCase` field covers these) |

Empty lists are kept as empty lists in both directions: `expected_tools=[]`
("no tool should be called") is a different assertion from an absent
`expected_tools` ("no expectation").

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

import copy
import warnings
from typing import Any, Dict, List, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval", "from_openeval", "test_results_to_openeval",
    "to_tool_calls", "graders_to_deepeval_metrics", "exact_match", "__version__",
]
__version__ = "0.2.0"

# TestCase.metadata key reserved for this adapter's own bookkeeping.
_NAMESPACE = "deepeval"
_TOOL_CALL_MODES = ("auto", "objects", "names")

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

        # List fields are emitted whenever they are set (`is not None`), so an
        # empty list survives as an empty list. That matters most for
        # `expected_tools=[]`, which asserts "no tool should be called" -- a
        # different claim from an absent `expected_tools` ("no expectation").
        context = _get(tc, "context")
        if context is not None:
            ec["context"] = [_stringify_context_item(c) for c in context]

        retrieval_context = _get(tc, "retrieval_context")
        if retrieval_context is not None:
            ec["retrieval_context"] = [_stringify_context_item(c) for c in retrieval_context]

        tools_called = _get(tc, "tools_called")
        if tools_called is not None:
            ec["tools_called"] = [n for n in (_tool_call_name(t) for t in tools_called) if n]

        expected_tools = _get(tc, "expected_tools")
        if expected_tools is not None:
            ec["expected_tools"] = [n for n in (_tool_call_name(t) for t in expected_tools) if n]

        tags = _get(tc, "tags")
        if tags is not None:
            ec["tags"] = list(tags)

        # `LLMTestCase.metadata` maps onto `TestCase.metadata` key for key;
        # `from_openeval()` is the exact inverse. The one reserved key is
        # "deepeval" (this adapter's own bookkeeping namespace), so a user
        # metadata key that happens to be called "deepeval" is parked inside
        # that namespace instead of being overwritten by it.
        metadata: Dict[str, Any] = {}
        deepeval_meta: Dict[str, Any] = {}
        user_metadata = _get(tc, "metadata")
        if user_metadata:
            user_metadata = copy.deepcopy(dict(user_metadata))
            if _NAMESPACE in user_metadata:
                deepeval_meta["metadata_deepeval"] = user_metadata.pop(_NAMESPACE)
            metadata.update(user_metadata)

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
            metadata[_NAMESPACE] = deepeval_meta

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


def _load_tool_call_class() -> Any:
    """Return `deepeval.test_case.ToolCall`, or None when deepeval isn't installed.

    Imported lazily so the adapter itself keeps working (and its framework-free
    tests keep running) without deepeval installed.
    """
    try:
        from deepeval.test_case import ToolCall
    except ModuleNotFoundError as e:
        # Only "deepeval isn't installed" means fall back to names. A broken
        # install (e.g. deepeval imported after dspy in one process, which
        # fails inside openai) should surface, not silently change the output.
        if (e.name or "").split(".")[0] == "deepeval":
            return None
        raise
    return ToolCall


def to_tool_calls(tools: Optional[Sequence[Any]]) -> Optional[List[Any]]:
    """Wrap tool names (or `ToolCall`-shaped dicts) as `deepeval.test_case.ToolCall`.

    `LLMTestCase.tools_called` / `expected_tools` are typed
    `Optional[List[ToolCall]]`, and deepeval 4.x rejects bare strings with a
    `TypeError`. `from_openeval()` already does this wrapping when deepeval is
    importable; use this helper on its output when you called it with
    `tool_calls="names"` (or in an environment without deepeval, then
    construct the `LLMTestCase` elsewhere). `None` stays `None` and `[]`
    stays `[]`. Existing `ToolCall` objects pass through unchanged.

    Raises ImportError if deepeval isn't installed.
    """
    if tools is None:
        return None
    tool_call_cls = _load_tool_call_class()
    if tool_call_cls is None:
        raise ImportError(
            "to_tool_calls() needs the 'deepeval' package: "
            "pip install 'deepeval-openeval-adapter[deepeval]'"
        )
    wrapped: List[Any] = []
    for t in tools:
        if isinstance(t, tool_call_cls):
            wrapped.append(t)
        elif isinstance(t, str):
            wrapped.append(tool_call_cls(name=t))
        else:
            wrapped.append(tool_call_cls(**dict(t)))
    return wrapped


def _tool_specs(names: List[str], full: Any) -> List[Any]:
    """Pair EvalPort tool names with the full `ToolCall` detail `to_openeval()`
    recorded under `metadata.deepeval.<field>_full`, when that detail still
    matches the names one for one (i.e. nobody edited the name list since).
    Otherwise fall back to name-only specs."""
    if isinstance(full, list) and [_get(f, "name") for f in full] == names:
        return [dict(f) for f in full]
    return [{"name": n} for n in names]


def from_openeval(suite: Dict[str, Any], *, tool_calls: str = "auto") -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of `LLMTestCase` constructor kwargs.

    Each returned dict is keyed exactly like `LLMTestCase`'s constructor
    kwargs (`input`, `expected_output`, `context`, `retrieval_context`,
    `tools_called`, `expected_tools`, `tags`, `metadata`, `name`, ...), so
    `LLMTestCase(**d)` works directly -- including for test cases with tools.

    - `metadata`: the EvalPort `TestCase.metadata`, deep-copied, minus the
      adapter's own reserved `"deepeval"` namespace. It becomes
      `LLMTestCase.metadata` (so e.g. TruthfulQA's `truthfulqa_all_correct`
      answer list is available to a DeepEval metric), and `to_openeval()`
      writes it back key for key.
    - `tools_called` / `expected_tools`: an empty list stays an empty list
      (`expected_tools: []` means "no tool should be called"); an absent
      field stays absent. How the entries are returned depends on
      `tool_calls`:

      * `"auto"` (default): `deepeval.test_case.ToolCall` objects when
        deepeval is importable -- restoring full `ToolCall` detail recorded
        by `to_openeval()` under `metadata.deepeval.*_full` when it still
        matches the names -- else plain name strings.
      * `"objects"`: always `ToolCall` objects (ImportError without deepeval).
      * `"names"`: plain name strings (the 0.1.x return shape); wrap them
        with `to_tool_calls()` before constructing an `LLMTestCase`.

    Multi-turn `input` (a list of strings) is rejected outright: DeepEval's
    `LLMTestCase.input` is `str` only (multi-turn lives in a separate
    `ConversationalTestCase.turns`, out of scope for this adapter -- see the
    README for why).
    """
    if tool_calls not in _TOOL_CALL_MODES:
        raise ValueError(f"tool_calls must be one of {_TOOL_CALL_MODES}, got {tool_calls!r}")
    wrap = tool_calls == "objects" or (tool_calls == "auto" and _load_tool_call_class() is not None)

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

        metadata = copy.deepcopy(tc.get("metadata") or {})
        deepeval_meta = metadata.pop(_NAMESPACE, None) or {}
        if "metadata_deepeval" in deepeval_meta:
            metadata[_NAMESPACE] = deepeval_meta["metadata_deepeval"]

        item: Dict[str, Any] = {"input": input_value}
        if "expected_output" in tc:
            item["expected_output"] = tc["expected_output"]
        if "context" in tc:
            item["context"] = list(tc["context"])
        if "retrieval_context" in tc:
            item["retrieval_context"] = list(tc["retrieval_context"])
        for field in ("tools_called", "expected_tools"):
            if field not in tc:
                continue
            names = list(tc[field])
            if wrap:
                item[field] = to_tool_calls(_tool_specs(names, deepeval_meta.get(f"{field}_full")))
            else:
                item[field] = names
        if "tags" in tc:
            item["tags"] = list(tc["tags"])
        if metadata:
            item["metadata"] = metadata

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


# ---------------------------------------------------------------------------
# Graders: opt-in, faithful EvalPort -> DeepEval metric mapping
# ---------------------------------------------------------------------------

# Params spec/SPEC.md defines for `exact_match` (both optional).
_EXACT_MATCH_PARAMS = ("ignore_case", "trim_whitespace")

# Why each other well-known grader type gets no DeepEval metric here. A
# mapping is only offered when a DeepEval metric computes exactly what the
# EvalPort grader definition says; "roughly similar" is not a mapping.
_UNSUPPORTED_GRADER_REASONS = {
    "semantic_similarity": "the score depends on the embedding model the runner picks; DeepEval "
                           "has no deterministic embedding-cosine metric to pin it to",
    "llm_judge": "DeepEval's GEval builds its own evaluation prompt around your criteria, so it "
                 "would not run the grader's `prompt` verbatim",
    "model graded": "alias of llm_judge (same reason)",
}


def exact_match(
    actual_output: Optional[str],
    expected_output: Optional[str],
    *,
    ignore_case: bool = False,
    trim_whitespace: bool = True,
) -> bool:
    """EvalPort `exact_match` semantics (spec/SPEC.md grader table).

    Mirrors the reference runner (`cli/src/run/graders/tier1.ts`,
    `gradeExactMatch`) step for step: trim both sides when
    `trim_whitespace` (default true), lowercase both sides when
    `ignore_case` (default false), then compare with `==`. No other
    normalization (punctuation, articles, number formatting) is applied.
    A missing `expected_output` compares as `""`, as in the reference.

    One documented difference: Python's `str.strip()`/`str.lower()` are
    used where the TypeScript reference uses `String.trim()`/
    `toLowerCase()`. Their whitespace sets differ on exactly six code
    points: U+FEFF is trimmed by JS only; U+001C-U+001F and U+0085 are
    stripped by Python only. Lowercasing follows each language's Unicode
    case tables.
    """
    a = "" if actual_output is None else str(actual_output)
    e = "" if expected_output is None else str(expected_output)
    if trim_whitespace:
        a, e = a.strip(), e.strip()
    if ignore_case:
        a, e = a.lower(), e.lower()
    return a == e


def _exact_match_params(grader: Dict[str, Any]) -> Dict[str, bool]:
    params = dict(grader.get("params") or {})
    unknown = sorted(k for k in params if k not in _EXACT_MATCH_PARAMS)
    if unknown:
        warnings.warn(
            f"grader {grader.get('id')!r}: exact_match param(s) {unknown} are not defined by the "
            f"EvalPort spec (which defines {list(_EXACT_MATCH_PARAMS)}) and are ignored, as the "
            "reference runner ignores them",
            UserWarning,
            stacklevel=3,
        )
    return {
        "ignore_case": params.get("ignore_case") is True,
        "trim_whitespace": params.get("trim_whitespace") is not False,
    }


_EXACT_MATCH_METRIC_CLASS: Any = None


def _exact_match_metric_class() -> Any:
    """Build (once) a real `deepeval.metrics.BaseMetric` subclass for
    EvalPort `exact_match`. Built lazily so importing this adapter never
    requires deepeval."""
    global _EXACT_MATCH_METRIC_CLASS
    if _EXACT_MATCH_METRIC_CLASS is not None:
        return _EXACT_MATCH_METRIC_CLASS
    try:
        from deepeval.metrics import BaseMetric
    except ImportError as e:
        raise ImportError(
            "graders_to_deepeval_metrics() needs the 'deepeval' package: "
            "pip install 'deepeval-openeval-adapter[deepeval]'"
        ) from e

    class EvalPortExactMatchMetric(BaseMetric):
        """EvalPort `exact_match`, run as a DeepEval metric.

        Deterministic, no model calls. Score 1.0 on match, 0.0 otherwise;
        threshold 1.0, so `success` is the match itself. Its name is the
        EvalPort grader id, so `test_results_to_openeval()` reports results
        under that same grader id.
        """

        _required_params: List[Any] = []

        def __init__(self, grader_id: str, *, ignore_case: bool = False,
                     trim_whitespace: bool = True) -> None:
            self.grader_id = grader_id
            self.ignore_case = ignore_case
            self.trim_whitespace = trim_whitespace
            self.threshold = 1.0
            self.async_mode = False
            self.verbose_mode = False
            self.include_reason = True
            self.strict_mode = False

        def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
            matched = exact_match(
                test_case.actual_output, test_case.expected_output,
                ignore_case=self.ignore_case, trim_whitespace=self.trim_whitespace,
            )
            self.score = 1.0 if matched else 0.0
            self.reason = (
                "exact match" if matched else
                f"expected {test_case.expected_output!r}, got {test_case.actual_output!r} "
                f"(ignore_case={self.ignore_case}, trim_whitespace={self.trim_whitespace})"
            )
            self.success = self.is_successful()
            return self.score

        async def a_measure(self, test_case: Any, *args: Any, **kwargs: Any) -> float:
            return self.measure(test_case)

        def is_successful(self) -> bool:
            return self.score is not None and self.score >= self.threshold

        @property
        def __name__(self) -> str:  # DeepEval reads a metric's name from here
            return self.grader_id

    _EXACT_MATCH_METRIC_CLASS = EvalPortExactMatchMetric
    return _EXACT_MATCH_METRIC_CLASS


def graders_to_deepeval_metrics(
    suite_or_graders: Any, *, skip_unsupported: bool = False,
) -> Dict[str, Any]:
    """Opt-in: map an EvalPort suite's grader definitions to DeepEval metrics.

    Returns `{grader_id: metric}` for every grader with a faithful DeepEval
    equivalent, ready for `deepeval.evaluate(test_cases, metrics=[...])`.
    Pass the suite dict (its `graders` list is read) or a list of grader
    dicts. Inline grader objects inside `TestCase.graders` are not read.

    Faithful mappings (the DeepEval metric computes exactly what the EvalPort
    grader defines):

    - `exact_match` -> `EvalPortExactMatchMetric` (a `BaseMetric` subclass;
      see `exact_match()` for the semantics). DeepEval's own
      `ExactMatchMetric` is *not* used: it always strips and is always
      case-sensitive, so it can't express `ignore_case: true` or
      `trim_whitespace: false`.

    Any other type raises `ValueError` naming the grader and why there is no
    faithful mapping, unless `skip_unsupported=True`, in which case it is
    left out of the returned dict (compare the keys with the suite's grader
    ids to see what was skipped). Unknown `exact_match` params (e.g. `strip`,
    which is not a spec param) trigger a `UserWarning` and are ignored, as
    the reference runner ignores them.

    DeepEval applies every metric to every test case passed to
    `evaluate()`; to honour per-test-case `graders` lists, group test cases
    by the grader ids they reference and evaluate each group with its
    metrics.
    """
    graders = suite_or_graders.get("graders", []) if isinstance(suite_or_graders, dict) \
        else list(suite_or_graders)
    metrics: Dict[str, Any] = {}
    for grader in graders:
        gtype = grader.get("type")
        if gtype == "exact_match":
            cls = _exact_match_metric_class()
            metrics[grader["id"]] = cls(grader["id"], **_exact_match_params(grader))
            continue
        if skip_unsupported:
            continue
        reason = _UNSUPPORTED_GRADER_REASONS.get(
            gtype, "this adapter maps only exact_match, and no DeepEval built-in metric "
                   "computes exactly this grader type")
        raise ValueError(
            f"grader {grader.get('id')!r} (type {gtype!r}) has no faithful DeepEval metric: "
            f"{reason}. Pass skip_unsupported=True to map only the supported graders."
        )
    return metrics
