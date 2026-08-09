"""LangSmith <-> EvalPort adapter.

Standalone converter between LangSmith (https://smith.langchain.com)
experiment results and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside the
LangSmith SDK itself: it follows the same playbook that already worked for
AutoGen, CrewAI, and Ragas (see ../autogen-openeval-adapter,
../crewai-openeval-adapter, ../ragas-openeval-adapter) — it works against
LangSmith's public `Run` shape (the objects returned by
`Client.list_runs()`, exposing `.inputs` / `.outputs`, plus separately
fetched `Feedback` objects from `Client.list_feedback()`) from the outside,
so you get EvalPort import/export today without needing anything merged
into the langsmith SDK.

Tracked as https://github.com/adhabnr-ux/evalport/issues/2.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"

# Common single-field keys LangSmith runs use for their `inputs`/`outputs`
# dicts across popular chain/prompt shapes. When a run's inputs/outputs dict
# has exactly one of these keys, its value is used directly as the EvalPort
# `input`/`expected_output` string; otherwise the whole dict is serialized
# to JSON so no data is silently dropped.
_INPUT_FIELD_CANDIDATES = ("input", "question", "query", "prompt", "text")
_OUTPUT_FIELD_CANDIDATES = ("output", "answer", "response", "text", "result")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    LangSmith's `Run`/`Feedback` classes (pydantic models) and JSON-loaded
    API output both show up in the wild, so every accessor in this module
    goes through here rather than assuming one shape.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _flatten(value: Any, field_candidates: tuple) -> Optional[str]:
    """Reduce a LangSmith inputs/outputs dict to a single representative string.

    LangSmith run inputs/outputs are free-form dicts (whatever the chain's
    signature happens to be). If a well-known single field is present, use
    its value directly; if the dict has exactly one key, use that value;
    otherwise fall back to a JSON-serialized form of the whole dict so
    nothing is silently dropped.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if not value:
            return None
        for key in field_candidates:
            if key in value and value[key] is not None:
                v = value[key]
                return v if isinstance(v, str) else json.dumps(v)
        if len(value) == 1:
            only = next(iter(value.values()))
            return only if isinstance(only, str) else json.dumps(only)
        return json.dumps(value, sort_keys=True)
    return str(value)


def _feedback_for_run(run: Any, run_id: str, feedback_by_run: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Collect feedback entries for a run from either an attached attribute
    (`run.feedback` / `run["feedback"]`) or an external feedback list
    grouped by run id (passed separately, since LangSmith's
    `Client.list_feedback()` is typically a separate call from
    `Client.list_runs()`)."""
    attached = _get(run, "feedback", None) or []
    normalized = [
        {"key": _get(f, "key"), "score": _get(f, "score"), "comment": _get(f, "comment")}
        for f in attached
        if _get(f, "key") is not None
    ]
    normalized.extend(feedback_by_run.get(run_id, []))
    return normalized


def _run_payload(run: Any, index: int, feedback_by_run: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Normalize a single LangSmith run (+ its feedback) into an EvalPort TestCase dict."""
    run_id = str(_get(run, "id") or _get(run, "run_id") or f"tc_{index}")
    inputs = _get(run, "inputs", None)
    outputs = _get(run, "outputs", None)
    reference = _get(run, "reference_output", None) or _get(run, "expected_output", None)

    input_text = _flatten(inputs, _INPUT_FIELD_CANDIDATES) or ""
    output_text = _flatten(outputs, _OUTPUT_FIELD_CANDIDATES)
    expected_text = _flatten(reference, _OUTPUT_FIELD_CANDIDATES)

    feedback = _feedback_for_run(run, run_id, feedback_by_run)
    feedback_scores = {f["key"]: f["score"] for f in feedback if f.get("score") is not None}
    graders = [f"gr_{key}" for key in sorted(feedback_scores.keys())] or ["gr_langsmith_feedback"]

    tc: Dict[str, Any] = {
        "id": run_id,
        "input": input_text,
        "graders": graders,
    }
    if expected_text is not None:
        tc["expected_output"] = expected_text

    metadata: Dict[str, Any] = {}
    if feedback_scores:
        metadata["langsmith_feedback"] = feedback_scores
    if output_text is not None:
        # The run's actual output belongs on a Result, not a TestCase — kept
        # here as metadata so round-tripping doesn't lose it.
        metadata["langsmith_actual_output"] = output_text
    if metadata:
        tc["metadata"] = metadata
    return tc


def to_openeval(
    runs: Iterable[Any],
    feedback: Optional[Iterable[Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Export LangSmith experiment runs to an EvalPort-shaped suite (dict).

    `runs` is any iterable of LangSmith `Run` objects or dicts (matching
    what `Client.list_runs(project_name=..., is_root=True)` yields), each
    exposing `inputs` / `outputs` and optionally `reference_output` /
    `expected_output`.

    `feedback` is an optional flat iterable of LangSmith `Feedback` objects
    or dicts (matching `Client.list_feedback(run_ids=[...])`), each
    exposing `run_id`, `key`, and `score` — since LangSmith fetches
    feedback via a separate call from runs, this lets you pass both in
    together. A run may also carry feedback directly via a `feedback`
    attribute/key if you've already joined them yourself.

    Every distinct feedback `key` found (e.g. "correctness", "helpfulness")
    becomes its own EvalPort grader (`gr_<key>`, type "custom", handler
    `langsmith:<key>`), and the scores are preserved per test case under
    `metadata.langsmith_feedback` — an experiment run is already-scored
    data, not just a task definition, so nothing is thrown away.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    runs = list(runs)

    feedback_by_run: Dict[str, List[Dict[str, Any]]] = {}
    for f in feedback or []:
        f_run_id = str(_get(f, "run_id"))
        feedback_by_run.setdefault(f_run_id, []).append(
            {"key": _get(f, "key"), "score": _get(f, "score"), "comment": _get(f, "comment")}
        )

    test_cases = [_run_payload(r, i, feedback_by_run) for i, r in enumerate(runs)]

    feedback_keys = sorted(
        {name for tc in test_cases for name in (tc.get("metadata") or {}).get("langsmith_feedback", {}).keys()}
    )
    graders = [
        {
            "id": f"gr_{key}",
            "type": "custom",
            "description": f"LangSmith '{key}' feedback score",
            "params": {"handler": f"langsmith:{key}"},
        }
        for key in feedback_keys
    ]
    if not graders:
        graders = [{"id": "gr_langsmith_feedback", "type": "custom", "params": {"handler": "langsmith:feedback"}}]

    resolved_run_id = run_id or "langsmith_experiment"
    return {
        "version": OPENEVAL_VERSION,
        "id": f"langsmith_eval_{resolved_run_id}",
        "name": f"LangSmith experiment {resolved_run_id}",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "langsmith"}, "langsmith_feedback_keys": feedback_keys},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of LangSmith-shaped example dicts.

    Returns plain dicts with `inputs` / `outputs` keys matching what
    `Client.create_examples()` expects, so you can build a fresh LangSmith
    dataset to re-run an experiment against:

        from langsmith import Client
        from langsmith_openeval_adapter import from_openeval

        examples = from_openeval(suite)
        client = Client()
        dataset = client.create_dataset("my-dataset")
        client.create_examples(dataset_id=dataset.id, examples=examples)
    """
    examples: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        example: Dict[str, Any] = {"inputs": {"input": tc.get("input")}}
        if tc.get("expected_output") is not None:
            example["outputs"] = {"output": tc.get("expected_output")}
        examples.append(example)
    return examples
