"""luml <-> EvalPort adapter.

Standalone converter between luml's (https://github.com/luml-ai/luml)
public `EvalItem` / `EvalResult` / `EvalResults` evaluation types
(`luml.experiments.evaluation.types`) and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside luml
itself: a proposal to add a `luml-openeval-adapter` was opened on
https://github.com/luml-ai/luml/issues/632 with a full mapping sketch.
Maintainer OKUA1 (Oleh Kostromin) closed it as complete with: "You're of
course free to implement and maintain any adapter you find useful in your
own repository. Since this doesn't require any changes on our side, I'll
close this issue." This package is that adapter, built exactly as promised
in the follow-up comment on that issue -- against the real, public
`EvalItem`/`EvalResult`/`EvalResults` dataclasses, not the original sketch's
guesses.

Ground truth this module was written against (read directly from
luml-ai/luml's `main` branch, not guessed):
  - sdk/python/sdk/luml/experiments/evaluation/types.py
    (`EvalItem`, `EvalResult`, `EvalResults`, `REASONING_SUFFIX`)
  - sdk/python/sdk/luml/experiments/evaluation/evaluate.py
    (`evaluate()`, `_evaluate_single_item()`, `_aggregate_scores()` -- the
    real shape of what ends up inside `EvalResult.scores`: plain numeric/
    bool scorer outputs, `"<scorer>_reasoning"` companions for LLM-judge
    scorers, `"error"` on a whole-item failure, `"__error__<scorer>"` on a
    single failed scorer inside an otherwise-successful item)
  - sdk/python/sdk/luml/experiments/evaluation/scorers/base.py
    (`BaseScorer`/`SupervisedScorer`/`UnsupervisedScorer` -- confirms luml's
    scorers carry no category/kind beyond a name, unlike e.g. AgentEval's
    typed assertions)
  - sdk/python/sdk/luml/experiments/evaluation/scorers/builtin/_base.py
    (`LLMJudgeScorer.parse_judgment()` / `SupervisedLLMJudgeScorer.
    parse_judgment()` -- confirms the exact `{name: score, f"{name}
    {REASONING_SUFFIX}": reasoning}` shape every one of luml's five builtin
    scorers -- completeness, correctness, prompt_alignment, relevancy,
    summarization -- actually produces)

Why duck-typed rather than importing luml directly: luml's real Python
package is `luml_sdk` (per sdk/python/sdk/pyproject.toml), requires Python
>=3.12, and is not published to PyPI (`pip index versions luml_sdk` /
`pip download` both report no matching distribution as of 2026-09) -- it
cannot be installed as a dependency of this adapter or its test suite. This
adapter instead works against the public dataclass *shape* (attribute
access matching `EvalItem`/`EvalResult`/`EvalResults`' documented fields)
via the `_get()` helper below, the same duck-typing approach every other
standalone adapter in this repo uses for a target library that isn't a hard
dependency (see e.g. ../mlflow-openeval-adapter). If `luml_sdk` is ever
published and importable in your environment, pass real `EvalItem`/
`EvalResult`/`EvalResults` instances straight in -- `_get()` reads real
dataclass attributes exactly the same way it reads a dict.

Two independent directions are provided, matching the two distinct things
luml's types represent:

* `EvalItem` (an unscored test definition: inputs + optional expected
  output) <-> EvalPort `TestCase`/`EvalSuite` -- see `eval_item_to_test_case`/
  `eval_item_from_test_case` (single item) and `to_openeval_suite`/
  `from_openeval_suite` (a whole dataset).
* `EvalResult`/`EvalResults` (already-scored evidence from a completed
  `evaluate()` run) <-> EvalPort `Result`/`GraderResult`/`ResultSet` -- see
  `to_openeval`/`from_openeval`, the primary pair this adapter exists for.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "eval_item_to_test_case",
    "eval_item_from_test_case",
    "to_openeval_suite",
    "from_openeval_suite",
    "to_openeval",
    "from_openeval",
    "__version__",
]
__version__ = "0.1.0"

# Mirrors luml.experiments.evaluation.types.REASONING_SUFFIX exactly (verified
# against the real source, see module docstring). Hardcoded rather than
# imported since luml_sdk is not installable here (see module docstring).
REASONING_SUFFIX = "_reasoning"

# Mirrors the "__error__<scorer_name>" key luml's own evaluate.py
# (_evaluate_single_item) writes into EvalResult.scores when one scorer in a
# multi-scorer run raises but the item overall still completed -- other
# scorers' real results are present alongside this marker.
_SCORER_ERROR_PREFIX = "__error__"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    Lets every accessor in this module take either a real luml dataclass
    instance (EvalItem/EvalResult/EvalResults, attribute access) or a plain
    dict/duck-typed stand-in with the same field names (key access) --
    exactly the same convention this repo's other standalone adapters use
    for a target library that isn't a hard dependency.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _json_safe(value: Any) -> Any:
    """Best-effort recursive conversion of an arbitrary Python value into
    something json.dumps() can serialize, falling back to str() for anything
    that isn't natively JSON-safe.

    luml's own types declare `inputs: dict[str, Any]`, `expected_output: Any`,
    `model_response: Any`, and `scores: dict[str, Any]` -- genuinely
    unconstrained. Every other adapter in this repo promises its output is a
    plain dict safe to `json.dump()` directly; this is what keeps that
    promise true even when a caller's `Any` field holds something like a
    custom class instance or a numpy scalar.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _stringify(value: Any) -> str:
    """Render an arbitrary value as a non-empty-when-possible string, for the
    EvalPort fields (`TestCase.input`, `TestCase.expected_output`,
    `Result.actual_output`) that are string-typed by spec even though luml's
    matching fields are typed `Any`."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return str(value) if value is not None else ""


def _flatten_inputs(inputs: Any) -> str:
    """Render luml's `EvalItem.inputs` (a kwargs dict handed to the evaluated
    task, e.g. `{"question": "..."}`) as the single string EvalPort's
    `TestCase.input` requires.

    A single-key dict renders as just that value, so the common case (one
    "question"/"prompt"/... kwarg) reads as a plain prompt string rather than
    a JSON blob. Anything else (zero keys, multiple keys, or a single key
    whose value stringifies empty) renders as canonical sorted-key JSON so no
    information is silently dropped. Either way, the original structured
    dict is *also* preserved verbatim under `metadata.luml.inputs` by
    `eval_item_to_test_case()`, so a `eval_item_from_test_case()` round trip
    never has to reverse-parse this flattening.
    """
    if isinstance(inputs, dict) and len(inputs) == 1:
        candidate = _stringify(next(iter(inputs.values())))
        if candidate:
            return candidate
    if isinstance(inputs, dict):
        return json.dumps(_json_safe(inputs), sort_keys=True) if inputs else "{}"
    return _stringify(inputs) or "null"


# ---------------------------------------------------------------------------
# EvalItem <-> TestCase / EvalSuite
# ---------------------------------------------------------------------------


def eval_item_to_test_case(
    item: Any,
    grader_type: str = "custom",
    grader_id: str = "gr_luml_scorer",
    grader_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert one luml `EvalItem` (a real dataclass instance, or a dict/
    duck-typed stand-in exposing `id`/`inputs`/`expected_output`/`metadata`)
    into an EvalPort `TestCase` dict.

    `grader_type` defaults to `"custom"` with a placeholder
    `params.handler: "luml:scorer"` rather than guessing a specific grader
    (e.g. `"llm_judge"`) -- an `EvalItem` on its own carries no reference to
    which `Scorer`(s) it will actually be run through (those are supplied
    separately to `luml.experiments.evaluation.evaluate.evaluate()`), and
    EvalPort's `llm_judge` grader type requires real `model`/`prompt`
    params this adapter has no honest value for. All five of luml's builtin
    scorers (completeness, correctness, prompt_alignment, relevancy,
    summarization) are in fact LLM-judge scorers -- pass
    `grader_type="llm_judge"` with your own `grader_params={"model": ...,
    "prompt": ...}` once you know which scorer(s) apply, instead of relying
    on a fabricated default.

    Raises `ValueError` if `item.id` is missing or empty -- EvalPort's
    `TestCase.id` is required (see `openeval.validate.validate_test_case`).
    """
    item_id = _get(item, "id")
    if not item_id:
        raise ValueError("EvalItem.id is required and must be non-empty")

    inputs = _get(item, "inputs", {}) or {}
    expected_output = _get(item, "expected_output", None)
    item_metadata = dict(_get(item, "metadata", {}) or {})

    input_text = _flatten_inputs(inputs)

    grader: Dict[str, Any]
    if grader_type == "custom":
        grader = {
            "id": grader_id,
            "type": "custom",
            "description": (
                "Placeholder for whichever luml Scorer(s) this EvalItem is "
                "actually run through -- EvalItem carries no scorer "
                "reference of its own (scorers are supplied separately to "
                "evaluate()). Replace params.handler, or pass grader_type= "
                "/ grader_params= to eval_item_to_test_case(), once you "
                "know which luml Scorer(s) apply."
            ),
            "params": {"handler": "luml:scorer", **(grader_params or {})},
        }
    else:
        grader = {"id": grader_id, "type": grader_type}
        if grader_params:
            grader["params"] = dict(grader_params)

    test_case: Dict[str, Any] = {
        "id": str(item_id),
        "input": input_text,
        "graders": [grader],
    }
    if expected_output is not None:
        test_case["expected_output"] = _stringify(expected_output)

    luml_metadata: Dict[str, Any] = {"inputs": _json_safe(inputs)}
    if expected_output is not None and not isinstance(expected_output, str):
        luml_metadata["expected_output_type"] = type(expected_output).__name__
        luml_metadata["expected_output_raw"] = _json_safe(expected_output)
    if item_metadata:
        luml_metadata["item_metadata"] = _json_safe(item_metadata)
    test_case["metadata"] = {"luml": luml_metadata}

    return test_case


def eval_item_from_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one EvalPort `TestCase` dict back into the constructor kwargs
    for luml's `EvalItem(id, inputs, expected_output, metadata)`.

    Returns a plain dict, not a real `luml.experiments.evaluation.types.
    EvalItem` instance -- `luml_sdk` is not importable here (see module
    docstring) -- so build the real dataclass yourself:
    `EvalItem(**eval_item_from_test_case(tc))`.

    Prefers the original structured `inputs` dict this adapter itself
    stashed under `metadata.luml.inputs` (round-tripping losslessly through
    `eval_item_to_test_case()`) over re-parsing the flattened `input`
    string, since that flattening is intentionally lossy for anything but a
    single-key inputs dict. Falls back to `{"input": test_case["input"]}`
    for a `TestCase` this adapter didn't produce.
    """
    metadata = dict(test_case.get("metadata") or {})
    luml_metadata = dict(metadata.get("luml") or {})

    if "inputs" in luml_metadata and isinstance(luml_metadata["inputs"], dict):
        inputs = dict(luml_metadata["inputs"])
    else:
        inputs = {"input": test_case.get("input")}

    if "expected_output_raw" in luml_metadata:
        expected_output: Any = luml_metadata["expected_output_raw"]
    else:
        expected_output = test_case.get("expected_output")

    item_metadata = dict(luml_metadata.get("item_metadata") or {})

    return {
        "id": test_case.get("id"),
        "inputs": inputs,
        "expected_output": expected_output,
        "metadata": item_metadata,
    }


def to_openeval_suite(
    items: Any,
    suite_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    grader_type: str = "custom",
    grader_id: str = "gr_luml_scorer",
    grader_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export a luml evaluation dataset -- an iterable of `EvalItem` (real
    dataclass instances or dict/duck-typed stand-ins) -- to an EvalPort-
    shaped suite (dict).

    `suite_id` defaults to `f"luml_{dataset_id}"` when `dataset_id` is given,
    else the generic `"luml_dataset"`. See `eval_item_to_test_case()` for why
    `grader_type` defaults to a `"custom"` placeholder rather than a guessed
    concrete grader.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.

    Raises `ValueError` if `items` is empty -- EvalPort's
    `EvalSuite.test_cases` is required and non-empty (see
    `openeval.validate.validate_suite`), and there is no honest empty suite
    to emit for a dataset with nothing in it.
    """
    items = list(items)
    if not items:
        raise ValueError(
            "items is empty; EvalPort's EvalSuite.test_cases must be "
            "non-empty (see openeval.validate.validate_suite)"
        )

    test_cases = [
        eval_item_to_test_case(
            item, grader_type=grader_type, grader_id=grader_id, grader_params=grader_params
        )
        for item in items
    ]

    graders_by_id: Dict[str, Dict[str, Any]] = {}
    for test_case in test_cases:
        for grader in test_case["graders"]:
            graders_by_id.setdefault(grader["id"], grader)

    resolved_id = suite_id or (f"luml_{dataset_id}" if dataset_id else "luml_dataset")

    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": resolved_id,
        "test_cases": test_cases,
        "graders": list(graders_by_id.values()),
        "metadata": {"openeval": {"source": "luml"}},
    }
    if dataset_id:
        suite["metadata"]["luml"] = {"dataset_id": dataset_id}
    return suite


def from_openeval_suite(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of luml `EvalItem`-constructor
    kwarg dicts (see `eval_item_from_test_case`)."""
    return [eval_item_from_test_case(tc) for tc in suite.get("test_cases", [])]


# ---------------------------------------------------------------------------
# EvalResult / EvalResults <-> Result / GraderResult / ResultSet
# ---------------------------------------------------------------------------


def _scorer_grader_result(
    grader_id: str, raw_value: Any, reasoning: Optional[str], threshold: float
) -> Optional[Dict[str, Any]]:
    """Build one `GraderResult` from one luml scorer's raw output value.

    Returns `None` for a value that isn't numeric/bool -- i.e. isn't
    honestly scoreable -- so the caller can preserve it verbatim in
    `Result.metadata.luml.unrecognized_scores` instead of either crashing or
    silently fabricating a score for it. `EvalItem`/`EvalResult`'s `Any`
    typing means a hand-built `EvalResult` (not produced by luml's own
    `evaluate()`) could legitimately put something else in `scores`.
    """
    if isinstance(raw_value, bool):
        clamped = 1.0 if raw_value else 0.0
        was_clamped = False
        raw_kind = "bool"
    elif isinstance(raw_value, (int, float)):
        clamped = max(0.0, min(1.0, float(raw_value)))
        was_clamped = clamped != float(raw_value)
        raw_kind = None
    else:
        return None

    grader_result: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": "luml_llm_judge" if reasoning else "luml_scorer",
        "score": clamped,
        "passed": clamped >= threshold,
    }
    if reasoning:
        grader_result["reason"] = reasoning

    gr_metadata: Dict[str, Any] = {}
    if was_clamped:
        gr_metadata["luml_raw_score"] = raw_value
    if raw_kind:
        gr_metadata["luml_raw_type"] = raw_kind
    if gr_metadata:
        grader_result["metadata"] = gr_metadata

    return grader_result


def _result_for_eval_result(eval_result: Any, threshold: float) -> Dict[str, Any]:
    """Convert one luml `EvalResult` into one EvalPort `Result` dict."""
    eval_item = _get(eval_result, "eval_item")
    test_case_id = _get(eval_item, "id") if eval_item is not None else None
    if not test_case_id:
        raise ValueError(
            "EvalResult.eval_item.id is required to build a Result.test_case_id"
        )
    test_case_id = str(test_case_id)

    model_response = _get(eval_result, "model_response", None)
    scores = dict(_get(eval_result, "scores", {}) or {})
    trace_id = _get(eval_result, "trace_id", None)

    # Whole-item failure: luml's own evaluate.py (_evaluate_single_item)
    # writes exactly {"error": "<message>"} into EvalResult.scores (with
    # model_response left None) when inference or scoring raised for the
    # item as a whole. Represented as Result.error with grader_results: []
    # rather than a graded 0.0 -- "never ran" and "ran and scored 0" are
    # different facts, and EvalPort's own `error` object exists precisely
    # to keep that distinction instead of erasing it.
    if set(scores.keys()) == {"error"} and isinstance(scores.get("error"), str):
        result: Dict[str, Any] = {
            "test_case_id": test_case_id,
            "passed": False,
            "grader_results": [],
            "error": {"type": "runner_error", "message": scores["error"]},
        }
        if trace_id:
            result["metadata"] = {"luml": {"trace_id": str(trace_id)}}
        return result

    # Fold each "<name>_reasoning" companion (luml's LLMJudgeScorer /
    # SupervisedLLMJudgeScorer.parse_judgment() shape, see module docstring)
    # into its base scorer's GraderResult.reason instead of treating it as
    # its own separate, unscoreable grader.
    reasoning_by_base: Dict[str, str] = {
        key[: -len(REASONING_SUFFIX)]: value
        for key, value in scores.items()
        if key.endswith(REASONING_SUFFIX) and isinstance(value, str)
    }
    reasoning_keys = {f"{base}{REASONING_SUFFIX}" for base in reasoning_by_base}

    grader_results: List[Dict[str, Any]] = []
    unrecognized: Dict[str, Any] = {}

    for key, value in scores.items():
        if key in reasoning_keys:
            continue
        if key.startswith(_SCORER_ERROR_PREFIX):
            scorer_name = key[len(_SCORER_ERROR_PREFIX):] or key
            grader_results.append(
                {
                    "grader_id": scorer_name,
                    "type": "luml_scorer_error",
                    "score": None,
                    "passed": False,
                    "reason": str(value),
                }
            )
            continue
        grader_result = _scorer_grader_result(key, value, reasoning_by_base.get(key), threshold)
        if grader_result is not None:
            grader_results.append(grader_result)
        else:
            unrecognized[key] = _json_safe(value)

    passed = bool(grader_results) and all(gr["passed"] for gr in grader_results)
    actual_output = model_response if isinstance(model_response, str) else _stringify(model_response)

    result = {
        "test_case_id": test_case_id,
        "passed": passed,
        "actual_output": actual_output,
        "grader_results": grader_results,
    }

    luml_metadata: Dict[str, Any] = {}
    if trace_id:
        luml_metadata["trace_id"] = str(trace_id)
    if model_response is not None and not isinstance(model_response, str):
        luml_metadata["model_response_type"] = type(model_response).__name__
        luml_metadata["model_response_raw"] = _json_safe(model_response)
    if unrecognized:
        luml_metadata["unrecognized_scores"] = unrecognized
    if not grader_results and not unrecognized:
        luml_metadata["no_scores"] = True
    if luml_metadata:
        result["metadata"] = {"luml": luml_metadata}

    return result


def to_openeval(
    eval_results: Any,
    started_at: str,
    run_id: Optional[str] = None,
    suite_id: Optional[str] = None,
    completed_at: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Export a luml `EvalResults` -- the return value of
    `luml.experiments.evaluation.evaluate.evaluate()`, or a dict/duck-typed
    stand-in exposing `.results` / `.aggregated_scores` / `.dataset_id` --
    to an EvalPort `ResultSet` dict.

    `started_at` is a **required**, caller-supplied ISO 8601 timestamp (e.g.
    `datetime.now(timezone.utc).isoformat()` captured just before calling
    `evaluate()`). Every other field this adapter needs has a real source in
    luml's own dataclasses; a run start time does not -- `EvalResults`,
    `EvalResult`, and `EvalItem` carry no wall-clock timestamp anywhere
    (`EvalResult.trace_id` is an OpenTelemetry trace id, not a time), and
    EvalPort's `ResultSet.started_at` is a required field
    (`openeval.validate.validate_result_set`). Fabricating one would
    misrepresent when the run actually happened, so this adapter refuses to
    guess it for you rather than silently making one up.

    `run_id` defaults to `f"luml_{dataset_id}"` when not given -- a
    deterministic default built from data luml itself already tracked
    (`EvalResults.dataset_id`), not a fabricated value. Pass your own if you
    evaluate the same `dataset_id` more than once and need distinct
    `ResultSet`s (e.g. one per CI run).

    Every scorer name found across all results' `scores` dicts becomes a
    `GraderResult`: `type: "luml_llm_judge"` when a matching
    `"<name>_reasoning"` companion key is present (the shape every one of
    luml's five builtin LLM-judge scorers actually produces -- see module
    docstring), else `type: "luml_scorer"`. `passed` is computed as
    `score >= threshold` (default 0.5) since luml itself has no pass/fail
    concept, only numeric scores -- pass your own threshold to match your
    own bar. A whole-item failure becomes a `Result.error` object with
    `grader_results: []`, not a graded 0.0 (see `_result_for_eval_result`).
    A single scorer failing inside an otherwise-successful item (luml's own
    `"__error__<scorer>"` convention) becomes its own `GraderResult` with
    `score: null` and the failure message in `reason`, alongside the other
    scorers' real results for that item -- nothing about the rest of the
    item is discarded just because one scorer errored.

    `aggregated_scores` (luml's own per-scorer mean/min/max/count plus
    `total_items`/`successful_items`, from `evaluate.py`'s
    `_aggregate_scores()`) is preserved verbatim under
    `metadata.luml.aggregated_scores` -- an evaluation run is already-scored
    data, not just a task definition, so nothing luml computed is thrown
    away. This adapter's own `summary` (`total`/`passed`/`failed`/
    `pass_rate`/`avg_score`) is computed independently from the `Result`s it
    just built, since `aggregated_scores` has no overall pass/fail rollup of
    its own, only per-scorer statistics.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass
    it to `openeval.validate.validate_result_set()` to confirm compliance,
    or `json.dump()` it directly to share as a `.json` results file.

    Raises `ValueError` if `eval_results.results` is empty -- EvalPort's
    `ResultSet.results` is required and non-empty
    (`openeval.validate.validate_result_set`), and there is no honest empty
    `ResultSet` to emit for a run that scored nothing.
    """
    results_in = list(_get(eval_results, "results", []) or [])
    if not results_in:
        raise ValueError(
            "eval_results.results is empty; EvalPort's ResultSet.results "
            "must be non-empty (see openeval.validate.validate_result_set)"
        )

    dataset_id = _get(eval_results, "dataset_id", None)
    aggregated_scores = dict(_get(eval_results, "aggregated_scores", {}) or {})

    resolved_suite_id = suite_id or (str(dataset_id) if dataset_id else "luml_dataset")
    resolved_run_id = run_id or (f"luml_{dataset_id}" if dataset_id else "luml_run")

    results = [_result_for_eval_result(r, threshold) for r in results_in]

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    all_scores = [
        gr["score"]
        for r in results
        for gr in r["grader_results"]
        if gr.get("score") is not None
    ]

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": resolved_suite_id,
        "run_id": resolved_run_id,
        "started_at": started_at,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else 0.0,
            "avg_score": (sum(all_scores) / len(all_scores)) if all_scores else 0.0,
        },
        "metadata": {"openeval": {"source": "luml"}},
    }
    if completed_at:
        result_set["completed_at"] = completed_at

    luml_metadata: Dict[str, Any] = {}
    if dataset_id:
        luml_metadata["dataset_id"] = dataset_id
    if aggregated_scores:
        luml_metadata["aggregated_scores"] = _json_safe(aggregated_scores)
    if luml_metadata:
        result_set["metadata"]["luml"] = luml_metadata

    return result_set


def from_openeval(result_set: Dict[str, Any]) -> Dict[str, Any]:
    """Import an EvalPort `ResultSet` dict into a luml `EvalResults`-
    constructor kwargs dict (`results`, `aggregated_scores`, `dataset_id`).

    This is a **partial** reconstruction, not a lossless round trip of
    `to_openeval()`: a `Result.test_case_id` is the only link back to the
    `EvalItem` it graded, and a bare `ResultSet` does not reference the
    original `Suite`/dataset those items came from. Each returned
    `EvalResult`-shaped dict's `eval_item` therefore carries only `id`
    (`inputs={}`, `expected_output=None`, `metadata={}`) -- if you also have
    the original items (e.g. from `from_openeval_suite()` on the matching
    Suite), look each one up by `test_case_id` and merge it in yourself;
    this function alone cannot recover it.

    `model_response` is recovered from `Result.actual_output` as a string --
    if the original `model_response` wasn't a string, its real type is only
    recoverable from `Result.metadata.luml.model_response_type` /
    `model_response_raw` (present when this adapter itself produced the
    `ResultSet` via `to_openeval()`), not reconstructed automatically here.

    `scores` is rebuilt from `grader_results`: `grader_id -> score` (the
    EvalPort-clamped [0, 1] value -- not the unclamped
    `grader_result.metadata.luml_raw_score` original, if one was recorded;
    read that yourself if you need the exact pre-clamp number), plus each
    grader's `reason` folded back into a `"<grader_id>_reasoning"` key when
    present, mirroring `REASONING_SUFFIX`. A `luml_scorer_error`-typed
    `GraderResult` becomes a `"__error__<grader_id>"` key holding its
    `reason`, matching luml's own per-scorer-failure convention. A
    `Result.error` (whole-item failure) becomes `{"error": message}`,
    matching what luml's own `evaluate.py` writes for that case.

    `aggregated_scores` is taken from `metadata.luml.aggregated_scores` when
    present (i.e. this `ResultSet` round-trips one `to_openeval()` itself
    produced) -- otherwise an empty dict, since recomputing luml's exact
    `_aggregate_scores()` statistics from clamped/reconstructed scores would
    not honestly match what the original run measured.
    """
    metadata = dict(result_set.get("metadata") or {})
    luml_metadata = dict(metadata.get("luml") or {})

    results: List[Dict[str, Any]] = []
    for result in result_set.get("results", []):
        eval_item = {
            "id": result.get("test_case_id"),
            "inputs": {},
            "expected_output": None,
            "metadata": {},
        }

        error = result.get("error")
        if error is not None:
            scores: Dict[str, Any] = {"error": error.get("message", "")}
        else:
            scores = {}
            for grader_result in result.get("grader_results", []):
                grader_id = grader_result.get("grader_id")
                if not grader_id:
                    continue
                if grader_result.get("type") == "luml_scorer_error":
                    scores[f"{_SCORER_ERROR_PREFIX}{grader_id}"] = grader_result.get("reason", "")
                    continue
                scores[grader_id] = grader_result.get("score")
                reason = grader_result.get("reason")
                if reason:
                    scores[f"{grader_id}{REASONING_SUFFIX}"] = reason

        result_metadata = dict(result.get("metadata") or {})
        trace_id = dict(result_metadata.get("luml") or {}).get("trace_id", "")

        results.append(
            {
                "eval_item": eval_item,
                "model_response": result.get("actual_output"),
                "scores": scores,
                "trace_id": trace_id,
            }
        )

    return {
        "results": results,
        "aggregated_scores": dict(luml_metadata.get("aggregated_scores") or {}),
        "dataset_id": luml_metadata.get("dataset_id") or result_set.get("suite_id", ""),
    }
