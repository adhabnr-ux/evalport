"""Literal AI <-> EvalPort adapter.

Standalone converter between Literal AI (`pip install literalai`, tested
against 0.1.300) datasets/experiments and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport). Built for issue #23.

Literal AI's Dataset/DatasetItem/DatasetExperiment data model, confirmed by
importing the real package and reading `__dataclass_fields__` directly:

  Dataset               ['api', 'id', 'created_at', 'metadata', 'name',
                          'description', 'items', 'type']
  DatasetItem           ['id', 'created_at', 'dataset_id', 'metadata',
                          'input', 'expected_output', 'intermediary_steps']
  DatasetExperiment     ['api', 'id', 'created_at', 'name', 'dataset_id',
                          'params', 'prompt_variant_id', 'items']
  DatasetExperimentItem ['id', 'dataset_experiment_id', 'dataset_item_id',
                          'scores', 'input', 'output', 'experiment_run_id']
  ScoreDict             {'id', 'name', 'type': Literal['HUMAN','CODE','AI'],
                          'value': float, 'label', 'stepId',
                          'datasetExperimentItemId', 'comment', 'tags'}

Three real wrinkles this adapter solves (see issue #23):

  A. `DatasetItem.input` / `.expected_output` are raw `Dict`, but EvalPort's
     `TestCase.input` / `.expected_output` are `string | string[]`.
     `flatten_dict_field()` extracts a clean string from the dict, honestly
     documented rather than assuming which key is "the" input.

  B. `ScoreDict.value` is an unbounded float. EvalPort's
     `GraderResult.score` MUST be `null` or a number in [0.0, 1.0]
     (spec/SPEC.md, Validation Rule 5). `clamp_score()` clamps it and
     preserves the original under the spec's own reserved metadata key,
     `metadata.openeval.raw_score` (spec/SPEC.md, Appendix B) -- not an
     adapter-invented key.

  C. `ScoreDict.type` is `"HUMAN" | "CODE" | "AI"`, a 3-way categorical tag
     that isn't a 1:1 match for EvalPort's grader type vocabulary.
     `map_grader_type()` applies the documented mapping
     (AI -> llm_judge, CODE -> code, HUMAN -> human).

This is a plain-dataclass SDK (not pydantic, not attrs), so every accessor
below reads Literal AI objects via plain `getattr`/`dict.get` rather than
`.model_fields` or `attrs.fields()`.
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
    "flatten_dict_field",
    "clamp_score",
    "map_grader_type",
    "map_grader_type_reverse",
    "to_openeval",
    "from_openeval",
    "results_to_openeval",
    "__version__",
]
__version__ = "0.1.0"

# --- Challenge C: grader-type vocabulary map --------------------------------

_GRADER_TYPE_MAP = {"HUMAN": "human", "CODE": "code", "AI": "llm_judge"}
_GRADER_TYPE_MAP_REVERSE = {v: k for k, v in _GRADER_TYPE_MAP.items()}


def map_grader_type(literalai_type: str) -> str:
    """Literal AI `ScoreDict.type` ('HUMAN'/'CODE'/'AI') -> EvalPort grader type."""
    if literalai_type is None:
        raise ValueError("Literal AI score type is missing (None)")
    if not isinstance(literalai_type, str):
        raise TypeError(f"grader type must be a string, got {type(literalai_type)!r}")
    key = literalai_type.strip().upper()
    if key not in _GRADER_TYPE_MAP:
        raise ValueError(
            f"Unknown Literal AI score type {literalai_type!r}; expected one of "
            f"{sorted(_GRADER_TYPE_MAP)}"
        )
    return _GRADER_TYPE_MAP[key]


def map_grader_type_reverse(openeval_type: str) -> str:
    """EvalPort grader type -> Literal AI `ScoreDict.type`."""
    if openeval_type is None:
        raise ValueError("EvalPort grader type is missing (None)")
    if not isinstance(openeval_type, str):
        raise TypeError(f"grader type must be a string, got {type(openeval_type)!r}")
    key = openeval_type.strip().lower()
    if key not in _GRADER_TYPE_MAP_REVERSE:
        raise ValueError(
            f"Unknown EvalPort grader type {openeval_type!r}; expected one of "
            f"{sorted(_GRADER_TYPE_MAP_REVERSE)}"
        )
    return _GRADER_TYPE_MAP_REVERSE[key]


# --- Challenge A: dict -> string flattening ---------------------------------

_PREFERRED_KEYS = ("question", "input", "prompt", "text", "query", "output", "answer")


def flatten_dict_field(value: Any) -> str:
    """Flatten a Literal AI `input`/`expected_output` dict into a plain string.

    EvalPort's `TestCase.input`/`expected_output` require `string | string[]`
    (spec/SPEC.md ss1), but Literal AI's datasets are schema-free key-value
    rows -- there is no single "the input key" contract. This picks the
    first key from a small set of common names; if none match, it falls
    back to the first string-valued entry (dict insertion order is stable
    in Python, so this is deterministic); if there's still nothing usable,
    it serializes the whole dict to JSON rather than raising, since an
    arbitrary Literal AI row is still valid data that deserves a
    (documented) string representation, not a hard failure.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if not value:
            raise ValueError("cannot flatten an empty dict")
        for key in _PREFERRED_KEYS:
            v = value.get(key)
            if isinstance(v, str):
                return v
        for v in value.values():
            if isinstance(v, str):
                return v
        return json.dumps(value, default=str, sort_keys=True)
    raise TypeError(f"unsupported field type: {type(value)!r}")


# --- Challenge B: score clamping --------------------------------------------

def clamp_score(raw_value: float) -> Dict[str, Any]:
    """Literal AI `ScoreDict.value` (any scale) -> EvalPort score in [0.0, 1.0].

    Returns {"score": <clamped float>, "raw_score": <original>,
    "was_clamped": <bool>}. The raw value is always returned so callers can
    decide where to stash it -- `results_to_openeval()` below puts it under
    the spec's reserved `metadata.openeval.raw_score` key.
    """
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise TypeError(f"score must be numeric, got {type(raw_value)!r}")
    import math
    if math.isnan(raw_value):
        raise ValueError("score cannot be NaN")

    clamped = float(raw_value)
    was_clamped = False
    if clamped > 1.0:
        clamped = 1.0
        was_clamped = True
    elif clamped < 0.0:
        clamped = 0.0
        was_clamped = True

    return {"score": clamped, "raw_score": float(raw_value), "was_clamped": was_clamped}


# --- Accessor helper (dataclass-or-dict, like every other adapter here) ----

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --- Public API: to_openeval / from_openeval / results_to_openeval ---------

_DEFAULT_GRADER_ID = "gr_literalai_default"


def to_openeval(
    dataset: Any,
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """Convert a Literal AI `Dataset` (or a plain dict with the same shape)
    into an EvalPort suite (dict). Conforms to `schema/suite.json` --
    pass the result to `openeval.validate.validate_suite()` to confirm.

    `grader_type` selects the generated default grader: "llm_judge"
    (default; Literal AI datasets rarely ship a ground-truth grader
    definition of their own) or "exact_match" for cases with a clean
    string `expected_output`.
    """
    items = _get(dataset, "items", []) or []
    dataset_id = _get(dataset, "id") or "literalai_dataset"
    dataset_name = _get(dataset, "name") or dataset_id

    if grader_type == "exact_match":
        grader: Dict[str, Any] = {
            "id": _DEFAULT_GRADER_ID,
            "type": "exact_match",
            "params": {"ignore_case": True},
        }
    else:
        grader = {
            "id": _DEFAULT_GRADER_ID,
            "type": "llm_judge",
            "params": {
                "model": "gpt-4o",
                "prompt": (
                    "Does {output} correctly answer {input}? "
                    'Expected: {expected}. Return JSON: {"score": 0.0-1.0}.'
                ),
            },
        }

    test_cases: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        item_id = _get(item, "id") or f"tc_{idx}"
        raw_input = _get(item, "input")
        raw_expected = _get(item, "expected_output")
        raw_metadata = dict(_get(item, "metadata") or {})

        metadata: Dict[str, Any] = dict(raw_metadata)
        metadata["literalai"] = {
            "original_input": raw_input,
            "original_expected_output": raw_expected,
        }

        tc: Dict[str, Any] = {
            "id": str(item_id),
            "input": flatten_dict_field(raw_input) if raw_input is not None else "",
            "graders": [_DEFAULT_GRADER_ID],
            "metadata": metadata,
        }
        if raw_expected is not None:
            tc["expected_output"] = flatten_dict_field(raw_expected)

        test_cases.append(tc)

    return {
        "version": OPENEVAL_VERSION,
        "id": f"literalai_{dataset_id}",
        "name": dataset_name,
        "graders": [grader],
        "test_cases": test_cases,
        "metadata": {"openeval": {"source": "literalai"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite's test cases into `Dataset.create_item()`-ready
    dicts: `{"input": Dict, "expected_output": Optional[Dict], "metadata": Dict}`.

    If a test case's `metadata["literalai"]["original_input"]` is present
    (i.e. it round-trips a suite produced by `to_openeval()` above), the
    original dict shape is restored verbatim rather than re-wrapping the
    flattened string -- this is what makes the round trip lossless for
    suites that came from this adapter in the first place.
    """
    items: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        metadata = dict(tc.get("metadata") or {})
        literalai_meta = metadata.pop("literalai", None)

        if isinstance(literalai_meta, dict) and "original_input" in literalai_meta:
            input_value = literalai_meta["original_input"]
            expected_value = literalai_meta.get("original_expected_output")
        else:
            input_value = {"question": tc.get("input")}
            expected_value = (
                {"answer": tc["expected_output"]} if "expected_output" in tc else None
            )

        items.append(
            {
                "input": input_value,
                "expected_output": expected_value,
                "metadata": metadata,
            }
        )
    return items


def results_to_openeval(
    experiment_items: Any,
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a Literal AI `DatasetExperiment`'s `DatasetExperimentItem` list
    (each carrying real `scores: List[ScoreDict]`) into an EvalPort ResultSet.
    Conforms to `schema/resultset.json` -- pass the result to
    `openeval.validate.validate_result_set()` to confirm.

    `experiment_items` accepts either a `DatasetExperiment` object (its
    `.items` attribute is used) or a bare list of `DatasetExperimentItem`s.

    `started_at` defaults to the current UTC time if not supplied (the
    schema requires it); pass it explicitly for reproducible output.
    """
    if started_at is None:
        import datetime

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    items = _get(experiment_items, "items", None)
    if items is None:
        items = list(experiment_items)  # bare iterable of DatasetExperimentItem

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        test_case_id = _get(item, "dataset_item_id") or f"tc_{idx}"
        raw_output = _get(item, "output")
        scores = _get(item, "scores") or []

        grader_results: List[Dict[str, Any]] = []
        for score in scores:
            score_id = _get(score, "id") or _get(score, "name") or "literalai_score"
            score_type = _get(score, "type")
            score_value = _get(score, "value")
            comment = _get(score, "comment")

            clamp_info = clamp_score(score_value)
            gr: Dict[str, Any] = {
                "grader_id": str(score_id),
                "type": map_grader_type(score_type),
                "score": clamp_info["score"],
                "passed": clamp_info["score"] >= 0.5,
                "metadata": {"openeval": {"raw_score": clamp_info["raw_score"]}},
            }
            if comment:
                gr["reason"] = comment
            grader_results.append(gr)

        result: Dict[str, Any] = {
            "test_case_id": str(test_case_id),
            "grader_results": grader_results,
            "passed": all(gr["passed"] for gr in grader_results) if grader_results else False,
        }
        if raw_output is not None:
            result["actual_output"] = (
                flatten_dict_field(raw_output) if isinstance(raw_output, dict) else str(raw_output)
            )

        results.append(result)

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "runner": {"name": "literalai-openeval-adapter"},
    }
