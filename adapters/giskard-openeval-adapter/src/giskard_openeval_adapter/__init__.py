"""Convert Giskard ``giskard-checks`` Suites, Scenarios, and SuiteResults to and
from EvalPort, the open interchange format for portable LLM evaluation datasets.

Three entry points, matching the two directions of the EvalPort exchange plus
results export:

- :func:`to_openeval` -- a (not-yet-run) ``giskard.checks.Suite`` definition
  (scenarios + their static interacts + their checks) -> an EvalPort
  ``EvalSuite`` document.
- :func:`from_openeval` -- an EvalPort ``EvalSuite`` document -> a list of
  ``giskard.checks.Scenario`` objects, ready to have a target SUT bound
  (``scenario.with_target(...)``) and run.
- :func:`suite_result_to_openeval` -- an already-executed
  ``giskard.checks.SuiteResult`` -> an EvalPort ``ResultSet`` document.

See the package README for the full grader/check mapping table and an honest
accounting of what round-trips losslessly and what doesn't.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

__all__ = ["to_openeval", "from_openeval", "suite_result_to_openeval"]

# ---------------------------------------------------------------------------
# Grader/check kind mapping tables
# ---------------------------------------------------------------------------

# Giskard Check.kind -> EvalPort grader "type", for checks with a direct,
# unambiguous EvalPort equivalent (used by to_openeval).
_CLEAN_CHECK_KIND_TO_GRADER_TYPE: Dict[str, str] = {
    "equals": "exact_match",
    "string_matching": "contains",
    "regex_matching": "regex",
    "semantic_similarity": "semantic_similarity",
    "llm_judge": "llm_judge",
}

# Giskard comparison Check.kind -> EvalPort json_path grader "operator" param.
_COMPARISON_KIND_TO_OPERATOR: Dict[str, str] = {
    "equals": "eq",
    "not_equals": "ne",
    "greater_than": "gt",
    "less_than": "lt",
    "greater_than_equals": "gte",
    "less_than_equals": "lte",
}

# EvalPort json_path "operator" -> giskard comparison Check.kind, for checks
# keyed on a JSONPath field other than the whole "trace.last.outputs" value
# (used by from_openeval).
_OPERATOR_TO_COMPARISON_KIND: Dict[str, str] = {
    "eq": "equals",
    "ne": "not_equals",
    "gt": "greater_than",
    "lt": "less_than",
    "gte": "greater_than_equals",
    "lte": "less_than_equals",
}

# Giskard check kinds with no EvalPort-native equivalent. Exported as EvalPort
# "custom" graders (full definition preserved in metadata); on import, an
# EvalPort grader of any type not explicitly handled below is clean-skipped
# per the spec's "Custom grader handling" convention, the same way
# `openeval run` skips grader types it doesn't know how to execute.
_OPAQUE_CHECK_KINDS = {"all_of", "any_of", "not", "fn", "readability", "rego_policy"}

# EvalPort grader types this adapter can build a real giskard Check for.
_SUPPORTED_IMPORT_GRADER_TYPES = {
    "exact_match",
    "contains",
    "regex",
    "semantic_similarity",
    "llm_judge",
    "json_schema",
    "json_path",
}

_JSON_PATH_ROOT = "trace.last.outputs"


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of a value into something ``json.dumps`` accepts.

    Giskard's ``model_dump()`` (no ``mode="json"``) already returns plain
    Python containers for the pydantic models this adapter touches, but
    values threaded through ``details``/``metadata`` dicts can still be
    arbitrary Python objects (e.g. a stray dataclass or numpy scalar
    surfaced by a custom check). This mirrors the ``_json_safe`` helper used
    by the other adapters in this repository (see ``langfuse-openeval-adapter``).
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        # numpy scalar (e.g. np.float64 from cosine_similarity)
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_safe(value.model_dump())
    return str(value)


def _strip_jsonpath_root(path: str) -> Optional[str]:
    """Return the suffix of a giskard JSONPathStr past ``trace.last.outputs``.

    Returns ``""`` when ``path`` is exactly the root (the whole output value),
    a leading-dot-stripped suffix like ``"field.nested"`` for a sub-path, and
    ``None`` when ``path`` doesn't live under the last interaction's outputs
    at all (e.g. it reads ``trace.last.inputs`` or ``trace.annotations``) --
    those have no EvalPort ``json_path`` equivalent since EvalPort graders
    only ever look at ``actual_output``.
    """
    if path == _JSON_PATH_ROOT:
        return ""
    prefix = _JSON_PATH_ROOT + "."
    if path.startswith(prefix):
        return path[len(prefix):]
    return None


def _evalpath_to_giskard_key(evalpath: str) -> str:
    """Translate a simple EvalPort JSONPath (``"$.field.nested"``) into a
    giskard JSONPathStr rooted at the last interaction's output
    (``"trace.last.outputs.field.nested"``).

    This only handles plain dotted field access, the common case for
    grading a structured JSON output -- the same subset EvalPort's own
    hand-written ``json_path`` grader documents as its primary use case.
    Wildcards, filters, and slice expressions are passed through verbatim
    after the root swap; giskard's ``jsonpath_ng``-based resolver will
    raise a clear validation error at Check-construction time if the
    result isn't valid JSONPath syntax, rather than this adapter silently
    mistranslating it.
    """
    suffix = evalpath[1:] if evalpath.startswith("$") else evalpath
    suffix = suffix.lstrip(".")
    return f"{_JSON_PATH_ROOT}.{suffix}" if suffix else _JSON_PATH_ROOT


# ---------------------------------------------------------------------------
# to_openeval: Suite (definition) -> EvalPort EvalSuite
# ---------------------------------------------------------------------------


def _first_static_text(values: Sequence[Any]) -> Optional[List[str]]:
    texts: List[str] = []
    for value in values:
        if isinstance(value, str) and value:
            texts.append(value)
        else:
            return None
    return texts or None


def _check_to_grader(check: Any, index: int) -> Dict[str, Any]:
    """Convert one giskard Check into an EvalPort grader definition dict."""
    dump = check.model_dump()
    kind = dump.get("kind", type(check).__name__.lower())
    name = dump.get("name")
    description = dump.get("description")
    grader_id = name or f"{kind}_{index}"

    grader: Dict[str, Any] = {"id": grader_id}
    if description:
        grader["description"] = description

    if kind == "equals" and dump.get("key") == _JSON_PATH_ROOT and "expected_value" in dump:
        grader["type"] = "exact_match"
        grader["params"] = {}
    elif kind == "string_matching" and "keyword" in dump:
        grader["type"] = "contains"
        grader["params"] = {
            "substring": dump["keyword"],
            "ignore_case": not dump.get("case_sensitive", True),
        }
    elif kind == "regex_matching" and "pattern" in dump:
        grader["type"] = "regex"
        grader["params"] = {"pattern": dump["pattern"]}
    elif kind == "semantic_similarity":
        grader["type"] = "semantic_similarity"
        grader["params"] = {"threshold": dump.get("threshold", 0.95)}
    elif kind == "llm_judge":
        grader["type"] = "llm_judge"
        prompt = dump.get("prompt") or f"(giskard prompt template: {dump.get('prompt_path')})"
        if not any(token in prompt for token in ("{output}", "{input}", "{expected}")):
            # Giskard's LLMJudge prompts use Jinja2 templating
            # (`{{ trace.last.outputs }}`) rather than EvalPort's plain
            # `{output}`/`{input}`/`{expected}` substitution tokens, which
            # `validate_suite()` requires at least one of. Append a token
            # rather than rewrite the prompt in place, so the original
            # giskard prompt text survives untouched (also preserved
            # verbatim under `metadata.giskard.check.prompt` either way).
            prompt = f"{prompt}\n\nResponse to evaluate: {{output}}"
        grader["params"] = {
            # Giskard's LLMJudge has no per-check model field -- the model is
            # configured on the (global, or per-run) generator instead. This
            # placeholder satisfies EvalPort's required `params.model` while
            # documenting that it isn't a real per-grader model selection;
            # see the README's "what round-trips losslessly" section.
            "model": "giskard-default",
            "prompt": prompt,
        }
    elif kind in _COMPARISON_KIND_TO_OPERATOR and "key" in dump and "expected_value" in dump:
        suffix = _strip_jsonpath_root(dump["key"])
        if suffix is None:
            grader.update(_opaque_grader(check, dump, kind))
        else:
            path = "$" if suffix == "" else f"$.{suffix}"
            grader["type"] = "json_path"
            grader["params"] = {
                "path": path,
                "expected": str(dump["expected_value"]),
                "operator": _COMPARISON_KIND_TO_OPERATOR[kind],
            }
    elif kind == "json_valid" and (dump.get("expected_schema") or dump.get("schema")):
        # JsonValid.model_dump() serializes its `expected_schema` field under
        # its wire alias "schema" (the check is configured with
        # `serialize_by_alias=True`) rather than the Python field name --
        # check both so this doesn't silently misclassify a schema-bearing
        # check as opaque.
        grader["type"] = "json_schema"
        grader["params"] = {"schema": dump.get("expected_schema") or dump.get("schema")}
    else:
        grader.update(_opaque_grader(check, dump, kind))

    grader.setdefault("metadata", {})
    grader["metadata"]["giskard"] = {"check": _json_safe(dump)}
    return grader


def _opaque_grader(check: Any, dump: Mapping[str, Any], kind: str) -> Dict[str, Any]:
    """Build an EvalPort "custom" grader for a check with no direct mapping.

    The full check definition is preserved under `metadata.giskard.check` by
    the caller, so nothing is silently dropped -- a receiving tool that
    understands giskard-checks (i.e. `from_openeval` in this same adapter)
    can reconstruct the original check exactly; any other tool sees an
    inert "custom" grader it will clean-skip, per the spec's convention for
    grader types it doesn't know how to execute.
    """
    return {
        "type": "custom",
        "params": {"handler": f"giskard.checks.{kind}"},
    }


def _scenario_steps_to_test_cases(scenario: Any, scenario_index: int) -> List[Dict[str, Any]]:
    test_cases: List[Dict[str, Any]] = []
    multi_step = len(scenario.steps) > 1

    for step_index, step in enumerate(scenario.steps):
        texts = _first_static_text([interact.inputs for interact in step.interacts])
        if texts is None:
            # Dynamic (callable/generator) inputs have no static value to
            # export -- this step is skipped rather than exported with a
            # fabricated input. See README "what round-trips losslessly".
            continue

        test_case_id = scenario.name if not multi_step else f"{scenario.name}::step_{step_index}"
        graders: List[Dict[str, Any]] = [
            _check_to_grader(check, idx) for idx, check in enumerate(step.checks)
        ]
        if not graders:
            # A TestCase.graders array is required and must be non-empty. A
            # step with interacts but no checks has nothing to grade against,
            # so it can't become a spec-valid TestCase; skip it rather than
            # inventing a grader.
            continue

        test_case: Dict[str, Any] = {
            "id": test_case_id,
            "input": texts[0] if len(texts) == 1 else texts,
            "graders": graders,
        }

        expected_output = _infer_expected_output(step.checks)
        if expected_output is not None:
            test_case["expected_output"] = expected_output

        if scenario.tags:
            test_case["tags"] = list(scenario.tags)

        metadata: Dict[str, Any] = {"giskard": {"scenario_name": scenario.name}}
        if scenario.annotations:
            metadata["giskard"]["scenario_annotations"] = _json_safe(scenario.annotations)
        if multi_step:
            metadata["giskard"]["step_index"] = step_index
        test_case["metadata"] = metadata

        test_cases.append(test_case)

    return test_cases


def _infer_expected_output(checks: Iterable[Any]) -> Optional[str]:
    """Best-effort literal "golden answer" for a step, read off its checks.

    EvalPort's `TestCase.expected_output` has no first-class equivalent on a
    giskard Scenario -- it lives implicitly inside whichever check compares
    the output to a literal value. The first check with an unambiguous
    literal (an `equals`/`string_matching`/`regex_matching`/
    `semantic_similarity` check with a static, non-JSONPath-sourced value)
    wins. This is purely informational: the grader itself (already exported
    alongside) carries the operative comparison logic either way.
    """
    for check in checks:
        dump = check.model_dump()
        kind = dump.get("kind")
        if kind == "equals" and dump.get("key") == _JSON_PATH_ROOT and "expected_value" in dump:
            value = dump["expected_value"]
            if isinstance(value, str):
                return value
        if kind == "string_matching" and "keyword" in dump:
            return dump["keyword"]
        if kind == "regex_matching" and "pattern" in dump:
            return dump["pattern"]
        if kind == "semantic_similarity" and dump.get("reference_text"):
            return dump["reference_text"]
    return None


def to_openeval(
    suite: Any,
    suite_id: Optional[str] = None,
    version: str = "1.0.0",
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a `giskard.checks.Suite` definition into an EvalPort `EvalSuite`.

    `suite` is a *not-yet-run* `Suite` (built with the fluent
    `Scenario(...).interact(...).check(...)` API, or equivalent) -- this
    reads scenario/step/check *definitions*, not execution results. For
    exporting an already-run suite's outcomes, use
    `suite_result_to_openeval` instead.

    Parameters
    ----------
    suite:
        A `giskard.checks.Suite` instance.
    suite_id:
        EvalPort suite id. Defaults to `suite.name`.
    version:
        EvalPort spec version to stamp on the document. Defaults to the
        current stable spec version, "1.0.0".
    description:
        Optional human-readable suite description.

    Returns
    -------
    dict
        An EvalPort `EvalSuite` document. Validate it with
        `openeval.validate.validate_suite()` before using it.

    Notes
    -----
    Only scenario steps whose interacts all have static (already-resolved)
    string inputs are exported -- steps driven by a callable, generator, or
    `InputGenerator` have no fixed value to serialize and are skipped, since
    fabricating one would silently misrepresent the scenario. Steps with
    interacts but no checks are also skipped, since EvalPort requires every
    `TestCase` to carry at least one grader.
    """
    test_cases: List[Dict[str, Any]] = []
    for scenario in suite.scenarios:
        test_cases.extend(_scenario_steps_to_test_cases(scenario, len(test_cases)))

    result: Dict[str, Any] = {
        "version": version,
        "id": suite_id or suite.name,
        "name": suite.name,
        "test_cases": test_cases,
    }
    if description:
        result["description"] = description
    return result


# ---------------------------------------------------------------------------
# from_openeval: EvalPort EvalSuite -> list[giskard.checks.Scenario]
# ---------------------------------------------------------------------------


def _import_giskard_checks():
    try:
        import giskard.checks as gc
    except ImportError as exc:  # pragma: no cover - exercised via a dedicated test
        raise ImportError(
            "from_openeval() needs the giskard-checks package to construct "
            "real Scenario/Check objects. It isn't published to PyPI yet "
            "(pre-1.0 beta, monorepo-only as of this writing) -- install it "
            "from source: pip install "
            "\"giskard-checks @ git+https://github.com/Giskard-AI/giskard-oss.git"
            "#subdirectory=libs/giskard-checks\" "
            "(requires Python >=3.12). See this package's README for details."
        ) from exc
    return gc


def _grader_to_check(gc: Any, grader: Mapping[str, Any], expected_output: Optional[str]) -> Optional[Any]:
    grader_type = grader.get("type")
    params = grader.get("params") or {}
    name = grader.get("id")
    description = grader.get("description")
    common: Dict[str, Any] = {}
    if name:
        common["name"] = name
    if description:
        common["description"] = description

    if grader_type == "exact_match":
        return gc.Equals(key=_JSON_PATH_ROOT, expected_value=expected_output, **common)
    if grader_type == "contains":
        return gc.StringMatching(
            keyword=params["substring"],
            case_sensitive=not params.get("ignore_case", False),
            **common,
        )
    if grader_type == "regex":
        return gc.RegexMatching(pattern=params["pattern"], **common)
    if grader_type == "semantic_similarity":
        return gc.SemanticSimilarity(
            reference_text=expected_output,
            threshold=params.get("threshold", 0.95),
            **common,
        )
    if grader_type == "llm_judge":
        return gc.LLMJudge(prompt=params["prompt"], **common)
    if grader_type == "json_schema":
        return gc.JsonValid(schema=params["schema"], **common)
    if grader_type == "json_path":
        operator = params.get("operator", "eq")
        key = _evalpath_to_giskard_key(params["path"])
        expected = params.get("expected")
        if operator == "contains":
            return gc.StringMatching(text_key=key, keyword=str(expected), **common)
        comparison_kind = _OPERATOR_TO_COMPARISON_KIND.get(operator, "equals")
        comparison_cls = {
            "equals": gc.Equals,
            "not_equals": gc.NotEquals,
            "greater_than": gc.GreaterThan,
            "less_than": gc.LessThan,
            "greater_than_equals": gc.GreaterThanEquals,
            "less_than_equals": gc.LessThanEquals,
        }[comparison_kind]
        return comparison_cls(key=key, expected_value=expected, **common)

    # Unsupported grader type (code, human, custom, "model graded", or any
    # future type this adapter doesn't yet know how to build a Check for):
    # clean-skip, matching the spec's "Custom grader handling" convention
    # instead of raising and aborting the whole conversion.
    return None


def from_openeval(suite: Mapping[str, Any]) -> List[Any]:
    """Build `giskard.checks.Scenario` objects from an EvalPort `EvalSuite`.

    Each EvalPort `TestCase` becomes one `Scenario` with one step: an
    `interact()` per input turn (a single `interact()` for a string
    `TestCase.input`, one per element for an array/multi-turn input) followed
    by a `check()` per grader this adapter knows how to build a real giskard
    `Check` for. Every returned `Scenario` leaves `outputs` unbound (`MISSING`)
    on its interacts -- call `scenario.with_target(your_sut)` (or pass
    `target=` to `Suite.run()`/`Scenario.run()`) before running it, since this
    adapter has no live system under test to call.

    Grader types with no giskard-checks equivalent (`code`, `human`, `custom`,
    `"model graded"`, and any inline grader object of an unrecognized type)
    are clean-skipped rather than raising -- if a `TestCase` ends up with zero
    checks as a result, its `Scenario` is still returned (with just the
    interacts), so callers can inspect and handle that case rather than
    having the whole suite import silently fail.

    Requires the `giskard-checks` package (not on PyPI yet -- see this
    package's README for the install command).
    """
    gc = _import_giskard_checks()
    scenarios: List[Any] = []

    for test_case in suite.get("test_cases", []):
        scenario = gc.Scenario(test_case["id"])

        raw_input = test_case["input"]
        turns = raw_input if isinstance(raw_input, list) else [raw_input]
        for turn in turns:
            scenario.interact(turn)

        expected_output = test_case.get("expected_output")
        for grader in test_case.get("graders", []):
            if isinstance(grader, str):
                # A bare grader-id string references a suite-level grader
                # definition this adapter doesn't have visibility into here
                # (from_openeval operates on one suite document at a time,
                # same as the other adapters in this repo) -- skip it rather
                # than guessing at a shared definition.
                continue
            check = _grader_to_check(gc, grader, expected_output)
            if check is not None:
                scenario.check(check)

        tags = test_case.get("tags")
        if tags:
            scenario.with_tags(list(tags))

        scenarios.append(scenario)

    return scenarios


# ---------------------------------------------------------------------------
# suite_result_to_openeval: SuiteResult (executed) -> EvalPort ResultSet
# ---------------------------------------------------------------------------


def _check_result_to_grader_result(check_result: Any, index: int) -> Dict[str, Any]:
    dump = check_result.model_dump()
    status = dump.get("status")
    details = dump.get("details") or {}
    grader_id = details.get("check_name") or details.get("check_kind") or f"check_{index}"
    grader_type = _CLEAN_CHECK_KIND_TO_GRADER_TYPE.get(
        details.get("check_kind"), details.get("check_kind") or "custom"
    )

    if status == "pass":
        score, passed = 1.0, True
    elif status == "fail":
        score, passed = 0.0, False
    else:  # "error" or "skip" -- no verdict was reached, per giskard's own
        # ERROR/SKIP-are-distinct-from-FAIL convention (see result.py's
        # module docstring). EvalPort scores null rather than 0.0 so an
        # errored/skipped check isn't misread as a failed one downstream.
        score, passed = None, False

    grader_result: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": grader_type,
        "score": score,
        "passed": passed,
    }
    message = dump.get("message")
    if message:
        grader_result["reason"] = message

    metadata: Dict[str, Any] = {"giskard": {"status": status}}
    if dump.get("metrics"):
        metadata["giskard"]["metrics"] = _json_safe(dump["metrics"])
    if details:
        metadata["giskard"]["details"] = _json_safe(
            {k: v for k, v in details.items() if k not in ("check_kind", "check_name", "check_description")}
        )
    grader_result["metadata"] = metadata

    return grader_result


def _scenario_result_to_results(scenario_result: Any) -> List[Dict[str, Any]]:
    dump_steps = scenario_result.steps
    multi_step = len(dump_steps) > 1
    results: List[Dict[str, Any]] = []

    for step_index, test_case_result in enumerate(dump_steps):
        test_case_id = (
            scenario_result.scenario_name
            if not multi_step
            else f"{scenario_result.scenario_name}::step_{step_index}"
        )
        grader_results = [
            _check_result_to_grader_result(check_result, idx)
            for idx, check_result in enumerate(test_case_result.results)
        ]

        result: Dict[str, Any] = {
            "test_case_id": test_case_id,
            "grader_results": grader_results,
            "passed": bool(test_case_result.passed),
            "duration_ms": int(test_case_result.duration_ms),
        }

        if test_case_result.error is not None:
            result["error"] = {
                "type": "runner_error",
                "message": test_case_result.error.summary(),
            }

        metadata: Dict[str, Any] = {"giskard": {"status": test_case_result.status.value}}
        if step_index == len(dump_steps) - 1:
            # Only the final step of a scenario carries the accumulated
            # trace's last interaction -- attach the actual model output
            # there, where it's unambiguous which turn produced it.
            last = scenario_result.final_trace.last
            if last is not None and isinstance(last.outputs, str):
                result["actual_output"] = last.outputs
        result["metadata"] = metadata

        results.append(result)

    return results


def suite_result_to_openeval(
    suite_result: Any,
    suite_id: str,
    run_id: str,
    started_at: str,
    completed_at: Optional[str] = None,
    version: str = "1.0.0",
) -> Dict[str, Any]:
    """Convert an executed `giskard.checks.SuiteResult` into an EvalPort `ResultSet`.

    Each `ScenarioResult` becomes one or more EvalPort results -- one per
    step (matching `to_openeval`'s scenario-with-multiple-steps ->
    `"{scenario}::step_{i}"` test case id convention, so a `SuiteResult` from
    running a `Suite` built by `from_openeval` lines back up with the
    original `TestCase.id`s). Each `CheckResult` within a step becomes one
    `GraderResult`; giskard's four-state `CheckStatus` (`PASS`/`FAIL`/
    `ERROR`/`SKIP`) collapses onto EvalPort's boolean `passed` plus a
    `score` of `1.0`/`0.0`/`null`/`null` respectively -- `null` for both
    `ERROR` and `SKIP` since, per giskard's own result semantics, neither
    reached a pass/fail verdict.

    Parameters
    ----------
    suite_result:
        A `giskard.checks.SuiteResult`, e.g. from `await suite.run()`.
    suite_id:
        EvalPort suite id this result set belongs to.
    run_id:
        Unique identifier for this run.
    started_at:
        ISO 8601 run start timestamp.
    completed_at:
        Optional ISO 8601 run completion timestamp.
    version:
        EvalPort spec version to stamp on the document. Defaults to "1.0.0".

    Returns
    -------
    dict
        An EvalPort `ResultSet` document. Validate it with
        `openeval.validate.validate_result_set()` before using it.
    """
    results: List[Dict[str, Any]] = []
    for scenario_result in suite_result.results:
        results.extend(_scenario_result_to_results(scenario_result))

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "summary": {
            "total": len(suite_result.results),
            "passed": suite_result.passed_count,
            "failed": suite_result.failed_count,
            "skipped": suite_result.skipped_count,
            "duration_ms": int(suite_result.duration_ms),
        },
        "metadata": {"giskard": {"errored_count": suite_result.errored_count}},
    }
    if suite_result.pass_rate is not None:
        result_set["summary"]["pass_rate"] = suite_result.pass_rate
    if completed_at:
        result_set["completed_at"] = completed_at

    return result_set
