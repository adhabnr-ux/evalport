"""Convert Eval-ai-library (https://github.com/meshkovQA/Eval-ai-library, PyPI
``eval-ai-library``, import path ``eval_lib``) test cases and evaluation
results to/from the EvalPort open evaluation format.

Built at the maintainer's request in meshkovQA/Eval-ai-library#2, where the
owner asked that the adapter "live in the EvalPort repo alongside your other
adapters" and confirmed that ``eval_lib.evaluation_schema`` (``TestCaseResult``,
``MetricResult``) "is public and stable, so you can import it from an external
package without any changes on our side".

The upstream surface this module targets was read directly from
``eval-ai-library==0.7.26`` (not guessed):

- ``eval_lib.testcases_schema.EvalTestCase`` (pydantic): ``input: str``,
  ``actual_output: str``, ``expected_output: Optional[str]``,
  ``retrieval_context: Optional[List[str]]``, ``tools_called`` /
  ``expected_tools: Optional[List[str]]``, ``reasoning``, ``name``, plus the
  reliability fields ``execution_trace``, ``agent_confidence``,
  ``perturbation_group``, ``planning_steps``, ``resource_usage`` and the
  free-form ``extra_fields``. ``to_openeval()`` converts a list of these into
  an EvalPort ``Suite``; ``from_openeval()`` reverses it.

- ``eval_lib.evaluation_schema.MetricResult`` (dataclass): ``name``,
  ``score: float``, ``threshold: float``, ``success: bool``,
  ``evaluation_cost: Optional[float]``, ``reason: Optional[str]``,
  ``evaluation_model: str``, ``evaluation_log: Optional[Any]``.

- ``eval_lib.evaluation_schema.TestCaseResult`` (dataclass): ``input``,
  ``actual_output``, ``expected_output``, ``retrieval_context``, ``success``
  (the AND of every metric's ``success``), ``metrics_data: List[MetricResult]``,
  ``tools_called``, ``expected_tools``. ``eval_lib.evaluate()`` returns
  ``List[Tuple[None, List[TestCaseResult]]]``; ``results_to_openeval()``
  accepts that tuple shape as well as a flat list of ``TestCaseResult``.

- Every metric family in ``eval_lib`` (RAG/LLM-judge, agent, security,
  deterministic, vector, reliability) reports ``score`` already normalized to
  ``0.0..1.0`` and computes ``success = score >= threshold`` (checked in
  ``metric_pattern.py`` and each ``*_metric.py``; even ``CustomEvalMetric``'s
  raw 0-10 judge score is normalized before it is stored). EvalPort's
  ``GraderResult.score`` is also ``[0, 1]`` or ``null``, so the mapping is the
  identity, clamped defensively, with the raw upstream value always kept in
  ``metadata.eval_ai_library.score`` so nothing is lost if a third-party
  ``MetricPattern`` subclass ever reports outside that range.

Grader mapping (see ``README.md`` for the table): metric *names* are what
``MetricResult.name`` carries, e.g. ``"exactMatchMetric"``,
``"answerRelevancyMetric"``, ``"gEval"``. Only mappings whose EvalPort grader
type needs no params this module cannot honestly supply are used --
``exactMatchMetric -> exact_match`` always; ``semanticSimilarityMetric ->
semantic_similarity`` (its required ``threshold`` is exactly the metric's
threshold); ``regexMatchMetric -> regex``, ``jsonSchemaMetric -> json_schema``
and single-keyword ``containsMetric -> contains`` only when the caller passes
the real metric *object* so ``pattern`` / ``schema`` / ``keywords`` are known.
Everything else, including every LLM-as-judge metric, becomes ``custom`` with
``params.handler = "eval_ai_library:<metricName>"``: EvalPort's ``llm_judge``
type requires ``params.model`` and ``params.prompt``, and this module has no
honest prompt value without the caller's live metric configuration, so it
follows the same convention as ``haystack-openeval-adapter`` rather than
fabricating one.

Duck typing: every function accepts the real ``eval_lib`` pydantic/dataclass
objects OR plain dicts with the same field names (e.g. from
``dataclasses.asdict()``/``model_dump()``/a JSON dump of a dashboard cache).
``eval_lib`` is never imported at module import time; ``from_openeval(...,
as_models=True)`` imports it lazily and only on request.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "results_to_openeval",
    "metric_to_grader",
    "test_case_id",
    "__version__",
]
__version__ = "0.1.0"

_NS = "eval_ai_library"
_HANDLER_PREFIX = "eval_ai_library:"
_DEFAULT_GRADER_ID = "gr_eval_ai_library"

# EvalTestCase / TestCaseResult fields that map 1:1 onto EvalPort TestCase
# fields, plus the ones that only fit in metadata.
_TC_LIST_FIELDS = ("retrieval_context", "tools_called", "expected_tools")
_TC_EXTRA_FIELDS = (
    "reasoning",
    "execution_trace",
    "agent_confidence",
    "perturbation_group",
    "planning_steps",
    "resource_usage",
    "extra_fields",
)


# --------------------------------------------------------------------------- #
# Duck-typed field access
# --------------------------------------------------------------------------- #
def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a dict, a dataclass or a pydantic model alike."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _has(obj: Any, name: str) -> bool:
    if isinstance(obj, Mapping):
        return name in obj
    return hasattr(obj, name)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of an arbitrary upstream value (pydantic model,
    dataclass, numpy scalar, ...) into something ``json.dumps`` accepts.
    Falls back to ``repr()`` rather than dropping the value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)  # pydantic v2
    if callable(dump):
        try:
            return _jsonable(dump())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return _jsonable(dataclasses.asdict(value))
    item = getattr(value, "item", None)  # numpy scalars
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:  # pragma: no cover - defensive
            pass
    return repr(value)


def _str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


# --------------------------------------------------------------------------- #
# Metric name -> grader
# --------------------------------------------------------------------------- #
_CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def _slug(name: str) -> str:
    """``"answerRelevancyMetric"`` -> ``"answer_relevancy_metric"``. Any
    character outside ``[a-z0-9_]`` becomes ``_`` so the id is always safe."""
    s = _CAMEL_RE_1.sub(r"\1_\2", str(name))
    s = _CAMEL_RE_2.sub(r"\1_\2", s)
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "metric"


def _grader_id(metric_name: str) -> str:
    return f"gr_{_slug(metric_name)}"


def metric_to_grader(metric: Any) -> Dict[str, Any]:
    """Build one EvalPort ``Grader`` dict from an ``eval_lib`` metric.

    ``metric`` may be:

    - a real ``MetricPattern`` instance (``ExactMatchMetric(...)``,
      ``AnswerRelevancyMetric(model=..., threshold=...)``, ...) -- the most
      informative input, since ``pattern`` / ``schema`` / ``keywords`` on the
      deterministic metrics allow a typed ``regex`` / ``json_schema`` /
      ``contains`` grader instead of ``custom``;
    - a ``MetricResult`` (or dict shaped like one) -- ``name``, ``threshold``,
      ``evaluation_model`` are used;
    - a dict ``{"name": ..., "threshold": ..., "model": ...}``;
    - a bare metric-name string.

    The grader id is ``gr_<snake_case(name)>``, so the same metric always
    yields the same id whichever of these shapes it arrives in.
    """
    if isinstance(metric, str):
        name, threshold, model = metric, None, None
        src: Any = {}
    else:
        src = metric
        name = _get(metric, "name")
        if not name:
            raise ValueError(f"metric has no 'name': {metric!r}")
        threshold = _get(metric, "threshold")
        model = _get(metric, "model", None)
        if model is None:
            model = _get(metric, "evaluation_model", None)
    name = str(name)

    ns: Dict[str, Any] = {"metric": name}
    if threshold is not None:
        ns["threshold"] = _jsonable(threshold)
    if model is not None:
        ns["evaluation_model"] = str(model)

    grader: Dict[str, Any] = {"id": _grader_id(name), "type": "custom"}
    params: Dict[str, Any] = {}

    if name == "exactMatchMetric":
        grader["type"] = "exact_match"
        for opt in ("case_sensitive", "strip_whitespace"):
            if _has(src, opt):
                params[opt] = bool(_get(src, opt))
    elif name == "semanticSimilarityMetric" and threshold is not None:
        grader["type"] = "semantic_similarity"
        params["threshold"] = float(threshold)
        if model:
            params["model"] = str(model)
    elif name == "regexMatchMetric" and _get(src, "pattern"):
        grader["type"] = "regex"
        params["pattern"] = str(_get(src, "pattern"))
        if _get(src, "full_match"):
            ns["full_match"] = True
    elif name == "jsonSchemaMetric" and isinstance(_get(src, "schema"), Mapping) and _get(src, "schema"):
        grader["type"] = "json_schema"
        params["schema"] = _jsonable(_get(src, "schema"))
    elif (
        name == "containsMetric"
        and isinstance(_get(src, "keywords"), (list, tuple))
        and len(_get(src, "keywords")) == 1
        and _get(src, "mode", "any") in ("any", "all")
        and str(_get(src, "keywords")[0])
    ):
        # EvalPort's `contains` takes exactly one substring. With one keyword,
        # eval_lib's "any"/"all" modes are the same check; multi-keyword or
        # mode="none" (absence) has no honest equivalent and stays `custom`.
        grader["type"] = "contains"
        params["substring"] = str(_get(src, "keywords")[0])
        params["ignore_case"] = not bool(_get(src, "case_sensitive", False))
        ns["mode"] = _get(src, "mode", "any")

    if grader["type"] == "custom":
        params["handler"] = _HANDLER_PREFIX + name
        # Carry the deterministic metrics' configuration through even when the
        # typed mapping did not apply, so nothing about the check is lost.
        for opt in ("keywords", "mode", "case_sensitive", "pattern", "full_match", "schema", "references", "aggregation"):
            if _has(src, opt) and _get(src, opt) is not None and not isinstance(src, str):
                ns[opt] = _jsonable(_get(src, opt))

    params[_NS] = ns
    grader["params"] = params
    return grader


def _normalize_metrics(metrics: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    graders: Dict[str, Dict[str, Any]] = {}
    for m in metrics or []:
        g = metric_to_grader(m)
        graders.setdefault(g["id"], g)
    return list(graders.values())


# --------------------------------------------------------------------------- #
# Test case ids
# --------------------------------------------------------------------------- #
def test_case_id(item: Any, index: int, *, id_prefix: str = "tc_") -> str:
    """Deterministic id for the ``index``-th (0-based) test case: the
    ``EvalTestCase.name`` when present, else ``f"{id_prefix}{index + 1:04d}"``.
    ``TestCaseResult`` has no ``name`` field, so results fall back to the
    positional id -- which is why ``to_openeval()`` and ``results_to_openeval()``
    called on the same ordered list always agree (see ``tests``)."""
    name = _get(item, "name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"{id_prefix}{index + 1:04d}"


def _resolve_ids(items: Sequence[Any], ids: Optional[Sequence[str]], id_prefix: str) -> List[str]:
    if ids is not None:
        if len(ids) != len(items):
            raise ValueError(f"ids has {len(ids)} entries but {len(items)} items were given")
        out = [str(i) for i in ids]
    else:
        out = [test_case_id(item, i, id_prefix=id_prefix) for i, item in enumerate(items)]
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for tc_id in out:
        if tc_id in seen:
            seen[tc_id] += 1
            unique.append(f"{tc_id}__{seen[tc_id]}")
        else:
            seen[tc_id] = 1
            unique.append(tc_id)
    return unique


def _flatten_results(results: Iterable[Any]) -> List[Any]:
    """Accept ``eval_lib.evaluate()``'s ``[(None, [TestCaseResult]), ...]``
    as well as a flat ``[TestCaseResult, ...]`` (or dicts)."""
    flat: List[Any] = []
    for item in results:
        if isinstance(item, tuple) or (isinstance(item, list) and not isinstance(item, Mapping)):
            for part in item:
                if part is None:
                    continue
                if isinstance(part, (list, tuple)):
                    flat.extend(p for p in part if p is not None)
                else:
                    flat.append(part)
        else:
            flat.append(item)
    return flat


# --------------------------------------------------------------------------- #
# Suite direction
# --------------------------------------------------------------------------- #
def _test_case_to_openeval(item: Any, tc_id: str, grader_ids: List[str]) -> Dict[str, Any]:
    raw_input = _get(item, "input")
    if isinstance(raw_input, (list, tuple)):
        input_value: Union[str, List[str]] = [str(v) for v in raw_input]
    else:
        input_value = "" if raw_input is None else str(raw_input)
    if not input_value:
        raise ValueError(f"test case {tc_id!r} has an empty 'input' (EvalPort requires minLength 1)")

    ns: Dict[str, Any] = {}
    tc: Dict[str, Any] = {"id": tc_id, "input": input_value, "graders": list(grader_ids)}

    expected = _get(item, "expected_output")
    if expected is not None:
        tc["expected_output"] = str(expected)

    # EvalPort TestCase has no actual_output slot (that lives on Result), but
    # eval_lib bakes the model's answer into the test case itself; keep it so
    # from_openeval() can rebuild a complete EvalTestCase.
    actual = _get(item, "actual_output")
    if actual is not None:
        ns["actual_output"] = str(actual)

    for field in _TC_LIST_FIELDS:
        value = _str_list(_get(item, field))
        if value is not None:
            tc[field] = value

    name = _get(item, "name")
    if isinstance(name, str) and name.strip() and name.strip() != tc_id:
        # When the name *is* the id (the default), it is recoverable from
        # TestCase.id, so only a differing name needs to be stored.
        ns["name"] = name

    for field in _TC_EXTRA_FIELDS:
        value = _get(item, field)
        if value is not None:
            ns[field] = _jsonable(value)

    if ns:
        tc["metadata"] = {_NS: ns}
    return tc


def to_openeval(
    test_cases: Sequence[Any],
    metrics: Optional[Iterable[Any]] = None,
    *,
    suite_id: str = "eval_ai_library_suite",
    name: Optional[str] = None,
    description: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    id_prefix: str = "tc_",
) -> Dict[str, Any]:
    """Convert ``eval_lib`` test cases into an EvalPort ``Suite`` (plain dict).

    ``test_cases``: ``EvalTestCase`` objects, ``TestCaseResult`` objects (they
    carry the same input/expected/context fields), or dicts with those field
    names. ``eval_lib.evaluate()``'s ``[(None, [TestCaseResult])]`` output is
    accepted too.

    ``metrics``: the metrics the suite is graded with -- real ``MetricPattern``
    instances (best: typed ``regex``/``json_schema``/``contains`` graders),
    ``MetricResult`` objects, dicts or bare names (see ``metric_to_grader``).
    When omitted, graders are derived from each item's ``metrics_data`` (so a
    list of ``TestCaseResult`` needs nothing else); if neither is available a
    single ``custom`` grader ``gr_eval_ai_library`` is declared, because
    EvalPort requires at least one grader per test case.

    Test case ids come from ``EvalTestCase.name`` or ``tc_0001``-style
    positions (override with ``ids``); ``results_to_openeval()`` uses the same
    rule so a Suite and ResultSet built from the same ordered list join on id.

    Pass the result to ``openeval.validate.validate_suite()`` to confirm
    compliance, or ``json.dump()`` it to share as a ``.json`` suite file.
    """
    items = _flatten_results(test_cases)
    if not items:
        raise ValueError("test_cases is empty -- nothing to convert")
    for item in items:
        if _has(item, "dialogue") and not _has(item, "input"):
            raise ValueError(
                "ConversationalTestCaseResult / ConversationalEvalTestCase is not "
                "supported by this adapter yet; pass single-turn EvalTestCase / "
                "TestCaseResult objects"
            )

    tc_ids = _resolve_ids(items, ids, id_prefix)

    graders: List[Dict[str, Any]]
    if metrics is not None:
        graders = _normalize_metrics(metrics)
        if not graders:
            raise ValueError("metrics was given but is empty")
    else:
        derived: Dict[str, Dict[str, Any]] = {}
        for item in items:
            for mr in _get(item, "metrics_data", None) or []:
                g = metric_to_grader(mr)
                derived.setdefault(g["id"], g)
        graders = list(derived.values())
        if not graders:
            graders = [
                {
                    "id": _DEFAULT_GRADER_ID,
                    "type": "custom",
                    "params": {"handler": _HANDLER_PREFIX + "evaluate"},
                    "description": (
                        "Placeholder: no eval_lib metrics were supplied to to_openeval(); "
                        "the suite is graded by whichever eval_lib metrics the runner applies."
                    ),
                }
            ]

    grader_ids = [g["id"] for g in graders]
    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "graders": graders,
        "test_cases": [_test_case_to_openeval(item, tc_id, grader_ids) for item, tc_id in zip(items, tc_ids)],
        "metadata": {_NS: {"source": "eval_ai_library", "adapter_version": __version__}},
    }
    if name is not None:
        suite["name"] = name
    if description is not None:
        suite["description"] = description
    return suite


def from_openeval(suite: Mapping[str, Any], *, as_models: bool = False) -> List[Any]:
    """Reverse ``to_openeval()``: recover ``EvalTestCase``-shaped dicts
    (``input``, ``actual_output``, ``expected_output``, ``retrieval_context``,
    ``tools_called``, ``expected_tools``, ``name`` and any reliability /
    ``extra_fields`` data this module stashed in ``metadata.eval_ai_library``).

    Works on any EvalPort suite, not only ones this module produced: a foreign
    test case simply yields ``actual_output=""`` (eval_lib requires the field;
    fill it in after running your system) and no metadata extras.

    ``as_models=True`` imports ``eval_lib.testcases_schema.EvalTestCase``
    lazily and returns real pydantic instances; it raises ``ImportError`` if
    ``eval-ai-library`` is not installed.
    """
    out: List[Any] = []
    for tc in suite.get("test_cases", []) or []:
        ns = ((tc.get("metadata") or {}).get(_NS)) or {}
        raw_input = tc.get("input", "")
        if isinstance(raw_input, list):
            raw_input = "\n".join(str(v) for v in raw_input)
        record: Dict[str, Any] = {
            "input": raw_input,
            "actual_output": ns.get("actual_output", ""),
            "expected_output": tc.get("expected_output"),
            "retrieval_context": tc.get("retrieval_context"),
            "tools_called": tc.get("tools_called"),
            "expected_tools": tc.get("expected_tools"),
            "name": ns.get("name", tc.get("id")),
        }
        for field in _TC_EXTRA_FIELDS:
            if field in ns:
                record[field] = ns[field]
        out.append(record)

    if as_models:
        from eval_lib.testcases_schema import EvalTestCase  # lazy, on request only

        return [EvalTestCase(**r) for r in out]
    return out


# --------------------------------------------------------------------------- #
# Result direction
# --------------------------------------------------------------------------- #
def _score_to_openeval(raw: Any) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(spec_score, raw_score)``. Spec score is ``None`` for a
    missing/NaN score, else the raw value clamped into ``[0, 1]``."""
    if raw is None or isinstance(raw, bool):
        return (None, None) if raw is None else (1.0 if raw else 0.0, float(raw))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, None
    if math.isnan(value) or math.isinf(value):
        return None, None
    return max(0.0, min(1.0, value)), value


def _metric_result_to_grader_result(mr: Any, grader_by_id: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    grader = metric_to_grader(mr)
    grader_id = grader["id"]
    grader_type = grader_by_id.get(grader_id, grader)["type"]

    score, raw = _score_to_openeval(_get(mr, "score"))
    success = bool(_get(mr, "success", False))

    ns: Dict[str, Any] = {"metric": str(_get(mr, "name"))}
    if raw is not None:
        ns["score"] = raw
    elif _get(mr, "score") is not None:
        ns["score"] = _jsonable(_get(mr, "score"))
    for field in ("threshold", "evaluation_cost", "evaluation_model"):
        value = _get(mr, field)
        if value is not None:
            ns[field] = _jsonable(value)
    log = _get(mr, "evaluation_log")
    if log is not None:
        ns["evaluation_log"] = _jsonable(log)
        if isinstance(log, Mapping) and log.get("skipped"):
            # eval_lib's empty-actual_output guard: it *reports* score 0.0 /
            # success False (and its own aggregate counts it that way), so the
            # score is kept as reported rather than nulled; the skip is flagged.
            ns["skipped"] = True
            if log.get("skip_reason") is not None:
                ns["skip_reason"] = log.get("skip_reason")

    gr: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": grader_type,
        "score": score,
        "passed": success,
        "metadata": {_NS: ns},
    }
    reason = _get(mr, "reason")
    if reason is not None:
        gr["reason"] = str(reason)
    return gr


_PROVIDER_ERROR_HINTS = ("api", "provider", "litellm", "openai", "anthropic", "ratelimit", "rate_limit", "auth", "connection", "http")


def _error_to_openeval(err: Any) -> Dict[str, Any]:
    """Coerce an exception / str / ``{"type","message",...}`` dict into an
    EvalPort ``Result.error`` (``type`` in timeout|provider_error|runner_error)."""
    if isinstance(err, Mapping):
        etype = str(err.get("type") or "runner_error")
        message = err.get("message")
        out: Dict[str, Any] = {"type": etype if etype in ("timeout", "provider_error", "runner_error") else "runner_error"}
        if message is not None:
            out["message"] = str(message)
        if err.get("code") is not None:
            out["code"] = err["code"] if isinstance(err["code"], (str, int)) else str(err["code"])
        if isinstance(err.get("retryable"), bool):
            out["retryable"] = err["retryable"]
        if etype not in ("timeout", "provider_error", "runner_error"):
            out["message"] = f"[{etype}] " + out.get("message", "")
        return out
    if isinstance(err, BaseException):
        cls = type(err).__name__
        lowered = cls.lower()
        if "timeout" in lowered:
            etype = "timeout"
        elif any(h in lowered for h in _PROVIDER_ERROR_HINTS):
            etype = "provider_error"
        else:
            etype = "runner_error"
        return {"type": etype, "message": f"{cls}: {err}"}
    return {"type": "runner_error", "message": str(err)}


def results_to_openeval(
    results: Sequence[Any],
    metrics: Optional[Iterable[Any]] = None,
    *,
    suite_id: str = "eval_ai_library_suite",
    run_id: str,
    started_at: str,
    completed_at: Optional[str] = None,
    test_cases: Optional[Sequence[Any]] = None,
    ids: Optional[Sequence[str]] = None,
    id_prefix: str = "tc_",
    errors: Optional[Mapping[Union[int, str], Any]] = None,
    durations_ms: Optional[Mapping[Union[int, str], Union[int, float]]] = None,
    provider: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert ``eval_lib`` evaluation results into an EvalPort ``ResultSet``.

    ``results``: ``eval_lib.evaluate()``'s return value
    (``[(None, [TestCaseResult]), ...]``) or a flat list of ``TestCaseResult``
    objects / dicts shaped like them.

    One ``Result`` per ``TestCaseResult`` (``passed`` = its ``success``,
    ``actual_output`` carried through) and one ``GraderResult`` per
    ``MetricResult`` in ``metrics_data``: ``grader_id = gr_<snake_case(name)>``,
    ``score`` = the metric's 0..1 score (clamped; ``null`` when the upstream
    score is ``None``/NaN), ``passed`` = its ``success``, ``reason`` preserved,
    and ``threshold`` / raw ``score`` / ``evaluation_cost`` /
    ``evaluation_model`` / ``evaluation_log`` under
    ``metadata.eval_ai_library`` so the conversion is lossless.

    ``metrics``: pass the same metric objects you passed to ``to_openeval()``
    so ``GraderResult.type`` matches the Suite's typed graders (a
    ``RegexMatchMetric`` instance yields ``regex``; from a bare
    ``MetricResult`` it can only be ``custom``).

    ``test_cases``: the ``EvalTestCase`` list you passed to
    ``eval_lib.evaluate()``, in the same order. ``TestCaseResult`` drops
    ``EvalTestCase.name``, so without this (or explicit ``ids``) results get
    positional ``tc_0001`` ids; with it they get exactly the ids
    ``to_openeval(test_cases)`` produced.

    ``errors``: optional ``{index_or_test_case_id: Exception | str | dict}``
    for test cases whose evaluation raised instead of producing a
    ``TestCaseResult`` (eval_lib has no error slot of its own). Those entries
    become ``Result.error`` with ``passed=False`` and no grader results; when
    the key is an index beyond ``results``, a result is appended for it.

    ``durations_ms``: optional per-case wall time (same keys), stored as
    ``Result.duration_ms``. ``provider``: optional ``ResultSet.provider``
    (e.g. ``{"model": "gpt-4o"}``).
    """
    items = _flatten_results(results)
    errors = dict(errors or {})
    if not items and not errors:
        raise ValueError("results is empty -- nothing to convert")
    for item in items:
        if _has(item, "dialogue") and not _has(item, "input"):
            raise ValueError(
                "ConversationalTestCaseResult is not supported by this adapter yet; "
                "pass single-turn TestCaseResult objects"
            )

    # Each entry is (item_or_None, position). Positional error keys beyond the
    # successful results become extra, item-less entries at their position.
    entries: List[Tuple[Any, int]] = [(item, i) for i, item in enumerate(items)]
    entries += [(None, k) for k in sorted(k for k in errors if isinstance(k, int) and k >= len(items))]
    if ids is None and test_cases is not None:
        source = _flatten_results(test_cases)
        if len(source) != len(entries):
            raise ValueError(f"test_cases has {len(source)} entries but {len(entries)} results were given")
        ids = [test_case_id(tc, i, id_prefix=id_prefix) for i, tc in enumerate(source)]
    if ids is not None:
        if len(ids) != len(entries):
            raise ValueError(f"ids has {len(ids)} entries but {len(entries)} results were given")
        tc_ids = _resolve_ids([e[0] for e in entries], ids, id_prefix)
    else:
        tc_ids = _resolve_ids([e[0] for e in entries], [test_case_id(e[0], e[1], id_prefix=id_prefix) for e in entries], id_prefix)

    grader_by_id: Dict[str, Dict[str, Any]] = {g["id"]: g for g in _normalize_metrics(metrics)}
    durations = dict(durations_ms or {})

    def _lookup(mapping: Mapping[Any, Any], position: int, tc_id: str) -> Any:
        if position in mapping:
            return mapping[position]
        if tc_id in mapping:
            return mapping[tc_id]
        return None

    out_results: List[Dict[str, Any]] = []
    total_cost = 0.0
    score_sum = 0.0
    score_n = 0
    by_grader: Dict[str, Dict[str, Any]] = {}

    for (item, position), tc_id in zip(entries, tc_ids):
        result: Dict[str, Any] = {"test_case_id": tc_id, "grader_results": [], "passed": False}
        err = _lookup(errors, position, tc_id)

        if item is not None:
            actual = _get(item, "actual_output")
            if actual is not None:
                result["actual_output"] = str(actual)
            result["passed"] = bool(_get(item, "success", False))
            for mr in _get(item, "metrics_data", None) or []:
                gr = _metric_result_to_grader_result(mr, grader_by_id)
                result["grader_results"].append(gr)
                cost = _get(mr, "evaluation_cost")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    total_cost += float(cost)
                agg = by_grader.setdefault(gr["grader_id"], {"passed": 0, "failed": 0, "_sum": 0.0, "_n": 0})
                agg["passed" if gr["passed"] else "failed"] += 1
                if gr["score"] is not None:
                    agg["_sum"] += gr["score"]
                    agg["_n"] += 1
                    score_sum += gr["score"]
                    score_n += 1
            ns: Dict[str, Any] = {}
            for field in ("input", "expected_output", "retrieval_context", "tools_called", "expected_tools"):
                value = _get(item, field)
                if value is not None:
                    ns[field] = _jsonable(value)
            if ns:
                result["metadata"] = {_NS: ns}

        if err is not None:
            result["error"] = _error_to_openeval(err)
            result["passed"] = False

        duration = _lookup(durations, position, tc_id)
        if duration is not None:
            result["duration_ms"] = int(round(float(duration)))

        out_results.append(result)

    passed_n = sum(1 for r in out_results if r["passed"])
    total_n = len(out_results)
    summary: Dict[str, Any] = {
        "total": total_n,
        "passed": passed_n,
        "failed": total_n - passed_n,
        "pass_rate": (passed_n / total_n) if total_n else 0.0,
    }
    if score_n:
        summary["avg_score"] = score_sum / score_n
    if by_grader:
        summary["by_grader"] = {
            gid: {
                "passed": agg["passed"],
                "failed": agg["failed"],
                **({"avg_score": agg["_sum"] / agg["_n"]} if agg["_n"] else {}),
            }
            for gid, agg in by_grader.items()
        }

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": out_results,
        "summary": summary,
        "runner": {"name": "eval_ai_library"},
        "metadata": {
            _NS: {
                "source": "eval_ai_library",
                "adapter_version": __version__,
                "total_evaluation_cost": total_cost,
            },
            # PROPOSED (evalport Discussion #49, alternative B), NOT in the spec on main: a Result with
            # some null-scored and some scored graders must declare how its passed was derived.
            # `producer`: Result.passed is eval_lib's own per-item `success`, not re-derived from the metrics.
            # On main the key is free-form metadata and nothing reads it.
            "openeval": {"aggregation": {"strategy": "producer"}},
        },
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    if provider:
        result_set["provider"] = {k: v for k, v in provider.items() if k in ("model", "api_base", "temperature", "max_tokens", "extra")}
    return result_set
