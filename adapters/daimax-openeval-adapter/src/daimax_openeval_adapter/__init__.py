"""Convert daimax-appbench (https://github.com/open-daimax/daimax-appbench)
evaluation runs and benchmark items to/from the EvalPort open evaluation
format.

Filed and built following open-daimax/daimax-appbench#9, where the daimax
maintainer asked for exactly this: a standalone package, no changes to the
``evalapp`` core, ``ResultSet``-first. Every shape below was read from the
real ``evalapp`` source (not guessed) before writing this module:

- ``evalapp/evaluation/results/models/execution.py``: ``EvalRun`` (``run_id``,
  ``generator_name``, ``run_type``, ``sample_source``, ``timestamp`` -- a
  *naive* ``datetime.now().isoformat()`` string with no offset --
  ``prompt_results``, ``summary``), ``PromptResult`` (``prompt_id``,
  ``sample_id``, ``platform``, ``generator_name``, ``generation_success``,
  ``generation_duration`` in float seconds, ``error_message``,
  ``process_data``, ``result_data``, ``test_results``, and the three optional
  top-level metrics ``success_rate`` / ``quality`` / ``experience``; the
  computed ``item_id`` is ``sample_id or prompt_id``), ``TestCaseResult``
  (``test_case_id``, ``passed``, ``status`` in ``"PASS"``/``"FAIL"``/
  ``"SKIPPED"``, ``details``, ``duration`` float seconds,
  ``report_started_at``/``report_generated_at`` unix seconds,
  ``verifications``).
- ``evalapp/evaluation/metrics/models.py``: ``SuccessRateMetrics``,
  ``QualityMetrics``, ``ExperienceMetrics``. All three carry a
  ``composite_score`` on a 0..100 scale; ``ExperienceMetrics.aesthetics_score``
  is 0..10 (or ``None`` when not scored).
- ``evalapp/benchset/samples/models.py``: ``EvalSample`` (``sample_id``,
  ``title``, ``requirement``, ``platforms``, ``app_type``, ``complexity``,
  ``top_category``, ``core_functions``, ``constraints``, ``notes``,
  ``requires_backend``, ``requires_auth``, ...).
- ``evalapp/benchset/testcases/models.py``: ``TestCase`` (``id``, ``name``,
  ``description``, ``steps``, ``expected_result``, ``priority`` in
  ``P0``/``P1``/``P2``, ``category``).

Mapping summary (the README "Mapping" section has the full table; the
maintainer's constraints from #9 are quoted there):

- One EvalPort ``Result`` per ``PromptResult``; ``test_case_id`` is daimax's
  ``item_id`` (``sample_id or prompt_id``).
- THREE synthetic graders, one per top-level metric -- ``gr_daimax_success_rate``,
  ``gr_daimax_quality``, ``gr_daimax_experience`` -- each preserving its OWN
  metric's breakdown in its own ``metadata.daimax`` (the maintainer was
  explicit that ``usecase_completeness`` / ``stability_deduction`` /
  ``backend_deduction`` belong to ``QualityMetrics`` and must not be flattened
  into a single grader). ``score`` is ``composite_score / 100``; a metric that
  is ``None`` yields ``score: null``.
- One ``GraderResult`` per ``TestCaseResult`` with grader id ``gr_daimax_e2e``
  and the namespaced type ``org.daimax.vision_e2e_step``; the shared Suite
  grader definition keeps ``daimax:vision_e2e_step`` as ``params.handler``.
  EvalPort's ``grader.json`` treats ``type`` as an open string (any
  non-well-known type is validated like ``custom``, i.e. ``params.handler``
  is required), so the namespaced type is used directly rather than falling
  back to ``type: "custom"``. ``"SKIPPED"`` checks get ``score: null``.
- Scores and statuses are kept intact; ``passed`` comes from native booleans
  (``generation_success``, ``TestCaseResult.passed``) unless the caller opts
  into an explicit ``pass_threshold``, which is then recorded as an adapter
  policy in metadata. There is no daimax-wide 70-point cutoff.
- ``P0``/``P1``/``P2`` -> ``3``/``2``/``1`` is recorded as
  ``metadata.daimax.priority_weight`` and documented as a project convention.
"""
from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "run_to_openeval",
    "run_from_openeval",
    "to_openeval",
    "from_openeval",
    "PRIORITY_WEIGHTS",
    "GRADER_SUCCESS_RATE",
    "GRADER_QUALITY",
    "GRADER_EXPERIENCE",
    "GRADER_E2E",
    "E2E_GRADER_TYPE",
    "E2E_GRADER_HANDLER",
    "__version__",
]
__version__ = "0.1.0"

_NS = "daimax"

# Project convention (open-daimax/daimax-appbench#9): daimax's TestCase
# priority is a closed Literal["P0", "P1", "P2"] with P0 the most important.
# EvalPort has no priority field, so the adapter records an integer weight in
# metadata.daimax.priority_weight. The maintainer accepted P0/P1/P2 -> 3/2/1
# on the condition that it be documented as a convention, not a daimax-native
# quantity -- it is NOT used to compute any score in this module.
PRIORITY_WEIGHTS: Dict[str, int] = {"P0": 3, "P1": 2, "P2": 1}

GRADER_SUCCESS_RATE = "gr_daimax_success_rate"
GRADER_QUALITY = "gr_daimax_quality"
GRADER_EXPERIENCE = "gr_daimax_experience"
GRADER_E2E = "gr_daimax_e2e"

# Namespaced grader type for the per-TestCaseResult E2E check (maintainer
# request in #9). grader.json's `type` is an open string: anything outside the
# well-known list is validated exactly like `custom`, requiring
# `params.handler`, so we keep the original handler string there.
E2E_GRADER_TYPE = "org.daimax.vision_e2e_step"
E2E_GRADER_HANDLER = "daimax:vision_e2e_step"

_SYNTHETIC_TYPE = "custom"
_SYNTHETIC_HANDLERS = {
    GRADER_SUCCESS_RATE: "daimax:success_rate_metrics",
    GRADER_QUALITY: "daimax:quality_metrics",
    GRADER_EXPERIENCE: "daimax:experience_metrics",
}
_SYNTHETIC_METRIC_FIELDS = {
    GRADER_SUCCESS_RATE: "success_rate",
    GRADER_QUALITY: "quality",
    GRADER_EXPERIENCE: "experience",
}
_SYNTHETIC_DESCRIPTIONS = {
    GRADER_SUCCESS_RATE: (
        "daimax SuccessRateMetrics: weighted average of initial_generation_rate, "
        "issue_fix_rate and requirement_extension_rate (each 0..100, with its "
        "own weight and reason). score = composite_score / 100."
    ),
    GRADER_QUALITY: (
        "daimax QualityMetrics: usecase_completeness (E2E pass rate) minus "
        "stability_deduction minus backend_deduction, floor 0, all on 0..100. "
        "score = composite_score / 100."
    ),
    GRADER_EXPERIENCE: (
        "daimax ExperienceMetrics: duration_score (60%), package_size_score "
        "(20%), aesthetics_score (20%, native scale 0..10), tokens displayed "
        "only. score = composite_score / 100."
    ),
}

_SUITE_GRADERS: List[Dict[str, Any]] = [
    {
        "id": gid,
        "type": _SYNTHETIC_TYPE,
        "params": {"handler": _SYNTHETIC_HANDLERS[gid]},
        "description": _SYNTHETIC_DESCRIPTIONS[gid],
    }
    for gid in (GRADER_SUCCESS_RATE, GRADER_QUALITY, GRADER_EXPERIENCE)
] + [
    {
        "id": GRADER_E2E,
        "type": E2E_GRADER_TYPE,
        "params": {"handler": E2E_GRADER_HANDLER},
        "description": (
            "One daimax TestCaseResult from the independent ai-ui-test / "
            "Midscene E2E engine (PromptResult.test_results, the authoritative "
            "E2E source). PASS -> score 1.0, FAIL -> 0.0, SKIPPED -> null."
        ),
    },
]


# ---------------------------------------------------------------------------
# Duck-typed access helpers: real pydantic models or plain dicts both work.
# ---------------------------------------------------------------------------


def _dump(obj: Any) -> Any:
    """Return a plain-dict view of ``obj``: ``model_dump()`` for pydantic
    models, ``dict`` for mappings, ``None`` passthrough."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    dump = getattr(obj, "dict", None)  # pydantic v1 fallback
    if callable(dump):
        return dump()
    return obj


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _item_id(prompt_result: Any) -> str:
    """Mirror ``PromptResult.item_id``: ``sample_id or prompt_id``."""
    explicit = _get(prompt_result, "item_id")
    if explicit:
        return str(explicit)
    sample_id = _get(prompt_result, "sample_id") or ""
    prompt_id = _get(prompt_result, "prompt_id") or ""
    item_id = sample_id or prompt_id
    if not item_id:
        raise ValueError(
            "PromptResult requires at least one of prompt_id or sample_id to be non-empty"
        )
    return str(item_id)


def _to_iso(raw: Any, assume_timezone: Optional[tzinfo]) -> Dict[str, Any]:
    """Normalize daimax's naive ``EvalRun.timestamp`` (``datetime.now()
    .isoformat()``, no offset) to an offset-bearing ISO 8601 string.

    Returns ``{"iso": str, "raw": str, "assumed_timezone": str|None}``. When
    the input already carries an offset it is kept as-is. When it is naive
    (the daimax default), ``assume_timezone`` is attached and recorded so a
    consumer can see the assumption was made by this adapter rather than by
    daimax. ``assume_timezone=None`` with a naive input leaves the string
    unchanged and records ``assumed_timezone: null``.
    """
    if isinstance(raw, datetime):
        dt = raw
        raw_str = raw.isoformat()
    else:
        raw_str = str(raw)
        try:
            dt = datetime.fromisoformat(raw_str.replace("Z", "+00:00"))
        except ValueError:
            return {"iso": raw_str, "raw": raw_str, "assumed_timezone": None}
    assumed: Optional[str] = None
    if dt.tzinfo is None and assume_timezone is not None:
        dt = dt.replace(tzinfo=assume_timezone)
        assumed = str(assume_timezone)
    return {"iso": dt.isoformat(), "raw": raw_str, "assumed_timezone": assumed}


def _scale_100(value: Any) -> Optional[float]:
    if value is None:
        return None
    score = float(value) / 100.0
    return min(1.0, max(0.0, score))


def _scale_10(value: Any) -> Optional[float]:
    if value is None:
        return None
    score = float(value) / 10.0
    return min(1.0, max(0.0, score))


def _test_case_index(test_cases: Any) -> Dict[str, Dict[str, Any]]:
    """Index daimax ``TestCase`` objects (or dicts, or ``TestDesignOutput``
    objects) by ``id`` so E2E grader results can be enriched with
    ``priority``/``priority_weight``/``name``/``category``."""
    index: Dict[str, Dict[str, Any]] = {}
    if not test_cases:
        return index
    for tc in test_cases:
        nested = _get(tc, "test_cases")
        if nested is not None and _get(tc, "id") is None:
            # TestDesignOutput(prompt_id, platform, test_cases=[...])
            index.update(_test_case_index(nested))
            continue
        d = _dump(tc)
        if isinstance(d, dict) and d.get("id"):
            index[str(d["id"])] = d
    return index


def _priority_weight(priority: Any) -> Optional[int]:
    if priority is None:
        return None
    return PRIORITY_WEIGHTS.get(str(priority).upper())


# ---------------------------------------------------------------------------
# ResultSet export (primary deliverable, #9)
# ---------------------------------------------------------------------------


def _synthetic_grader_result(
    grader_id: str,
    metric: Any,
    *,
    generation_success: bool,
    pass_threshold: Optional[float],
) -> Dict[str, Any]:
    metric_dump = _dump(metric)
    composite = _get(metric_dump, "composite_score") if metric_dump is not None else None
    score = _scale_100(composite)

    if pass_threshold is None:
        # Adapter policy "native_booleans": daimax's metrics carry no
        # pass/fail of their own, so the only native per-item verdict is
        # generation_success. Nothing is derived from the score.
        passed = bool(generation_success)
    else:
        # Adapter policy "composite_threshold": explicit, caller-supplied,
        # on daimax's native 0..100 scale. A missing metric cannot reach a
        # threshold verdict, so it is recorded as not passed.
        passed = composite is not None and float(composite) >= float(pass_threshold)

    meta: Dict[str, Any] = {"metric": _SYNTHETIC_METRIC_FIELDS[grader_id]}
    if metric_dump is None:
        meta["skip_reason"] = "metric_not_computed"
    else:
        meta["composite_score_raw"] = composite
        # The FULL native breakdown of THIS metric only -- Quality keeps
        # usecase_completeness/stability_deduction/backend_deduction etc.,
        # SuccessRate keeps its three rates + weights + reasons, Experience
        # keeps duration/package/token/aesthetics. Nothing is cross-mixed.
        meta["breakdown"] = dict(metric_dump)
        if grader_id == GRADER_EXPERIENCE:
            meta["normalized"] = {
                "duration_score": _scale_100(_get(metric_dump, "duration_score")),
                "package_size_score": _scale_100(_get(metric_dump, "package_size_score")),
                # aesthetics_score is natively 0..10 (None = not scored)
                "aesthetics_score": _scale_10(_get(metric_dump, "aesthetics_score")),
            }
        elif grader_id == GRADER_QUALITY:
            meta["normalized"] = {
                "usecase_completeness": _scale_100(_get(metric_dump, "usecase_completeness")),
                "stability_score": _scale_100(_get(metric_dump, "stability_score")),
                "backend_completeness": _scale_100(_get(metric_dump, "backend_completeness")),
                "compliance_score": _scale_100(_get(metric_dump, "compliance_score")),
            }
        elif grader_id == GRADER_SUCCESS_RATE:
            meta["normalized"] = {
                "initial_generation_rate": _scale_100(_get(metric_dump, "initial_generation_rate")),
                "issue_fix_rate": _scale_100(_get(metric_dump, "issue_fix_rate")),
                "requirement_extension_rate": _scale_100(
                    _get(metric_dump, "requirement_extension_rate")
                ),
            }

    result: Dict[str, Any] = {
        "grader_id": grader_id,
        "type": _SYNTHETIC_TYPE,
        "score": score,
        "passed": passed,
        "metadata": {_NS: meta},
    }
    reason = None
    if metric_dump is not None:
        if grader_id == GRADER_QUALITY:
            reason = _get(metric_dump, "usecase_reason")
        elif grader_id == GRADER_SUCCESS_RATE:
            reason = _get(metric_dump, "initial_generation_reason")
        elif grader_id == GRADER_EXPERIENCE:
            reason = _get(metric_dump, "duration_reason")
    if reason:
        result["reason"] = str(reason)
    return result


def _e2e_grader_result(
    test_result: Any, tc_index: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    tr = _dump(test_result)
    test_case_id = str(_get(tr, "test_case_id"))
    status = str(_get(tr, "status") or "")
    passed = bool(_get(tr, "passed", False))
    skipped = status.upper() == "SKIPPED"

    if skipped:
        # Unexecuted check -> EvalPort's score: null convention (#9).
        score: Optional[float] = None
    else:
        score = 1.0 if passed else 0.0

    duration_s = _get(tr, "duration", 0.0) or 0.0
    meta: Dict[str, Any] = {
        "test_case_id": test_case_id,
        "status": status,
        "duration_s_raw": duration_s,
        "duration_ms": int(round(float(duration_s) * 1000)),
    }
    if skipped:
        meta["skip_reason"] = "SKIPPED"
    for key in ("report_path", "report_started_at", "report_generated_at", "verifications"):
        value = _get(tr, key)
        if value not in (None, "", 0, 0.0):
            meta[key] = value

    tc = tc_index.get(test_case_id)
    if tc is not None:
        for key in ("name", "category", "priority"):
            if tc.get(key) not in (None, ""):
                meta[key] = tc[key]
        weight = _priority_weight(tc.get("priority"))
        if weight is not None:
            meta["priority_weight"] = weight
            meta["priority_weight_convention"] = "P0/P1/P2 -> 3/2/1 (adapter convention, see README)"

    result: Dict[str, Any] = {
        "grader_id": GRADER_E2E,
        "type": E2E_GRADER_TYPE,
        "score": score,
        "passed": passed,
        "metadata": {_NS: meta},
    }
    details = _get(tr, "details")
    if details:
        result["reason"] = str(details)
    return result


def _error_for(prompt_result: Any) -> Optional[Dict[str, Any]]:
    """``Result.error`` only when daimax's own ``generation_success`` is
    False. The generator is the system under test -- the EvalPort analogue of
    the provider that should have produced the output -- so a failed
    generation is ``provider_error``; when daimax's ``process_data.error_type``
    names a timeout it is ``timeout``. Nothing about the evaluation harness
    itself failed, so ``runner_error`` is deliberately not used."""
    if _get(prompt_result, "generation_success", False):
        return None
    process = _dump(_get(prompt_result, "process_data")) or {}
    error_type = str(process.get("error_type") or "").strip()
    message = _get(prompt_result, "error_message") or process.get("error_message") or ""
    err: Dict[str, Any] = {
        "type": "timeout" if "timeout" in error_type.lower() else "provider_error",
    }
    if message:
        err["message"] = str(message)
    if error_type:
        err["code"] = error_type
    return err


def run_to_openeval(
    eval_run: Any,
    *,
    suite_id: Optional[str] = None,
    pass_threshold: Optional[float] = None,
    test_cases: Optional[Iterable[Any]] = None,
    assume_timezone: Optional[tzinfo] = timezone.utc,
) -> Dict[str, Any]:
    """Convert one daimax ``EvalRun`` (a pydantic model, or a dict shaped like
    ``EvalRun.model_dump()`` / the ``results.json`` daimax writes) into an
    EvalPort ``ResultSet`` (plain dict).

    ``suite_id`` defaults to ``EvalRun.sample_source`` or ``"daimax_appbench"``.

    ``pass_threshold`` (default ``None``) is the ONLY knob that derives a
    verdict from a score. ``None`` means every ``passed`` comes from a native
    daimax boolean (``generation_success`` for the three metric graders,
    ``TestCaseResult.passed`` for E2E checks). A number (daimax's native
    0..100 scale, e.g. ``70``) makes each metric grader's ``passed`` equal
    ``composite_score >= pass_threshold``; the policy is then written to
    ``metadata.daimax.adapter_policy`` on the ResultSet and each Result, so no
    consumer can mistake it for a daimax-native cutoff. ``Result.passed`` is
    always ``all(grader.passed)``.

    ``test_cases`` optionally supplies daimax ``TestCase`` objects (or
    ``TestDesignOutput`` objects, or dicts) so each E2E grader result can be
    enriched with ``name``/``category``/``priority``/``priority_weight``.

    ``assume_timezone`` is attached to daimax's naive ``EvalRun.timestamp``
    (default UTC); the raw string and the assumption are kept in metadata.
    """
    run = eval_run
    prompt_results = _get(run, "prompt_results") or []
    if not prompt_results:
        raise ValueError("EvalRun.prompt_results is empty -- nothing to convert")
    if pass_threshold is not None and not (0 <= float(pass_threshold) <= 100):
        raise ValueError("pass_threshold must be on daimax's native 0..100 scale")

    tc_index = _test_case_index(test_cases)
    ts = _to_iso(_get(run, "timestamp") or datetime.now().isoformat(), assume_timezone)

    policy: Dict[str, Any]
    if pass_threshold is None:
        policy = {
            "name": "native_booleans",
            "pass_threshold": None,
            "description": (
                "passed is taken only from daimax's own booleans "
                "(generation_success, TestCaseResult.passed); no verdict is "
                "derived from any score."
            ),
        }
    else:
        policy = {
            "name": "composite_threshold",
            "pass_threshold": float(pass_threshold),
            "pass_threshold_scale": "0..100 (daimax composite_score scale)",
            "pass_threshold_normalized": float(pass_threshold) / 100.0,
            "description": (
                "Metric graders pass when composite_score >= pass_threshold. "
                "This threshold is an adapter-side policy chosen by the caller, "
                "not a daimax-wide pass criterion."
            ),
        }

    results: List[Dict[str, Any]] = []
    total_duration_ms = 0
    for pr in prompt_results:
        item_id = _item_id(pr)
        generation_success = bool(_get(pr, "generation_success", False))
        grader_results: List[Dict[str, Any]] = []
        for gid in (GRADER_SUCCESS_RATE, GRADER_QUALITY, GRADER_EXPERIENCE):
            grader_results.append(
                _synthetic_grader_result(
                    gid,
                    _get(pr, _SYNTHETIC_METRIC_FIELDS[gid]),
                    generation_success=generation_success,
                    pass_threshold=pass_threshold,
                )
            )
        for tr in _get(pr, "test_results") or []:
            grader_results.append(_e2e_grader_result(tr, tc_index))

        duration_s = float(_get(pr, "generation_duration", 0.0) or 0.0)
        duration_ms = int(round(duration_s * 1000))
        total_duration_ms += duration_ms

        result_data = _dump(_get(pr, "result_data")) or {}
        process_data = _dump(_get(pr, "process_data")) or {}
        meta: Dict[str, Any] = {
            "prompt_id": _get(pr, "prompt_id") or "",
            "sample_id": _get(pr, "sample_id") or "",
            "item_type": _get(pr, "item_type") or "prompt",
            "platform": _get(pr, "platform") or "",
            "generator_name": _get(pr, "generator_name") or "",
            "generation_success": generation_success,
            "generation_duration_s_raw": duration_s,
            "requires_backend": bool(_get(pr, "requires_backend", False)),
            "adapter_policy": policy,
        }
        for key in (
            "sample_title",
            "sample_complexity",
            "sample_top_category",
            "requirement",
            "session_id",
            "project_id",
            "project_path",
            "error_message",
            "e2e_report_path",
        ):
            value = _get(pr, key)
            if value not in (None, ""):
                meta[key] = value
        for key in ("build_status", "install_status", "launch_status", "generation_status"):
            if result_data.get(key) not in (None, "", "unknown"):
                meta.setdefault("result_data", {})[key] = result_data[key]
        if process_data.get("error_type"):
            meta["process_error_type"] = process_data["error_type"]
        tokens = {
            k: process_data.get(k)
            for k in ("token_input", "token_output", "token_total")
            if process_data.get(k) is not None
        }
        if tokens:
            meta["tokens"] = tokens

        result: Dict[str, Any] = {
            "test_case_id": item_id,
            "grader_results": grader_results,
            "passed": all(gr["passed"] for gr in grader_results),
            "duration_ms": duration_ms,
            "metadata": {_NS: meta},
        }
        error = _error_for(pr)
        if error is not None:
            result["error"] = error
        results.append(result)

    passed_count = sum(1 for r in results if r["passed"])
    scored = [
        gr["score"]
        for r in results
        for gr in r["grader_results"]
        if gr["score"] is not None
    ]
    summary: Dict[str, Any] = {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results),
        "duration_ms": total_duration_ms,
    }
    if scored:
        summary["avg_score"] = sum(scored) / len(scored)

    run_meta: Dict[str, Any] = {
        "source": "daimax-appbench",
        "generator_name": _get(run, "generator_name") or "",
        "run_type": _get(run, "run_type") or "prompt",
        "sample_source": _get(run, "sample_source") or "",
        "timestamp_raw": ts["raw"],
        "timestamp_assumed_timezone": ts["assumed_timezone"],
        "adapter_policy": policy,
        "priority_weight_convention": PRIORITY_WEIGHTS,
        "graders": _SUITE_GRADERS,
    }
    native_summary = _dump(_get(run, "summary"))
    if native_summary:
        run_meta["summary"] = native_summary

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id or _get(run, "sample_source") or "daimax_appbench",
        "run_id": str(_get(run, "run_id") or "daimax_run"),
        "started_at": ts["iso"],
        "results": results,
        "summary": summary,
        "metadata": {_NS: run_meta},
    }
    generator = _get(run, "generator_name")
    if generator:
        result_set["provider"] = {"model": str(generator)}
    result_set["runner"] = {"name": "daimax-appbench"}
    return result_set


def run_from_openeval(result_set: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reverse ``run_to_openeval()``: recover ``PromptResult``-shaped dicts
    (``prompt_id``, ``sample_id``, ``platform``, ``generator_name``,
    ``generation_success``, ``generation_duration``, ``error_message``,
    ``test_results``, ``success_rate``/``quality``/``experience``, ...) from a
    ResultSet this module produced. The dicts validate as real
    ``evalapp...PromptResult`` models when ``evalapp`` is installed.

    Lossy: ``process_data`` (except tokens/error_type), ``result_data``
    (except the four status strings), ``project_path`` and the daimax
    ``EvalSummary`` are only partially carried in metadata, so they come back
    partially or with defaults. Results whose metadata lacks this module's
    ``"daimax"`` namespace are skipped.
    """
    recovered: List[Dict[str, Any]] = []
    for result in result_set.get("results", []):
        meta = (result.get("metadata") or {}).get(_NS)
        if not meta:
            continue
        pr: Dict[str, Any] = {
            "prompt_id": meta.get("prompt_id") or "",
            "sample_id": meta.get("sample_id") or "",
            "item_type": meta.get("item_type", "prompt"),
            "platform": meta.get("platform", ""),
            "generator_name": meta.get("generator_name", ""),
            "generation_success": bool(meta.get("generation_success", False)),
            "generation_duration": float(
                meta.get("generation_duration_s_raw", result.get("duration_ms", 0) / 1000.0)
            ),
            "error_message": meta.get("error_message", ""),
            "requires_backend": bool(meta.get("requires_backend", False)),
            "test_results": [],
            "success_rate": None,
            "quality": None,
            "experience": None,
        }
        if not pr["prompt_id"] and not pr["sample_id"]:
            pr["prompt_id"] = result["test_case_id"]
        for key in (
            "sample_title",
            "sample_complexity",
            "sample_top_category",
            "requirement",
            "session_id",
            "project_id",
            "project_path",
            "e2e_report_path",
        ):
            if key in meta:
                pr[key] = meta[key]
        if "result_data" in meta:
            pr["result_data"] = dict(meta["result_data"])
        process: Dict[str, Any] = {}
        if "process_error_type" in meta:
            process["error_type"] = meta["process_error_type"]
        if "tokens" in meta:
            process.update(meta["tokens"])
        if process:
            pr["process_data"] = process

        for gr in result.get("grader_results", []):
            gmeta = (gr.get("metadata") or {}).get(_NS) or {}
            gid = gr.get("grader_id")
            if gid in _SYNTHETIC_METRIC_FIELDS:
                if "breakdown" in gmeta:
                    pr[_SYNTHETIC_METRIC_FIELDS[gid]] = dict(gmeta["breakdown"])
            elif gid == GRADER_E2E:
                tr: Dict[str, Any] = {
                    "test_case_id": gmeta.get("test_case_id", ""),
                    "passed": bool(gr.get("passed", False)),
                    "status": gmeta.get("status", ""),
                    "details": gr.get("reason", ""),
                    "duration": float(gmeta.get("duration_s_raw", 0.0)),
                }
                for key in ("report_path", "report_started_at", "report_generated_at", "verifications"):
                    if key in gmeta:
                        tr[key] = gmeta[key]
                pr["test_results"].append(tr)
        recovered.append(pr)
    return recovered


# ---------------------------------------------------------------------------
# Suite export (bonus scope, #9: "ResultSet-first is a perfectly reasonable
# v1 scope")
# ---------------------------------------------------------------------------


def to_openeval(
    samples: Sequence[Any],
    *,
    test_cases: Optional[Mapping[str, Iterable[Any]]] = None,
    suite_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert daimax ``EvalSample`` objects (or dicts shaped like
    ``EvalSample.model_dump()``) into an EvalPort ``Suite``.

    One EvalPort ``TestCase`` per sample: ``id`` = ``sample_id`` (the same
    ``item_id`` ``run_to_openeval()`` uses, so a Suite and a ResultSet join
    on it), ``input`` = ``requirement``, ``graders`` = the four shared daimax
    graders. ``test_cases`` optionally maps ``sample_id`` to that sample's
    daimax ``TestCase`` objects (or one ``TestDesignOutput``); they are
    recorded under ``metadata.daimax.test_cases`` with the
    ``priority_weight`` convention applied.
    """
    if not samples:
        raise ValueError("samples is empty -- nothing to convert")

    suite_test_cases: List[Dict[str, Any]] = []
    for sample in samples:
        s = _dump(sample)
        sample_id = s.get("sample_id")
        if not sample_id:
            raise ValueError(f"EvalSample missing 'sample_id': {s!r}")
        requirement = s.get("requirement") or ""
        if not requirement:
            raise ValueError(f"EvalSample {sample_id!r} has an empty 'requirement'")

        meta: Dict[str, Any] = {
            k: s[k]
            for k in (
                "title",
                "platforms",
                "app_type",
                "game_category",
                "complexity",
                "top_category",
                "core_functions",
                "constraints",
                "notes",
                "requires_backend",
                "requires_auth",
                "dataset_version",
                "status",
                "deprecated_reason",
                "pages",
                "mock_resources",
            )
            if k in s
        }
        if test_cases and sample_id in test_cases:
            tcs = test_cases[sample_id]
            tc_dicts: List[Dict[str, Any]] = []
            for tc in list(tcs) if not _get(tcs, "test_cases") else _get(tcs, "test_cases"):
                d = dict(_dump(tc))
                weight = _priority_weight(d.get("priority"))
                if weight is not None:
                    d["priority_weight"] = weight
                tc_dicts.append(d)
            meta["test_cases"] = tc_dicts
            meta["priority_weight_convention"] = PRIORITY_WEIGHTS

        tc_out: Dict[str, Any] = {
            "id": str(sample_id),
            "input": requirement,
            "graders": [g["id"] for g in _SUITE_GRADERS],
            "metadata": {_NS: meta},
        }
        tags = [t for t in (s.get("app_type"), s.get("top_category"), s.get("complexity")) if t]
        if tags:
            tc_out["tags"] = tags
        suite_test_cases.append(tc_out)

    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": suite_id or "daimax_appbench",
        "test_cases": suite_test_cases,
        "graders": [dict(g) for g in _SUITE_GRADERS],
        "metadata": {
            _NS: {
                "source": "daimax-appbench",
                "priority_weight_convention": PRIORITY_WEIGHTS,
            }
        },
    }
    if name is not None:
        suite["name"] = name
    if description is not None:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reverse ``to_openeval()``: recover ``EvalSample``-shaped dicts from a
    Suite this module produced. Test cases whose metadata lacks the
    ``"daimax"`` namespace are skipped. Attached daimax ``TestCase`` dicts, if
    any, are returned under the extra key ``"test_cases"`` (with the
    adapter-only ``priority_weight`` key removed)."""
    samples: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        meta = (tc.get("metadata") or {}).get(_NS)
        if meta is None:
            continue
        sample: Dict[str, Any] = {"sample_id": tc["id"], "requirement": tc.get("input", "")}
        for key, value in meta.items():
            if key in ("test_cases", "priority_weight_convention"):
                continue
            sample[key] = value
        if "test_cases" in meta:
            sample["test_cases"] = [
                {k: v for k, v in d.items() if k != "priority_weight"} for d in meta["test_cases"]
            ]
        samples.append(sample)
    return samples
