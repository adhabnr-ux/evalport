"""safelabs-eval <-> EvalPort adapter.

Standalone converter between AgentSafeLabs' safelabs-eval
(https://github.com/AgentSafeLabs/safelabs-eval) — an OWASP Agentic Security
Initiative (ASI) red-teaming/eval framework for AI agents — and the EvalPort
interchange format (https://github.com/adhabnr-ux/evalport).

Follows the same "outside" playbook as every other adapter in this repo
(see ../crewai-openeval-adapter, ../langsmith-openeval-adapter): it works
against safelabs-eval's public `PromptEntry` / `EvalRecord` / `EvalResult` /
`ScoringResult` shapes (objects or dicts, matching the pydantic model field
names in `safelabs/prompts/schemas.py`, `safelabs/runner.py`, and
`safelabs/scoring/models.py` exactly) from the outside, so you get EvalPort
import/export today without anything merged into safelabs-eval itself.

Unlike the CrewAI/LangSmith adapters, safelabs-eval has a genuine split
between its *test definitions* (the 30-prompt OWASP ASI library, as
`PromptEntry` objects) and its *run results* (`EvalResult`, produced by
`safelabs.runner.run_eval()`), which maps cleanly onto EvalPort's own
EvalSuite / ResultSet split rather than collapsing both into one document:

    PromptEntry  -> EvalPort TestCase           (prompts_to_suite / to_openeval)
    EvalRecord   -> EvalPort Result              (record_to_result)
    EvalResult   -> EvalPort ResultSet           (eval_result_to_resultset / to_openeval)
    detector     -> EvalPort `custom` Grader     (params.handler = eval_type)

Mapping confirmed by @iamwaqarjaved (safelabs-eval maintainer) on
https://github.com/AgentSafeLabs/safelabs-eval/issues/1, including the
VerdictLevel -> score/passed table below.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "prompt_to_testcase",
    "prompts_to_suite",
    "record_to_result",
    "eval_result_to_resultset",
    "CATEGORY_EVAL_TYPE",
    "VERDICT_SCORE",
    "__version__",
]
__version__ = "0.1.0"

# Mirrors safelabs/runner.py's own CATEGORY_EVAL_TYPE constant: maps each OWASP
# ASI category to the detector eval_type that best covers it. Duplicated here
# (rather than imported from `safelabs.runner`) because this adapter is
# duck-typed against safelabs-eval's public shapes and does not depend on the
# safelabs-eval package itself — see module docstring. Kept in sync by hand;
# if safelabs-eval's own mapping changes, an entry's `category` metadata will
# still round-trip correctly since prompt_to_testcase() always carries the
# original category through in metadata regardless of which eval_type it maps to.
CATEGORY_EVAL_TYPE: Dict[str, str] = {
    "ASI01": "prompt_injection",
    "ASI02": "prompt_injection",   # insecure output — best available
    "ASI03": "scope_violation",    # excessive agency / unauthorised actions
    "ASI04": "prompt_injection",   # resource management
    "ASI05": "prompt_injection",   # tool use safety
    "ASI06": "data_leakage",
    "ASI07": "prompt_injection",   # trust boundaries
    "ASI08": "jailbreak",
    "ASI09": "scope_violation",
    "ASI10": "hallucination",
}

# VerdictLevel -> (score, passed) mapping, confirmed by @iamwaqarjaved on
# AgentSafeLabs/safelabs-eval#1: VULNERABLE and FAIL collapse to the same
# numeric score (a partial score for FAIL would misrepresent safelabs-eval's
# detectors' binary semantics) and are distinguished only via
# `metadata.safelabs.verdict`. UNCERTAIN maps to `score: None` (EvalPort spec
# Rule 6: "not verified" is distinct from "verified failing"), mirroring how
# the giskard-openeval-adapter treats Giskard's ERROR/SKIP as no verdict
# reached rather than a falsely-confident 0.0.
VERDICT_SCORE: Dict[str, Optional[float]] = {
    "pass": 1.0,
    "fail": 0.0,
    "vulnerable": 0.0,
    "uncertain": None,
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    safelabs-eval's PromptEntry/EvalRecord/EvalResult/ScoringResult are
    pydantic BaseModel instances (attribute access) but JSON-loaded eval
    output (e.g. from `EvalResult.model_dump()` or a saved report) shows up
    as plain dicts, so every accessor in this module goes through here
    rather than assuming one shape.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _enum_value(x: Any) -> Any:
    """Return `.value` for an Enum member (PromptCategory, VerdictLevel), else x unchanged."""
    return getattr(x, "value", x)


def _eval_type_for(entry: Any) -> str:
    """Resolve the detector eval_type for a PromptEntry via its category.

    Falls back to "prompt_injection" for an unrecognized category, matching
    `safelabs.runner.CATEGORY_EVAL_TYPE.get(..., "prompt_injection")`.
    """
    category = _enum_value(_get(entry, "category"))
    return CATEGORY_EVAL_TYPE.get(str(category), "prompt_injection")


def prompt_to_testcase(entry: Any, *, eval_type: Optional[str] = None) -> Dict[str, Any]:
    """Convert a single safelabs-eval `PromptEntry` into an EvalPort TestCase dict.

    `entry` exposes `id`, `category`, `severity`, `prompt`, `expected_behavior`,
    `tags` (object or dict, matching `safelabs/prompts/schemas.py::PromptEntry`).

    Field mapping (confirmed on AgentSafeLabs/safelabs-eval#1):
        id                 -> id
        prompt             -> input
        expected_behavior  -> expected_output
        tags               -> metadata.safelabs.tags
        category, severity -> carried in metadata.safelabs (EvalPort has no
                               first-class OWASP-ASI field)
        detector eval_type -> graders: ["gr_<eval_type>"]

    Pass an explicit `eval_type` to override the category-derived detector
    (e.g. when building a suite for a `Scorer` configured with custom
    detectors); otherwise it's resolved from `entry.category` via
    `CATEGORY_EVAL_TYPE`.
    """
    resolved_eval_type = eval_type or _eval_type_for(entry)
    category = _enum_value(_get(entry, "category"))
    severity = _get(entry, "severity")
    tags = _get(entry, "tags") or []
    expected_behavior = _get(entry, "expected_behavior")

    tc: Dict[str, Any] = {
        "id": str(_get(entry, "id")),
        "input": _get(entry, "prompt"),
        "graders": [f"gr_{resolved_eval_type}"],
        "metadata": {
            "safelabs": {
                "category": str(category) if category is not None else None,
                "severity": severity,
                "tags": list(tags),
            }
        },
    }
    if expected_behavior is not None:
        tc["expected_output"] = expected_behavior
    return tc


def _grader_for_eval_type(eval_type: str) -> Dict[str, Any]:
    """Build the EvalPort `custom` Grader definition for a detector eval_type.

    Every safelabs-eval detector (PromptInjectionDetector, JailbreakDetector,
    DataLeakageDetector, HallucinationDetector, ScopeViolationDetector) maps
    to a `custom` EvalPort grader with `params.handler` set to the detector's
    `eval_type` string, per the mapping confirmed on
    AgentSafeLabs/safelabs-eval#1.
    """
    return {
        "id": f"gr_{eval_type}",
        "type": "custom",
        "description": f"safelabs-eval {eval_type} detector",
        "params": {"handler": f"safelabs:{eval_type}"},
    }


def prompts_to_suite(
    entries: Union[Any, Iterable[Any]],
    *,
    suite_id: str = "safelabs_owasp_asi",
    name: str = "safelabs-eval OWASP ASI Prompt Library",
) -> Dict[str, Any]:
    """Export safelabs-eval `PromptEntry` objects to an EvalPort EvalSuite dict.

    `entries` may be a `PromptLibrary` (anything exposing an `.entries`
    list/attribute or `["entries"]` key) or a plain iterable of
    `PromptEntry`-shaped objects/dicts (e.g. `PromptLibrary.by_category(...)`,
    or the full `get_library().entries`).

    One `custom` grader is emitted per distinct detector eval_type actually
    used by the entries (not all five unconditionally), so a suite built from
    e.g. `library.by_category("ASI06")` only declares `gr_data_leakage`.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass it
    to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    # Accept a PromptLibrary (has `.entries`) as well as a bare iterable.
    maybe_entries = _get(entries, "entries", None)
    entry_list: List[Any] = list(maybe_entries) if maybe_entries is not None else list(entries)

    eval_types = [_eval_type_for(e) for e in entry_list]
    test_cases = [
        prompt_to_testcase(e, eval_type=et) for e, et in zip(entry_list, eval_types)
    ]

    seen: List[str] = []
    for et in eval_types:
        if et not in seen:
            seen.append(et)
    graders = [_grader_for_eval_type(et) for et in seen]

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name,
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "safelabs-eval"}, "openeval.profile": "safety"},
    }


def record_to_result(record: Any) -> Dict[str, Any]:
    """Convert a single safelabs-eval `EvalRecord` into an EvalPort Result dict.

    `record` exposes `prompt_id`, `category`, `severity`, `prompt`, `response`,
    `latency_ms`, `scoring_result` (a `ScoringResult`), `error` (object or
    dict, matching `safelabs/runner.py::EvalRecord`).

    `record.response` (the raw agent output, archived before scoring per
    `DATA_INTEGRITY_RULES.md`) is carried through unchanged as
    `actual_output` — this adapter never becomes the sole retention point for
    it, per the agreement on AgentSafeLabs/safelabs-eval#1; it's just not
    dropped here either.

    The full `ScoringResult` (`reasoning`, `indicators`, `remediation_hint`,
    `confidence`) is preserved under `grader_results[0].metadata.safelabs` —
    nothing silently dropped, the same convention every adapter in this repo
    uses (see giskard-openeval-adapter's grader/check mapping).
    """
    scoring_result = _get(record, "scoring_result")
    verdict = _enum_value(_get(scoring_result, "verdict"))
    verdict_str = str(verdict) if verdict is not None else "uncertain"
    eval_type = _get(scoring_result, "eval_type")
    score = VERDICT_SCORE.get(verdict_str)
    passed = verdict_str == "pass"

    error_text = _get(record, "error")
    latency_ms = _get(record, "latency_ms")

    grader_result: Dict[str, Any] = {
        "grader_id": f"gr_{eval_type}",
        "type": "custom",
        "score": score,
        "passed": passed,
        "reason": _get(scoring_result, "reasoning"),
        "metadata": {
            "safelabs": {
                "verdict": verdict_str,
                "confidence": _get(scoring_result, "confidence"),
                "indicators": list(_get(scoring_result, "indicators") or []),
                "remediation_hint": _get(scoring_result, "remediation_hint"),
                "eval_type": eval_type,
            }
        },
    }

    result: Dict[str, Any] = {
        "test_case_id": str(_get(record, "prompt_id")),
        "actual_output": _get(record, "response"),
        "grader_results": [grader_result],
        # Result-level `passed` is the AND of non-skipped grader results (spec
        # default aggregation, Rule 6 / "all" strategy) — with exactly one
        # grader per record, that's just this grader's own `passed`.
        "passed": passed,
        "metadata": {
            "safelabs": {
                "category": _get(record, "category"),
                "severity": _get(record, "severity"),
            }
        },
    }
    if latency_ms is not None:
        result["duration_ms"] = round(latency_ms)
    if error_text:
        result["error"] = {"type": "runner_error", "message": str(error_text)}
    return result


def eval_result_to_resultset(
    result: Any,
    *,
    suite_id: str = "safelabs_owasp_asi",
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a safelabs-eval `EvalResult` (from `run_eval()`) to an EvalPort ResultSet dict.

    `result` exposes `records` (list of `EvalRecord`) and `categories_run`
    (object or dict, matching `safelabs/runner.py::EvalResult`).

    EvalPort's `ResultSet` requires `suite_id`, `run_id`, and `started_at`
    (ISO 8601) — none of which `EvalResult` itself tracks, since
    `run_eval()` doesn't stamp a run/suite identity or start time. Pass them
    explicitly if you have them (e.g. from your own harness around
    `run_eval()`); otherwise `run_id` defaults to a timestamp-based value and
    `started_at`/`completed_at` default to "now" so the output is always a
    valid, if approximate, `ResultSet`.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass it
    to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved_run_id = run_id or f"safelabs_run_{now.replace(':', '').replace('-', '')}"
    resolved_started_at = started_at or now
    resolved_completed_at = completed_at or now

    records = _get(result, "records") or []
    categories_run = list(_get(result, "categories_run") or [])

    results = [record_to_result(r) for r in records]

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": resolved_run_id,
        "started_at": resolved_started_at,
        "completed_at": resolved_completed_at,
        "runner": {"name": "safelabs-eval"},
        "results": results,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
        },
        "metadata": {
            "openeval": {"source": "safelabs-eval"},
            "safelabs": {"categories_run": categories_run},
        },
    }


def to_openeval(obj: Any, **kwargs: Any) -> Dict[str, Any]:
    """Convenience dispatcher: export either a prompt collection or a run result.

    - If `obj` looks like an `EvalResult` (has `.records`/`["records"]`),
      delegates to `eval_result_to_resultset(obj, **kwargs)` and returns an
      EvalPort ResultSet.
    - Otherwise `obj` is treated as a `PromptLibrary` or an iterable of
      `PromptEntry`, and this delegates to `prompts_to_suite(obj, **kwargs)`,
      returning an EvalPort EvalSuite.

    Prefer calling `prompts_to_suite()` / `eval_result_to_resultset()`
    directly when you know which one you have — this dispatcher exists only
    to match the `to_openeval()`/`from_openeval()` naming convention used by
    every other adapter in this repo.
    """
    if _get(obj, "records", None) is not None:
        return eval_result_to_resultset(obj, **kwargs)
    return prompts_to_suite(obj, **kwargs)


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of safelabs-eval-shaped PromptEntry dicts.

    Returns plain dicts (id, category, severity, prompt, expected_behavior,
    tags) rather than real `PromptEntry` instances, since constructing one
    requires a valid `PromptCategory` value this module has no way to
    guarantee for a suite that didn't originate from safelabs-eval. Pass the
    fields straight into `PromptEntry(**entry)` yourself (validating/mapping
    `category` as needed), or use them directly with a `Scorer`.

    Round-tripping a suite this adapter itself produced (via
    `prompts_to_suite`) recovers the original `category`/`severity`/`tags`
    from `metadata.safelabs`. For a suite from another EvalPort producer
    (the point of `from_openeval` per AgentSafeLabs/safelabs-eval#1 — "the 30
    OWASP ASI prompts get pulled into another EvalPort-based pipeline" runs
    in reverse too: someone else's test cases get run through safelabs-eval's
    detectors), `category` defaults to `"ASI01"` and `severity` to `"medium"`
    when absent, since every `PromptEntry` requires a category/severity and
    a caller running third-party test cases through safelabs-eval's
    detectors is expected to override these defaults deliberately rather
    than have the adapter silently guess a specific one.
    """
    entries: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        safelabs_meta = (tc.get("metadata") or {}).get("safelabs") or {}
        entries.append(
            {
                "id": tc.get("id"),
                "category": safelabs_meta.get("category") or "ASI01",
                "severity": safelabs_meta.get("severity") or "medium",
                "prompt": tc.get("input"),
                "expected_behavior": tc.get("expected_output") or "",
                "tags": list(safelabs_meta.get("tags") or []),
            }
        )
    return entries
