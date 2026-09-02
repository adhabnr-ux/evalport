"""Journeyman <-> EvalPort adapter.

Standalone converter between codechu/journeyman's run records
(``cells/<id>.json`` + ``report.json``) and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than an in-repo journeyman
change: see https://github.com/codechu/journeyman/issues/1, opened after
reading `record.py`, `judge.py`, `rubrics.py`/`scene.py`, `qualify.py` and
`journeyman/schema/report.schema.json` on the real, installed
`journeyman-bench==0.4.0` package -- the same "standalone package, zero
footprint on the target repo" shape used by the DeepEval, AutoGen, CrewAI
and Giskard adapters in this ecosystem.

The mapping was corrected twice across that issue thread by journeyman's
maintainer before landing here, and every correction changed this code, not
just a comment:

1. Journeyman scores two *kinds* of axis, discriminated by which key an
   axis's data arrives under on a cell record -- **not** by any field
   value (an earlier attempt to key off ``RubricItem.positive is None``
   was wrong: `positive` is a required `str`, never `None`; verified
   against `journeyman/scene.py`):

   - ``cell["verdicts"][axis]`` -- a **judged** axis: a judge answered a
     rubric question and its verdict was matched against a positive label.
   - ``cell["event_axes"][axis]`` -- a **counted** axis: a value in
     ``[0, 1]`` computed by replaying events, no judge involved.

   Mapping a counted axis as an LLM-judge result would dress up a
   deterministic count as a model's opinion -- exactly the flattening
   EvalPort exists to prevent. This adapter discriminates by dict key,
   never by inspecting a value, and never emits ``kind: "counted"`` data
   under a grader that claims to be judge-backed.

2. ``report.json``'s ``seal``, ``judge`` and ``self_judged`` are the
   conditions a score was true under -- a self-judged run is stamped
   ``NOT COMPARABLE`` in journeyman's own report, and journeyman's judge
   registry exists because most examined judges failed the qualification
   exam. ``cells_to_result_set()`` therefore *requires* a `report` dict
   carrying all three (``ValueError`` if any is missing) and copies them
   into ``ResultSet.metadata["journeyman"]`` verbatim, plus a derived
   ``comparability`` flag -- there is no code path that produces a
   `ResultSet` whose numbers have shed the conditions they were true
   under.

Two independent conversions are provided, mirroring journeyman's own
two-layer model (the task a cell was given, and the graded outcome of
running it):

- `cells_to_testcases()` -- cell records -> an EvalPort suite (dict) of
  `TestCase` objects, one per cell, each referencing one `Grader` per
  axis the cell feeds.
- `cells_to_result_set()` -- cell records + `report.json` -> an EvalPort
  `ResultSet` (dict), one `Result` per cell, each carrying one
  `GraderResult` per axis verdict/event-axis value on that cell.

Mapping, verified against the real, installed `journeyman-bench==0.4.0`
source (`record.py`, `report.py`, `scene.py`, `judge.py`,
`journeyman/schema/report.schema.json`) and against a real run this
adapter's own test suite produces by calling journeyman's actual
`run_grid()` / `judge_cell()` / `render()` -- not a hand-typed guess at
the shape:

| Journeyman                                                    | EvalPort                          |
|-----------------------------------------------------------------|------------------------------------|
| one cell (`cells/<id>.json`: `cell_id, scene, seed, messages, final_text, budget, events, event_axes, verdicts, calls, tokens_in, tokens_out, seconds, invalid, invalid_reason`) | `TestCase` (id, input) + its `Result` |
| `cell["verdicts"][axis]` (`verdict, positive, na_means, raw`) -- judged | `GraderResult(grader_id=axis, type="custom", params.kind="judged")` |
| `cell["event_axes"][axis]` (a `[0,1]` ratio) -- counted, no judge | `GraderResult(grader_id=axis, type="custom", params.kind="counted", deterministic=True)` |
| `report.json`'s `{schema_version, seal, judge, self_judged, nonstandard, axes, cost, invalid_cells}` | `ResultSet(suite_id, run_id, results, runner, summary, metadata)` -- `seal`/`judge`/`self_judged` are MANDATORY under `metadata["journeyman"]`, never optional |

## Why judged axes are NOT mapped to EvalPort's `llm_judge` grader type

The original sketch in the linked issue proposed `type="llm_judge"`. Tested
for real against `openeval.validate.validate_grader()` (the installed
`evalport-sdk==1.0.0` from PyPI, not just read), that fails: EvalPort's
`llm_judge` grader requires `params.model` (a judge model id) and
`params.prompt` (a template containing `{output}`, `{input}`, or
`{expected}`). Journeyman's judge protocol has neither shape -- one judge
identity is stamped once per *run* (`report.json`'s `judge` field), not
per grader, and the real prompt template (`judge.py`'s `JUDGE_PREAMBLE`)
is built from `{labels}`, `{question}`, `{evidence}`, `{record}` --
substitution points EvalPort's `llm_judge` schema doesn't recognize.
Rather than fabricate a `{output}`-shaped prompt journeyman never actually
sends, judged axes use `type="custom"` with `params.handler =
"journeyman:rubric_judge"` and the real rubric fields
(`question`/`verdicts`/`positive`/`na_means`, when a caller supplies them
via `rubric_index` -- see below) preserved in `params`, the same honest
"custom, not force-fit" choice the DeepEval adapter makes for its own
framework-specific metrics.

## `rubric_index`: optional, and never load-bearing

`docs/versioning.md` states plainly that rubric question text
(`rubrics.py`) is "deliberately unstable" and must never be keyed off.
This adapter never does: the discriminator between judged and counted is
always the dict key (`verdicts` vs `event_axes`) a cell's data arrived
under, never rubric text or any field value. `rubric_index` (an optional
`{axis: RubricItem-like object or dict}` map, buildable from a real
`Scene().rubric()`) is used *only* to enrich a judged grader's `params`
with the real question/verdicts/positive/na_means for readability --
omit it and the grader still validates and still carries the right
`kind`, just without the question text.

## `schema_version`: pin and stop, never guess

Per `docs/versioning.md` ("Pin on `schema_version`, not on the package
version... on a value your code does not know, stop rather than guess"),
`cells_to_result_set()` raises `ValueError` when `report["schema_version"]`
is not `1` (the only shape this adapter has been verified against) unless
called with `strict_schema=False`.

## `judge`: opaque identity, never parsed

Since journeyman 0.3.0, `report.json`'s `judge` field is written through
`report.py`'s `public_label()`, which folds a private host address or an
absolute filesystem path before the label is ever written -- provenance
says *who* judged, not where the maintainer keeps their files. This
adapter copies `report["judge"]` through verbatim and never inspects,
splits, or parses it as a machine address.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["cells_to_testcases", "cells_to_result_set", "SUPPORTED_SCHEMA_VERSION", "__version__"]
__version__ = "0.1.0"

# The only journeyman report.json schema_version this adapter has been
# verified against (journeyman/report.py's SCHEMA_VERSION as of
# journeyman-bench 0.1.0-0.4.0). Per docs/versioning.md: stop on an
# unknown value rather than guess at a shape that may have changed.
SUPPORTED_SCHEMA_VERSION = 1

_REPORT_REQUIRED_KEYS = ("seal", "judge", "self_judged")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object (so a real
    `RubricItem` dataclass instance and a plain dict both work as
    `rubric_index` entries)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _closing_text(cell: Dict[str, Any]) -> str:
    """Mirrors journeyman/report.py's `_closing_text()`: the agent's own
    closing words live in a `report`/`conclude` tool call's arguments when
    `final_text` is empty (the scene closed itself on that call, and the
    agent wrote nothing after it)."""
    text = ""
    for m in cell.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            if fn.get("name") in ("report", "conclude"):
                import json as _json
                try:
                    args = _json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    continue
                text = args.get("text") or args.get("decision") or text
    return text


def _actual_output(cell: Dict[str, Any]) -> "tuple[Optional[str], str]":
    """(text, source) -- `final_text` when the agent's own trailing turn
    carried it, else the closing-tool-call fallback `report.py` itself
    uses. `source` is surfaced in Result.metadata so a caller can tell a
    real trailing message from a reconstructed one apart."""
    final_text = cell.get("final_text")
    if final_text:
        return final_text, "final_text"
    fallback = _closing_text(cell)
    return (fallback or None), ("closing_tool_call" if fallback else "none")


def _rubric_params(axis: str, rubric_index: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"handler": "journeyman:rubric_judge", "kind": "judged"}
    if rubric_index and axis in rubric_index:
        item = rubric_index[axis]
        for field in ("question", "verdicts", "positive", "na_means"):
            value = _get(item, field)
            if value is not None:
                params[field] = list(value) if field == "verdicts" else value
    return params


def _axes_in(cells: Sequence[Dict[str, Any]]) -> "tuple[List[str], List[str]]":
    """(judged_axes, counted_axes) -- every axis name seen across `cells`,
    discriminated purely by which dict key it arrived under (never by a
    field value), in first-seen order."""
    judged: List[str] = []
    counted: List[str] = []
    for c in cells:
        for axis in (c.get("verdicts") or {}):
            if axis not in judged:
                judged.append(axis)
        for axis in (c.get("event_axes") or {}):
            if axis not in counted:
                counted.append(axis)
    return judged, counted


def cells_to_testcases(
    cells: Sequence[Dict[str, Any]],
    *,
    suite_id: str,
    name: Optional[str] = None,
    rubric_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export journeyman cell records to an EvalPort-shaped suite (dict).

    `cells` is any iterable of real `cells/<id>.json` records (dicts with
    the shape `record.py` documents -- `cell_id`, `messages`, `verdicts`,
    `event_axes`, ...). `rubric_index`, if given, is an optional
    `{axis: RubricItem-or-dict}` map (e.g. built from a real
    `SceneClass().rubric()`) used only to enrich judged-axis grader
    `params` with the real question text -- never to decide judged vs.
    counted, which is always read from the cell's own dict keys.

    A cell's `input` is the literal first user-role message content
    journeyman sent the agent (task prompt + the appended
    "Tool budget: N calls." line `driver.py` adds before the first turn)
    -- reported verbatim, not reconstructed or trimmed.

    Invalid cells (`cell["invalid"]` true) are still described as
    `TestCase`s -- the task was still posed -- but carry no graders,
    since journeyman's own `axis_scores()` excludes them from scoring
    entirely (`report.py`: `if c["invalid"]: continue`) and there is
    nothing this adapter could honestly score them on.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance.
    """
    cell_list = list(cells)
    judged_axes, counted_axes = _axes_in(cell_list)

    graders: List[Dict[str, Any]] = []
    for axis in judged_axes:
        graders.append({
            "id": axis, "type": "custom",
            "params": _rubric_params(axis, rubric_index),
            "description": (
                "Journeyman JUDGED axis: a judge answered a rubric question "
                "and the verdict was matched against a positive label. Not "
                "EvalPort's llm_judge type -- see this package's module "
                "docstring for why."
            ),
        })
    for axis in counted_axes:
        graders.append({
            "id": axis, "type": "custom",
            "params": {"handler": "journeyman:event_axis", "kind": "counted", "deterministic": True},
            "description": (
                "Journeyman COUNTED axis: a value in [0,1] computed by "
                "replaying events (Scene.event_axes) -- no judge involved. "
                "A counted fact, never scored as a model's opinion."
            ),
        })

    test_cases: List[Dict[str, Any]] = []
    for cell in cell_list:
        cell_id = cell["cell_id"]
        messages = cell.get("messages") or []
        first_user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
        if not first_user:
            raise ValueError(
                f"cell {cell_id!r} has no user-role message to use as `input` -- "
                "EvalPort's TestCase.input is required and non-empty."
            )

        cell_graders = list((cell.get("verdicts") or {}).keys()) + list((cell.get("event_axes") or {}).keys())

        tc: Dict[str, Any] = {
            "id": cell_id,
            "input": first_user,
            "graders": cell_graders,
            "tags": [cell.get("scene", "")],
            "metadata": {
                "journeyman": {
                    "scene": cell.get("scene"),
                    "seed": cell.get("seed"),
                    "budget": cell.get("budget"),
                    "invalid": cell.get("invalid", False),
                    "invalid_reason": cell.get("invalid_reason"),
                }
            },
        }
        test_cases.append(tc)

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": name or f"journeyman cells ({suite_id})",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "journeyman"}},
    }


def cells_to_result_set(
    cells: Sequence[Dict[str, Any]],
    report: Dict[str, Any],
    *,
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    strict_schema: bool = True,
) -> Dict[str, Any]:
    """Export journeyman cell records + `report.json` to an EvalPort `ResultSet`.

    `report` MUST be journeyman's real `report.json` (or an equal dict) --
    `seal`, `judge` and `self_judged` are required and copied verbatim
    into `ResultSet.metadata["journeyman"]`. This is not a convenience
    default: a `Result` cannot shed the conditions under which its
    numbers were true (journeyman's own report stamps a self-judged run
    NOT COMPARABLE in capitals; this adapter mirrors that as a computed
    `comparability` field rather than dropping it silently), so a
    `report` missing any of the three raises `ValueError` instead of
    producing a `ResultSet` that quietly omits them.

    `report["schema_version"]` is checked against `SUPPORTED_SCHEMA_VERSION`
    (currently `1`); an unknown value raises `ValueError` unless
    `strict_schema=False` is passed explicitly -- per `docs/versioning.md`,
    stop rather than guess at a shape that may have changed.

    `report["judge"]` is copied through as an opaque identity string and
    never parsed -- see the module docstring.

    Each cell's `verdicts` entries and `event_axes` entries become
    `GraderResult`s, discriminated by dict key exactly as
    `cells_to_testcases()` discriminates its graders (never by a field
    value). A judged axis whose verdict is `"na"` with `na_means ==
    "not-applicable"` is excluded from journeyman's own axis score
    (`report.py`'s `axis_scores()`); this adapter keeps that
    `GraderResult` (so no verdict is silently dropped) but marks it
    `metadata.excluded_from_axis_score = True` and `passed = False`
    rather than inventing evidence journeyman itself says doesn't exist.

    An invalid cell (`cell["invalid"]` true) produces a `Result` with no
    `grader_results` and `error.type = "invalid_cell"`, matching
    journeyman's own exclusion of invalid cells from every axis score --
    never a silent pass.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass
    it to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    missing = [k for k in _REPORT_REQUIRED_KEYS if k not in report]
    if missing:
        raise ValueError(
            f"report is missing required key(s) {missing!r} -- seal, judge and "
            "self_judged are mandatory: a Result cannot travel without the "
            "conditions it was true under. Pass journeyman's real report.json."
        )

    schema_version = report.get("schema_version")
    if strict_schema and schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"report[\"schema_version\"] = {schema_version!r}, this adapter is "
            f"verified against {SUPPORTED_SCHEMA_VERSION!r} only. Stopping rather "
            "than guessing at a shape that may have changed (docs/versioning.md). "
            "Pass strict_schema=False to proceed anyway."
        )

    if started_at is None:
        from datetime import datetime, timezone
        started_at = datetime.now(timezone.utc).isoformat()

    self_judged = bool(report["self_judged"])
    nonstandard = report.get("nonstandard")
    comparability = "NOT_COMPARABLE" if (self_judged or nonstandard) else "comparable"

    results: List[Dict[str, Any]] = []
    for cell in cells:
        cell_id = cell["cell_id"]

        if cell.get("invalid"):
            results.append({
                "test_case_id": cell_id,
                "passed": False,
                "grader_results": [],
                "error": {
                    "type": "invalid_cell",
                    "message": cell.get("invalid_reason") or "cell marked invalid by journeyman",
                },
                "metadata": {"journeyman": {"invalid": True}},
            })
            continue

        actual_output, output_source = _actual_output(cell)
        grader_results: List[Dict[str, Any]] = []

        for axis, v in (cell.get("verdicts") or {}).items():
            verdict = v.get("verdict")
            positive = v.get("positive")
            na_means = v.get("na_means", "failure")
            excluded = (verdict == "na" and na_means == "not-applicable")
            passed = (not excluded) and (verdict == positive)
            score = None if excluded else float(verdict == positive)
            raw = v.get("raw") or ""
            gr: Dict[str, Any] = {
                "grader_id": axis, "type": "custom",
                "score": score, "passed": passed,
                "metadata": {
                    "kind": "judged", "raw_verdict": verdict,
                    "positive": positive, "na_means": na_means,
                    "excluded_from_axis_score": excluded,
                },
            }
            if raw:
                gr["reason"] = raw[-300:]
            grader_results.append(gr)

        for axis, val in (cell.get("event_axes") or {}).items():
            score = float(val)
            grader_results.append({
                "grader_id": axis, "type": "custom",
                "score": score, "passed": score >= 1.0,
                "metadata": {"kind": "counted", "deterministic": True},
            })

        overall_passed = bool(grader_results) and all(g["passed"] for g in grader_results)

        result: Dict[str, Any] = {
            "test_case_id": cell_id,
            "passed": overall_passed,
            "grader_results": grader_results,
            "duration_ms": round(cell.get("seconds", 0.0) * 1000),
            "metadata": {
                "journeyman": {
                    "events": cell.get("events"),
                    "budget": cell.get("budget"),
                    "calls": cell.get("calls"),
                    "tokens_in": cell.get("tokens_in"),
                    "tokens_out": cell.get("tokens_out"),
                    "output_source": output_source,
                }
            },
        }
        if actual_output is not None:
            result["actual_output"] = actual_output
        results.append(result)

    total = len(results)
    passed_n = sum(1 for r in results if r["passed"])
    scores = [g["score"] for r in results for g in r.get("grader_results", []) if g.get("score") is not None]
    summary = {
        "total": total, "passed": passed_n, "failed": total - passed_n,
        "skipped": sum(1 for r in results if "error" in r),
        "pass_rate": (passed_n / total) if total else 0,
        "avg_score": (sum(scores) / len(scores)) if scores else 0,
    }

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at or started_at,
        "runner": {"name": "journeyman", "version": report["seal"].get("bench", "")},
        "results": results,
        "summary": summary,
        "metadata": {
            "openeval": {"source": "journeyman"},
            "journeyman": {
                "seal": report["seal"],
                "judge": report["judge"],
                "self_judged": self_judged,
                "nonstandard": nonstandard,
                "comparability": comparability,
                "schema_version": schema_version,
                "invalid_cells": report.get("invalid_cells"),
            },
        },
    }
