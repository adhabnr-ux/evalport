#!/usr/bin/env python3
"""Write real DeepEval and Inspect AI runs, in which a grader raised, as EvalPort documents.

``observe.py`` runs the two frameworks (real DeepEval and Inspect AI code, canned model,
graders that raise on purpose) and reduces each to neutral ``Row`` records. This module
turns those rows into a Suite and a ResultSet, under two ways of writing ``Result.passed``
for a row in which only *some* graders produced a score, and classifies every row three ways:

* what the framework itself did (``native``),
* what EvalPort ``main`` can say about it (Validation Rule 6, ``rule6_class``),
* what ``Result.verdict`` (EvalPort Discussion #49, NOT in the spec) would say (``verdict_for``).

Policies (``--policy``)

``spec-default``
    ``Result.passed`` is the AND of the non-null grader results, as the Aggregation Extension
    in SPEC.md says for its default ``all`` strategy; an all-null row is ``passed: false`` with
    ``metadata.openeval.aggregation_status: "unscored"`` (Rule 6).
``fail-closed``
    ``Result.passed`` is false unless every grader produced a passing score. The Python SDK
    validator accepts this (the TypeScript one was not run), but SPEC.md's prose does not
    describe it: no ``openeval.aggregation`` strategy means "any missing grader fails the row".

The metadata-convention alternative (``--mark-partial``)

    The competing option to a ``Result.verdict`` field, named in Discussion #49, is a
    ``metadata`` convention with no schema change: extend the existing
    ``metadata.openeval.aggregation_status`` (``"unscored"`` = every grader null, Rule 6) with a
    second value, ``"partial"`` = some but not all graders null. ``--mark-partial`` writes it,
    and ``convention_class`` is what a consumer that knows the convention can conclude from a
    Result. Both the field and the marker are in this module so the two can be compared on the
    same rows. ``"partial"`` is NOT in the spec either.

Not an official DeepEval or Inspect AI integration; nothing here has been proposed to, or
reviewed by, either project.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from observe import CASES, GRADERS, QUESTION, Observation, Row, observe_deepeval, observe_inspect

SPEC_VERSION = "1.0.0"
FIXED_STARTED_AT = "2026-10-03T00:00:00Z"
POLICIES = ("spec-default", "fail-closed")
FRAMEWORKS = {"deepeval": observe_deepeval, "inspect_ai": observe_inspect}


def grader_id(name: str) -> str:
    return f"gr_{name}"


# --------------------------------------------------------------------------------------
# Classifications
# --------------------------------------------------------------------------------------


def true_situation(row: Row) -> str:
    """What happened to the row, read from the framework's own observation.

    ``failed``     a grader that did run scored a failure, so the row is a failure whatever
                   the other graders would have said (an AND cannot be rescued);
    ``unverified`` no grader failed, but at least one never produced a score, so the
                   outcome was not established;
    ``passed``     every grader ran and passed.
    """
    scored = [g for g in row.graders if g.score is not None]
    if any(g.score < 1.0 for g in scored):
        return "failed"
    if len(scored) < len(row.graders):
        return "unverified"
    return "passed"


def result_passed(row: Row, policy: str) -> bool:
    scored = [g for g in row.graders if g.score is not None]
    if policy == "fail-closed":
        return len(scored) == len(row.graders) and all(g.score == 1.0 for g in scored)
    if not scored:
        return False
    return all(g.score == 1.0 for g in scored)


def rule6_class(result: Dict[str, Any]) -> str:
    """The most a reader of ``main`` can conclude from a Result, using Rule 6 and nothing else.

    Rule 6: a grader with ``score: null`` is "not verified"; ``passed: false`` on a numeric
    score is "verified failing". A Result whose graders are all null is "not verified". A
    Result with a mix has no name in the spec; the labels below are what its words allow.
    """
    scores = [g["score"] for g in result["grader_results"]]
    if all(s is None for s in scores):
        return "not verified"
    if result["passed"]:
        return "passed"
    return "verified failing"


def verdict_for(row: Row) -> str:
    """``Result.verdict`` as proposed in Discussion #49, for this example's reading of a row."""
    return true_situation(row)


def aggregation_status(row: Row, mark_partial: bool) -> Optional[str]:
    """``metadata.openeval.aggregation_status`` for a row.

    ``"unscored"`` is the spec's own value (every grader null, Rule 6). ``"partial"`` is the
    proposed extension of that key for a row in which some, but not all, graders are null; it
    is written only with ``mark_partial`` and is NOT in the spec.
    """
    nulls = sum(1 for g in row.graders if g.score is None)
    if nulls == len(row.graders):
        return "unscored"
    if nulls and mark_partial:
        return "partial"
    return None


def convention_class(result: Dict[str, Any]) -> str:
    """What a consumer that knows the ``"partial"`` convention can conclude from a Result.

    It reads ``aggregation_status`` and ``passed`` and nothing else, and it assumes the producer
    used the spec's default aggregation (``passed`` = AND of the scored graders), because that
    is the only reading under which ``partial`` + ``passed: true`` means "nothing failed, but
    not everything was measured":

    * ``unscored``                      -> ``unverified``
    * ``partial`` and ``passed: true``  -> ``unverified`` (the scored graders passed; the rest is unknown)
    * ``partial`` and ``passed: false`` -> ``failed`` (a scored grader failed, so the AND cannot be rescued)
    * no marker                         -> ``passed`` / ``failed`` from ``passed``

    The same labels as ``true_situation`` / ``verdict_for``, so the three can be compared.
    """
    status = result.get("metadata", {}).get("openeval", {}).get("aggregation_status")
    if status == "unscored":
        return "unverified"
    if status == "partial":
        return "unverified" if result["passed"] else "failed"
    return "passed" if result["passed"] else "failed"


# --------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------


def build_suite(obs: Observation) -> Dict[str, Any]:
    fw = obs.framework
    return {
        "$schema": "https://evalport.org/schema/suite.json",
        "version": SPEC_VERSION,
        "id": f"suite_errored_graders_{fw}",
        "name": f"Errored graders in {fw}",
        "description": (
            "Four rows graded by two exact_match graders. The second grader (flaky) stands in "
            "for an LLM judge that could not be reached: it raises on q2, q3 and q4, and the "
            "first grader (exact) also raises on q4. The comparison itself is the same in both."
        ),
        "graders": [{"id": grader_id(g), "type": "exact_match", "params": {"ignore_case": True}} for g in GRADERS],
        "test_cases": [
            {
                "id": c["id"],
                "input": QUESTION,
                "expected_output": c["expected"],
                "graders": [grader_id(g) for g in GRADERS],
            }
            for c in CASES
        ],
        "metadata": {fw: {"version": obs.version, "llm": "none (canned outputs, no network)"}},
    }


def _grader_result(framework: str, g, expected: str, actual: str) -> Dict[str, Any]:
    gr: Dict[str, Any] = {"grader_id": grader_id(g.name), "type": "exact_match"}
    if g.score is None:
        gr.update(score=None, passed=False)
        gr["reason"] = g.error or "grader produced no score"
        gr["metadata"] = {framework: {"produced_score": False}}
        return gr
    gr.update(score=g.score, passed=g.score == 1.0)
    if g.score != 1.0:
        gr["reason"] = f"Expected {expected}, got {actual}"
    return gr


def build_result(row: Row, policy: str, proposed_verdict: bool, mark_partial: bool = False) -> Dict[str, Any]:
    case = next(c for c in CASES if c["id"] == row.case_id)
    result: Dict[str, Any] = {
        "test_case_id": row.case_id,
        "actual_output": case["actual"],
        "grader_results": [_grader_result(row.framework, g, case["expected"], case["actual"]) for g in row.graders],
        "passed": result_passed(row, policy),
    }
    meta: Dict[str, Any] = {}
    status = aggregation_status(row, mark_partial)
    if status is not None:
        meta["openeval"] = {"aggregation_status": status}
    fw_meta: Dict[str, Any] = {}
    if row.native_row_success is not None:
        fw_meta["native_row_success"] = row.native_row_success
    if row.native_row_error:
        # Inspect AI records the exception on the sample; the row's own error is a runner_error.
        result["error"] = {"type": "runner_error", "message": row.native_row_error, "retryable": False}
    if fw_meta:
        meta[row.framework] = fw_meta
    if proposed_verdict:
        result["verdict"] = verdict_for(row)
    if meta:
        result["metadata"] = meta
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Producer-side summary. Same choices as examples/langroid: an all-null row is ``skipped``,
    stays in the pass_rate denominator, and null scores are left out of averages."""
    total = len(results)
    scored = lambda r: any(g["score"] is not None for g in r["grader_results"])  # noqa: E731
    passed = sum(1 for r in results if r["passed"])
    skipped = sum(1 for r in results if not scored(r))
    summary: Dict[str, Any] = {
        "total": total,
        "passed": passed,
        "failed": total - passed - skipped,
        "skipped": skipped,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }
    by_grader: Dict[str, Dict[str, Any]] = {}
    all_scores: List[float] = []
    for r in results:
        for g in r["grader_results"]:
            if g["score"] is None:
                continue
            slot = by_grader.setdefault(g["grader_id"], {"passed": 0, "failed": 0, "_scores": []})
            slot["passed" if g["passed"] else "failed"] += 1
            slot["_scores"].append(g["score"])
            all_scores.append(g["score"])
    if all_scores:
        summary["avg_score"] = round(sum(all_scores) / len(all_scores), 4)
        summary["by_grader"] = {
            gid: {"passed": s["passed"], "failed": s["failed"], "avg_score": round(sum(s["_scores"]) / len(s["_scores"]), 4)}
            for gid, s in by_grader.items()
        }
    return summary


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_documents(
    obs: Observation,
    *,
    policy: str = "spec-default",
    proposed_verdict: bool = False,
    deterministic: bool = False,
    mark_partial: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    results = [build_result(r, policy, proposed_verdict, mark_partial) for r in obs.rows]
    fw_meta: Dict[str, Any] = {"version": obs.version, "policy": policy, "run_level": obs.run_level}
    if mark_partial:
        fw_meta["partial_marker"] = "metadata.openeval.aggregation_status: \"partial\" on rows with some null graders (not in the spec)"
    doc: Dict[str, Any] = {
        "$schema": "https://evalport.org/schema/resultset.json",
        "version": SPEC_VERSION,
        "suite_id": f"suite_errored_graders_{obs.framework}",
        "run_id": f"run_errored_graders_{obs.framework}_{policy.replace('-', '_')}",
        "started_at": FIXED_STARTED_AT if deterministic else _now(),
        "runner": {"name": "errored-graders-example", "version": obs.version},
        "results": results,
        "summary": summarize(results),
        "metadata": {obs.framework: fw_meta},
    }
    if not deterministic:
        doc["completed_at"] = _now()
    return build_suite(obs), doc


def comparison_table(obs: Observation) -> str:
    """One line per row: native, both writings, Rule 6's reading, the ``partial`` convention's
    reading of each writing, and the verdict."""
    spec = {r["test_case_id"]: r for r in build_documents(obs, policy="spec-default")[1]["results"]}
    closed = {r["test_case_id"]: r for r in build_documents(obs, policy="fail-closed")[1]["results"]}
    spec_p = {r["test_case_id"]: r for r in build_documents(obs, policy="spec-default", mark_partial=True)[1]["results"]}
    closed_p = {r["test_case_id"]: r for r in build_documents(obs, policy="fail-closed", mark_partial=True)[1]["results"]}
    header = (
        f"{'row':<4} {'graders (exact, flaky)':<26} {'native':<8} {'spec-default':<14} {'fail-closed':<14} "
        f"{'Rule 6 reads (spec-default)':<28} {'partial marker reads (spec / closed)':<38} verdict would say"
    )
    lines = [header, "-" * len(header)]
    for row in obs.rows:
        gs = ", ".join("null" if g.score is None else f"{g.score:g}" for g in row.graders)
        native = {True: "passed", False: "failed", None: "(none)"}[row.native_row_success]
        marker = f"{convention_class(spec_p[row.case_id])} / {convention_class(closed_p[row.case_id])}"
        lines.append(
            f"{row.case_id:<4} {gs:<26} {native:<8} "
            f"{'passed' if spec[row.case_id]['passed'] else 'failed':<14} "
            f"{'passed' if closed[row.case_id]['passed'] else 'failed':<14} "
            f"{rule6_class(spec[row.case_id]):<28} {marker:<38} {verdict_for(row)}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run DeepEval / Inspect AI with a raising grader and write EvalPort documents.")
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("."))
    p.add_argument("--framework", choices=[*FRAMEWORKS, "all"], default="all")
    p.add_argument("--policy", choices=POLICIES, default="spec-default")
    p.add_argument("--proposed-verdict", action="store_true",
                   help="add Result.verdict as proposed in EvalPort Discussion #49 (NOT in the spec)")
    p.add_argument("--mark-partial", action="store_true",
                   help='write metadata.openeval.aggregation_status: "partial" on rows with some null graders '
                        "(the metadata-convention alternative to Result.verdict; NOT in the spec)")
    p.add_argument("--deterministic", action="store_true",
                   help="fixed started_at and no completed_at (for sample_output/ and diffs)")
    args = p.parse_args(argv)

    from openeval.validate import validate_result_set, validate_suite

    names = list(FRAMEWORKS) if args.framework == "all" else [args.framework]
    for name in names:
        obs = FRAMEWORKS[name]()
        suite, rs = build_documents(obs, policy=args.policy, proposed_verdict=args.proposed_verdict,
                                    deterministic=args.deterministic, mark_partial=args.mark_partial)
        for label, doc, fn in (("suite", suite, validate_suite), ("results", rs, validate_result_set)):
            v = fn(doc)
            if not v.valid:
                print(f"internal error: generated {name} {label} is invalid: {v.errors}", file=sys.stderr)
                return 1
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
        (out / "results.json").write_text(json.dumps(rs, indent=2) + "\n")
        s = rs["summary"]
        print(f"== {name} {obs.version}   policy={args.policy}   mark_partial={args.mark_partial}")
        print(comparison_table(obs))
        print(f"summary: total={s['total']} passed={s['passed']} failed={s['failed']} "
              f"skipped={s['skipped']} pass_rate={s['pass_rate']}")
        print(f"native run-level: {obs.run_level}")
        print(f"wrote {out / 'suite.json'} and {out / 'results.json'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
