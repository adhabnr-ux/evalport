"""Export an EvalPort Suite + ResultSet as MLCommons Croissant Tasks JSON-LD.

Croissant Tasks (https://github.com/mlcommons/croissant/tree/main/tasks) models
a benchmark as a ``cr:TaskProblem``, a submission as a ``cr:TaskSolution`` that
``schema:isBasedOn`` the problem, and its scoring as a ``cr:EvaluationTask``
whose ``cr:evaluationResults`` are ``(metric, value)`` pairs.

This script emits one JSON-LD document holding all three nodes. It is lossy on
purpose, and the losses are listed in the ``README.md`` next to this file:
``cr:EvaluationResult`` is a single metric/value pair, so per-test-case
results, per-attempt results and null ("not verified") grader scores are
summarized rather than carried. The summary keeps them countable instead of
silently folding them into failures.

Usage::

    python evalport_to_croissant_tasks.py SUITE.json RESULTSET.json \
        --base-uri https://example.org/evals/ > task.jsonld

Only the standard library is needed to convert. Validating the output against
the Croissant Tasks SHACL shapes needs ``rdflib`` and ``pyshacl`` (see the
tests next to this file).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

CONTEXT = {
    "croissant": "http://mlcommons.org/croissant/",
    "schema": "https://schema.org/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def _result(metric: str, value: Any) -> Dict[str, Any]:
    # JSON-LD maps a JSON float to xsd:double, which the Croissant Tasks shapes
    # do not accept (they allow xsd:decimal, xsd:integer, xsd:string or a
    # schema:QuantitativeValue), so non-integers are emitted as typed decimals.
    if isinstance(value, float):
        value = {"@value": format(value, ".6g"), "@type": "xsd:decimal"}
    return {"@type": "croissant:EvaluationResult", "croissant:metric": metric, "croissant:value": value}


def _grader_metrics(result_set: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-grader pass rate and mean score, computed over non-null scores only.

    EvalPort Rule 6: a null score means "not verified" and MUST NOT count as a
    scored failure, so null-scored grader results are excluded from each
    grader's denominator and reported separately as ``<grader>/unscored``.
    """
    per: Dict[str, Dict[str, Any]] = {}
    for res in result_set.get("results", []):
        for gr in res.get("grader_results", []):
            b = per.setdefault(gr["grader_id"], {"scores": [], "passed": 0, "unscored": 0})
            if gr.get("score") is None:
                b["unscored"] += 1
                continue
            b["scores"].append(float(gr["score"]))
            b["passed"] += 1 if gr.get("passed") else 0
    out: List[Dict[str, Any]] = []
    for gid in sorted(per):
        b = per[gid]
        scored = len(b["scores"])
        if scored:
            out.append(_result(f"{gid}/pass_rate", round(b["passed"] / scored, 6)))
            out.append(_result(f"{gid}/mean_score", round(sum(b["scores"]) / scored, 6)))
        out.append(_result(f"{gid}/scored_count", scored))
        out.append(_result(f"{gid}/unscored", b["unscored"]))
    return out


def to_croissant_tasks(
    suite: Dict[str, Any],
    result_set: Dict[str, Any],
    base_uri: str = "urn:evalport:",
    suite_url: Optional[str] = None,
    result_set_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a Croissant Tasks JSON-LD document for one suite and one run."""
    if result_set.get("suite_id") != suite.get("id"):
        raise ValueError(
            f"ResultSet.suite_id {result_set.get('suite_id')!r} does not match Suite.id {suite.get('id')!r}"
        )
    problem_id = f"{base_uri}suite/{suite['id']}"
    solution_id = f"{base_uri}run/{result_set['run_id']}"

    results = result_set.get("results", [])
    total = len(results)
    passed = sum(1 for r in results if r.get("passed") is True)
    unscored_cases = sum(
        1
        for r in results
        if r.get("grader_results") and all(g.get("score") is None for g in r["grader_results"])
    )

    evaluation_results = [
        _result("test_cases/total", total),
        _result("test_cases/passed", passed),
        # Cases with no verdict at all (every grader null-scored) are reported,
        # not hidden inside the failure count (EvalPort Rule 6).
        _result("test_cases/unscored", unscored_cases),
    ]
    if total:
        evaluation_results.append(_result("test_cases/pass_rate", round(passed / total, 6)))
    evaluation_results += _grader_metrics(result_set)

    problem: Dict[str, Any] = {
        "@type": "croissant:TaskProblem",
        "@id": problem_id,
        "schema:name": suite.get("name") or suite["id"],
        "croissant:input": {"@type": "schema:Dataset", "@id": suite_url or f"{problem_id}#test_cases"},
        # A TaskProblem must leave at least one component open (a Spec):
        # the model/system under test is what a solution provides.
        "croissant:implementation": {"@type": "croissant:ImplementationSpec", "schema:name": "system under test"},
        "croissant:output": {"@type": "croissant:OutputSpec", "schema:name": "EvalPort ResultSet"},
    }
    if suite.get("description"):
        problem["schema:description"] = suite["description"]

    implementation: Dict[str, Any] = {"@type": "schema:SoftwareApplication"}
    provider = result_set.get("provider") or {}
    runner = result_set.get("runner") or {}
    if provider.get("model"):
        implementation["schema:name"] = provider["model"]
    elif runner.get("name"):
        implementation["schema:name"] = runner["name"]
    else:
        implementation["schema:name"] = "unspecified"
    if runner.get("version"):
        implementation["schema:softwareVersion"] = runner["version"]

    solution: Dict[str, Any] = {
        "@type": "croissant:TaskSolution",
        "@id": solution_id,
        "schema:isBasedOn": {"@id": problem_id},
        "croissant:implementation": implementation,
        "croissant:output": {"@type": "schema:Dataset", "@id": result_set_url or f"{solution_id}#resultset"},
        "schema:dateCreated": result_set["started_at"],
    }

    evaluation = {
        "@type": "croissant:EvaluationTask",
        "@id": f"{solution_id}#evaluation",
        "croissant:evaluatedSolution": {"@id": solution_id},
        "croissant:evaluationResults": evaluation_results,
    }
    return {"@context": CONTEXT, "@graph": [problem, solution, evaluation]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suite")
    ap.add_argument("result_set")
    ap.add_argument("--base-uri", default="urn:evalport:")
    ap.add_argument("--suite-url")
    ap.add_argument("--result-set-url")
    args = ap.parse_args(argv)
    with open(args.suite, encoding="utf-8") as f:
        suite = json.load(f)
    with open(args.result_set, encoding="utf-8") as f:
        result_set = json.load(f)
    doc = to_croissant_tasks(suite, result_set, args.base_uri, args.suite_url, args.result_set_url)
    json.dump(doc, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
