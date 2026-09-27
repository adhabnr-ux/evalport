#!/usr/bin/env python3
"""Demo 2 -- results portability: DeepEval results -> EvalPort ResultSet -> Haystack -> EvalPort.

1. Runs a REAL deepeval.evaluate() -- offline, with DeepEval's own
   deterministic ExactMatchMetric -- over the first 10 GSM8K cases from
   benchmarks/ and the synthetic outputs in fixtures/gsm8k_outputs.json. The
   result is DeepEval's native EvaluationResult / TestResult / MetricData.
2. deepeval_openeval_adapter.test_results_to_openeval() -> EvalPort ResultSet,
   validated, serialized to JSON and parsed back (what actually crosses a
   tool boundary).
3. Imports that ResultSet into Haystack's native results container,
   haystack.evaluation.EvaluationRunResult, and asks Haystack for its own
   aggregated/detailed report.
4. haystack_openeval_adapter.evaluation_result_to_openeval() -> a second
   ResultSet, validated, and compared per test case with step 1 and step 2.

Finding this demo surfaces rather than hides: no adapter in adapters/ can
import a ResultSet into a framework's native results type -- every
framework adapter's from_openeval() takes a *Suite*. Step 3 therefore uses
`resultset_to_haystack()` below, a 15-line bridge written for this demo.

Exit status 0 only if every ResultSet validates and every per-test-case
score, pass/fail and actual_output agrees across DeepEval's native objects,
both ResultSets and Haystack's report, and every other difference between
the two ResultSets is listed in KNOWN_LOSSY. Otherwise 1.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = REPO / "benchmarks" / "gsm8k" / "gsm8k.json"
OUTPUTS = HERE / "fixtures" / "gsm8k_outputs.json"
STARTED_AT = "2026-09-27T00:00:00Z"  # fixed so the demo output is deterministic

# --- offline guard (see 1_dataset_portability.py) -----------------------------
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
NETWORK_ATTEMPTS: List[str] = []


def _blocked(*args: Any, **kwargs: Any) -> Any:
    NETWORK_ATTEMPTS.append(repr(args[1:] or args)[:80])
    raise OSError("network access is disabled in the EvalPort interop demos")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
socket.getaddrinfo = _blocked  # type: ignore[assignment]

import deepeval  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig  # noqa: E402
from deepeval.metrics import ExactMatchMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from haystack.evaluation import EvaluationRunResult  # noqa: E402

import deepeval_openeval_adapter as de_adapter  # noqa: E402
import haystack_openeval_adapter as hs_adapter  # noqa: E402
from openeval.validate import validate_result_set  # noqa: E402

# ResultSet paths that differ between the DeepEval-produced ResultSet and the
# one that came back through Haystack, and why. `[]` = any list index.
KNOWN_LOSSY = {
    "runner": "EvaluationRunResult has no runner/tool-version field",
    "completed_at": "EvaluationRunResult has no timestamps",
    "metadata": "ResultSet-level metadata ({'openeval': {'source': 'deepeval'}}) has no slot",
    "summary.avg_score": "the Haystack adapter's summary omits avg_score (optional field)",
    "summary.skipped": "the Haystack adapter's summary omits skipped (optional field)",
    "results[].metadata": "per-result metadata (deepeval index/conversational/multimodal, plus "
                          "the test case's own metadata as user_metadata) has no slot",
    "results[].grader_results[].reason": "EvaluationRunResult stores only numeric individual_scores; "
                                         "MetricData.reason is dropped",
    "results[].grader_results[].metadata": "threshold, strict_mode, evaluation_model, metric_name "
                                           "are replaced by Haystack's aggregate_score",
}
# Fields that carry the actual evaluation outcome: these must be identical.
MUST_MATCH = ("test_case_id", "actual_output", "passed", "grader_id", "score")


def run_deepeval(suite: Dict[str, Any], outputs: Dict[str, str]):
    cases = [LLMTestCase(**kw, actual_output=outputs[kw["name"]])
             for kw in de_adapter.from_openeval(suite)]
    # deepeval writes a .deepeval/ folder into the cwd and prints a summary;
    # keep both out of the repo and out of this demo's report.
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        os.chdir(tmp)
        try:
            result = deepeval.evaluate(
                cases, [ExactMatchMetric()],
                async_config=AsyncConfig(run_async=False),
                display_config=DisplayConfig(show_indicator=False, print_results=False),
                cache_config=CacheConfig(write_cache=False, use_cache=False),
            )
        finally:
            os.chdir(cwd)
    return result


def resultset_to_haystack(rs: Dict[str, Any]) -> EvaluationRunResult:
    """Demo-local bridge: EvalPort ResultSet -> haystack EvaluationRunResult.

    No adapter provides this direction (see module docstring). One metric
    per grader_id; `id` and `predicted_answers` columns carry test_case_id
    and actual_output, the column names haystack_openeval_adapter reads back.
    """
    results = rs["results"]
    grader_ids = sorted({g["grader_id"] for r in results for g in r["grader_results"]})
    metrics = {}
    for gid in grader_ids:
        scores = [next(g["score"] for g in r["grader_results"] if g["grader_id"] == gid)
                  for r in results]
        metrics[gid] = {"individual_scores": scores, "score": sum(scores) / len(scores)}
    inputs = {"id": [r["test_case_id"] for r in results],
              "predicted_answers": [r.get("actual_output") for r in results]}
    return EvaluationRunResult(run_name=rs["run_id"], inputs=inputs, results=metrics)


def flatten(value: Any, path: str = "") -> Dict[str, Any]:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            out.update(flatten(v, f"{path}.{k}" if path else k))
        return out if value else {path: {}}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        out = {}
        for i, v in enumerate(value):
            out.update(flatten(v, f"{path}[{i}]"))
        return out
    return {path: value}


def generic(path: str) -> str:
    """results[3].grader_results[0].metadata.threshold -> results[].grader_results[].metadata"""
    p = re.sub(r"\[\d+\]", "[]", path)
    for known in sorted(KNOWN_LOSSY, key=len, reverse=True):
        if p == known or p.startswith(known + "."):
            return known
    return p


def fmt(v: Any) -> str:
    return "-" if v is None else json.dumps(v)


def main() -> int:
    failures: List[str] = []
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    suite["test_cases"] = suite["test_cases"][:10]
    outputs = json.loads(OUTPUTS.read_text(encoding="utf-8"))["outputs"]
    print("EvalPort interop demo 2: results portability "
          "(DeepEval results -> EvalPort -> Haystack -> EvalPort)")

    # 1. DeepEval, natively.
    eval_result = run_deepeval(suite, outputs)
    native = {tr.name: tr for tr in eval_result.test_results}
    md0 = eval_result.test_results[0].metrics_data[0]
    print(f"\n[1] deepeval {deepeval.__version__}: evaluate() -> {type(eval_result).__name__} with "
          f"{len(native)} {type(eval_result.test_results[0]).__name__} "
          f"(metric '{md0.name}', threshold {md0.threshold})")

    # 2. -> EvalPort ResultSet, over the wire.
    rs_de = de_adapter.test_results_to_openeval(
        eval_result, suite_id=suite["id"], run_id="gsm8k-deepeval-exact-match",
        started_at=STARTED_AT, runner_version=deepeval.__version__)
    v1 = validate_result_set(rs_de)
    rs_de = json.loads(json.dumps(rs_de))
    print(f"[2] test_results_to_openeval() -> ResultSet: {len(rs_de['results'])} results, "
          f"valid={v1.valid}, {len(json.dumps(rs_de))} bytes of JSON")
    if not v1.valid:
        failures.append(f"DeepEval ResultSet invalid: {v1.errors[:3]}")

    # 3. -> Haystack's native results container.
    hs_run = resultset_to_haystack(rs_de)
    agg = hs_run.aggregated_report()
    detailed = hs_run.detailed_report()
    print(f"[3] resultset_to_haystack() -> {type(hs_run).__name__}; Haystack's own "
          f"aggregated_report(): {dict(zip(agg['metrics'], agg['score']))}")

    # 4. -> back to EvalPort through the Haystack adapter.
    rs_hs = hs_adapter.evaluation_result_to_openeval(
        # run_id is not read from EvaluationRunResult.run_name automatically;
        # pass it through so the run keeps its identity.
        hs_run, suite_id=suite["id"], run_id=hs_run.run_name, started_at=STARTED_AT)
    v2 = validate_result_set(rs_hs)
    print(f"[4] evaluation_result_to_openeval() -> ResultSet: {len(rs_hs['results'])} results, "
          f"valid={v2.valid}")
    if not v2.valid:
        failures.append(f"Haystack ResultSet invalid: {v2.errors[:3]}")

    # Per-test-case agreement: native DeepEval vs ResultSet #1 vs Haystack vs ResultSet #2.
    by_id_de = {r["test_case_id"]: r for r in rs_de["results"]}
    by_id_hs = {r["test_case_id"]: r for r in rs_hs["results"]}
    hs_rows = dict(zip(detailed["id"], detailed[agg["metrics"][0]]))
    print(f"\n  {'test case':<9} {'expected':>8}  {'actual_output':<13} "
          f"{'DeepEval':>8} {'ResultSet':>9} {'Haystack':>8} {'ResultSet2':>10}  agree")
    if set(native) != set(by_id_de) or set(by_id_de) != set(by_id_hs):
        failures.append("test_case_id sets differ between native results and the ResultSets")
    for tc in suite["test_cases"]:
        tid = tc["id"]
        md = native[tid].metrics_data[0]
        r1, r2 = by_id_de[tid], by_id_hs[tid]
        g1, g2 = r1["grader_results"][0], r2["grader_results"][0]
        scores = (md.score, g1["score"], hs_rows[tid], g2["score"])
        passes = (native[tid].success, r1["passed"], r2["passed"])
        ok = (len(set(scores)) == 1 and len(set(passes)) == 1
              and r1.get("actual_output") == r2.get("actual_output") == outputs[tid]
              and g1["grader_id"] == g2["grader_id"])
        if not ok:
            failures.append(f"{tid}: scores {scores}, passed {passes} disagree")
        print(f"  {tid:<9} {tc['expected_output']:>8}  {json.dumps(outputs[tid]):<13} "
              f"{fmt(scores[0]):>8} {fmt(scores[1]):>9} {fmt(scores[2]):>8} {fmt(scores[3]):>10}  "
              f"{'yes' if ok else 'NO'}")
    pr = (rs_de["summary"]["pass_rate"], rs_hs["summary"]["pass_rate"], agg["score"][0])
    print(f"  pass rate: DeepEval ResultSet {pr[0]:.2f} | Haystack ResultSet {pr[1]:.2f} | "
          f"Haystack aggregated_report {pr[2]:.2f}")
    if len(set(pr)) != 1:
        failures.append(f"pass rates disagree: {pr}")

    # Everything else that changed between the two ResultSets.
    f1, f2 = flatten(rs_de), flatten(rs_hs)
    changed: Dict[str, int] = {}
    for path in sorted(set(f1) | set(f2)):
        if f1.get(path) != f2.get(path):
            key = generic(path)
            changed[key] = changed.get(key, 0) + 1
    print("\n  Fields that did not survive DeepEval -> EvalPort -> Haystack -> EvalPort:")
    for key, count in changed.items():
        leaf = key.rsplit(".", 1)[-1].split("[")[0]
        if leaf in MUST_MATCH:
            failures.append(f"outcome field changed: {key} ({count}x)")
        why = KNOWN_LOSSY.get(key)
        if why is None:
            failures.append(f"undocumented change: {key} ({count}x)")
        print(f"    {'lossy' if why else 'UNEXPECTED'}: {key}  ({count} value(s)) -- "
              f"{why or 'not a documented lossy field'}")

    print(f"\nnetwork connections attempted: {len(NETWORK_ATTEMPTS)}")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS -- all 10 per-test-case scores, pass/fail verdicts and outputs agree "
          "across DeepEval, both ResultSets and Haystack; the only losses are listed above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
