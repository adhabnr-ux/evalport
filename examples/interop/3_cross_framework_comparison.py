#!/usr/bin/env python3
"""Demo 3 -- cross-framework comparison joined with ResultSet.group.

The same 10 GSM8K cases (benchmarks/) and the same synthetic model outputs
(fixtures/gsm8k_outputs.json) are scored by three frameworks' own built-in,
deterministic exact-match metrics -- each run through that framework's real
evaluation entry point, offline:

  * DeepEval 'deepeval'  -- deepeval.evaluate() + deepeval.metrics.ExactMatchMetric
  * DSPy     'dspy'      -- dspy.Evaluate() + dspy.evaluate.answer_exact_match
  * Haystack 'haystack'  -- AnswerExactMatchEvaluator().run() -> EvaluationRunResult

Each framework's native result is converted with its own adapter from
adapters/, stamped with the same `group.group_id` (plus `role`/`label`/
`sequence`, spec/SPEC.md "Grouped/Sibling ResultSets") and written to disk as
three independent ResultSet files. A consumer that knows nothing about any of
the frameworks then loads the files, joins them on group_id + test_case_id,
and builds the comparison table: the use case Discussion #45 designed
`group` for.

Then, opt-in, the suite's OWN exact_match grader is run inside DeepEval and
DSPy through the adapters' grader helpers (graders_to_deepeval_metrics(),
graders_to_dspy_metrics()) and checked against an independent transcription
of the reference runner's gradeExactMatch.

Exit status 0 only if all ResultSets validate, the three built-in-metric
ResultSets form one complete group over identical test_case_ids, every
per-test-case score in each ResultSet equals the score that framework
produced natively, and the suite grader gives the same verdict in DeepEval,
DSPy and the reference transcription. Disagreements BETWEEN the frameworks'
built-in metrics are the finding, not a failure -- they're printed.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import socket
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = REPO / "benchmarks" / "gsm8k" / "gsm8k.json"
OUTPUTS = HERE / "fixtures" / "gsm8k_outputs.json"
STARTED_AT = "2026-09-27T00:00:00Z"  # fixed so the demo output is deterministic
GROUP_ID = "gsm8k-exact-match-3-frameworks"

# --- offline guard (see 1_dataset_portability.py) -----------------------------
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
NETWORK_ATTEMPTS: List[str] = []


def _blocked(*args: Any, **kwargs: Any) -> Any:
    NETWORK_ATTEMPTS.append(repr(args[1:] or args)[:80])
    raise OSError("network access is disabled in the EvalPort interop demos")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
socket.getaddrinfo = _blocked  # type: ignore[assignment]

# deepeval before dspy: dspy installs a lazy `openai` module proxy that breaks
# deepeval's own `openai.types` imports if dspy is imported first.
import deepeval  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig  # noqa: E402
from deepeval.metrics import ExactMatchMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
import dspy  # noqa: E402
from dspy.evaluate import answer_exact_match  # noqa: E402
import haystack  # noqa: E402
from haystack.components.evaluators import AnswerExactMatchEvaluator  # noqa: E402
from haystack.evaluation import EvaluationRunResult  # noqa: E402

import deepeval_openeval_adapter as de_adapter  # noqa: E402
import dspy_openeval_adapter as dspy_adapter  # noqa: E402
import haystack_openeval_adapter as hs_adapter  # noqa: E402
from openeval.validate import validate_result_set  # noqa: E402

Native = Dict[str, float]  # test_case_id -> score, exactly as the framework produced it


def run_deepeval(suite: Dict[str, Any], outputs: Dict[str, str]) -> Tuple[Dict[str, Any], Native]:
    cases = [LLMTestCase(**kw, actual_output=outputs[kw["name"]])
             for kw in de_adapter.from_openeval(suite)]
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        os.chdir(tmp)  # deepeval writes .deepeval/ into the cwd
        try:
            result = deepeval.evaluate(
                cases, [ExactMatchMetric()],
                async_config=AsyncConfig(run_async=False),
                display_config=DisplayConfig(show_indicator=False, print_results=False),
                cache_config=CacheConfig(write_cache=False, use_cache=False))
        finally:
            os.chdir(cwd)
    native = {tr.name: tr.metrics_data[0].score for tr in result.test_results}
    rs = de_adapter.test_results_to_openeval(
        result, suite_id=suite["id"], run_id="gsm8k-deepeval", started_at=STARTED_AT,
        runner_version=deepeval.__version__)
    return rs, native


class FixedOutputs(dspy.Module):
    """A DSPy program that returns the fixture output instead of calling an LM."""

    def __init__(self, by_question: Dict[str, str]):
        super().__init__()
        self.by_question = by_question

    def forward(self, question: str) -> dspy.Prediction:
        return dspy.Prediction(answer=self.by_question[question])


def run_dspy(suite: Dict[str, Any], outputs: Dict[str, str]) -> Tuple[Dict[str, Any], Native]:
    devset = dspy_adapter.from_openeval(suite, input_keys=["question"], expected_key="answer")
    by_question = {tc["input"]: outputs[tc["id"]] for tc in suite["test_cases"]}
    logging.getLogger("dspy").setLevel(logging.WARNING)  # silence the "Average Metric" log line
    evaluate = dspy.Evaluate(devset=devset, metric=answer_exact_match, num_threads=1,
                             display_progress=False, display_table=False)
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluate(FixedOutputs(by_question))
    native = {getattr(ex, "_openeval_test_case_id"): float(score)
              for ex, _, score in result.results}
    rs = dspy_adapter.evaluation_result_to_openeval(
        result, suite_id=suite["id"], metric=answer_exact_match, run_id="gsm8k-dspy",
        started_at=STARTED_AT)
    return rs, native


def run_haystack(suite: Dict[str, Any], outputs: Dict[str, str]) -> Tuple[Dict[str, Any], Native]:
    cols = hs_adapter.from_openeval(suite, input_keys=["question"], expected_key="ground_truth")
    predicted = [outputs[i] for i in cols["id"]]
    scored = AnswerExactMatchEvaluator().run(ground_truth_answers=cols["ground_truth"],
                                             predicted_answers=predicted)
    run = EvaluationRunResult(run_name="gsm8k-haystack",
                              inputs={**cols, "predicted_answers": predicted},
                              results={"answer_exact_match": scored})
    native = dict(zip(cols["id"], (float(s) for s in scored["individual_scores"])))
    rs = hs_adapter.evaluation_result_to_openeval(run, suite_id=suite["id"],
                                                  run_id=run.run_name, started_at=STARTED_AT)
    return rs, native


# --- the suite's OWN grader, run inside DeepEval and DSPy ----------------------
# The adapters' opt-in grader helpers turn the suite's exact_match grader into
# a DeepEval metric / DSPy metric with EvalPort's semantics, so both frameworks
# grade by the suite's definition instead of their own "exact match".


def spec_exact_match(grader: Dict[str, Any], actual: str, expected: str) -> bool:
    """Independent transcription of the reference runner's gradeExactMatch
    (cli/src/run/graders/tier1.ts), used only to check the two adapters."""
    params = grader.get("params") or {}
    a, e = actual, expected
    if params.get("trim_whitespace") is not False:
        a, e = a.strip(), e.strip()
    if params.get("ignore_case") is True:
        a, e = a.lower(), e.lower()
    return a == e


def suite_grader_deepeval(suite: Dict[str, Any], outputs: Dict[str, str],
                          grader_id: str) -> Tuple[Dict[str, Any], Native]:
    metric = de_adapter.graders_to_deepeval_metrics(suite)[grader_id]
    cases = [LLMTestCase(**kw, actual_output=outputs[kw["name"]])
             for kw in de_adapter.from_openeval(suite)]
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        os.chdir(tmp)
        try:
            result = deepeval.evaluate(
                cases, [metric],
                async_config=AsyncConfig(run_async=False),
                display_config=DisplayConfig(show_indicator=False, print_results=False),
                cache_config=CacheConfig(write_cache=False, use_cache=False))
        finally:
            os.chdir(cwd)
    native = {tr.name: tr.metrics_data[0].score for tr in result.test_results}
    rs = de_adapter.test_results_to_openeval(
        result, suite_id=suite["id"], run_id="gsm8k-deepeval-suite-grader",
        started_at=STARTED_AT, runner_version=deepeval.__version__)
    return rs, native


def suite_grader_dspy(suite: Dict[str, Any], outputs: Dict[str, str],
                      grader_id: str) -> Tuple[Dict[str, Any], Native]:
    metric = dspy_adapter.graders_to_dspy_metrics(
        suite, expected_key="answer", output_key="answer")[grader_id]
    devset = dspy_adapter.from_openeval(suite, input_keys=["question"], expected_key="answer")
    by_question = {tc["input"]: outputs[tc["id"]] for tc in suite["test_cases"]}
    evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=1,
                             display_progress=False, display_table=False)
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluate(FixedOutputs(by_question))
    native = {getattr(ex, "_openeval_test_case_id"): float(score)
              for ex, _, score in result.results}
    rs = dspy_adapter.evaluation_result_to_openeval(
        result, suite_id=suite["id"], metric=metric, run_id="gsm8k-dspy-suite-grader",
        started_at=STARTED_AT)
    return rs, native


def run_suite_grader(suite: Dict[str, Any], outputs: Dict[str, str],
                     expected: Dict[str, str], failures: List[str]) -> None:
    grader = suite["graders"][0]
    gid = grader["id"]
    print(f"\n  Opt-in: the suite's own grader {gid} ({grader['type']} "
          f"{json.dumps(grader.get('params', {}))}) run inside each framework:")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs = [("deepeval", *suite_grader_deepeval(suite, outputs, gid)),
                ("dspy", *suite_grader_dspy(suite, outputs, gid))]
    notes = sorted({str(w.message).split(": ", 1)[1] for w in caught
                    if issubclass(w.category, UserWarning) and "exact_match param" in str(w.message)})
    for role, rs, native in runs:
        v = validate_result_set(rs)
        ids = {g["grader_id"] for r in rs["results"] for g in r["grader_results"]}
        print(f"    {role:<9} valid={v.valid}  grader_id={', '.join(sorted(ids))}")
        if not v.valid:
            failures.append(f"{role} suite-grader ResultSet invalid: {v.errors[:3]}")
        if ids != {gid}:
            failures.append(f"{role} suite-grader ResultSet grader_ids {ids} != {{{gid!r}}}")
        for r in rs["results"]:
            if r["grader_results"][0]["score"] != native[r["test_case_id"]]:
                failures.append(f"{role}/{r['test_case_id']}: ResultSet score != native score")
    print(f"\n  {'test case':<9} {'expected':>8}  {'output':<11}{'deepeval':>10}{'dspy':>10}"
          f"{'spec':>10}  agree")
    passes = {role: {r["test_case_id"]: r["passed"] for r in rs["results"]} for role, rs, _ in runs}
    for tid in sorted(expected, key=lambda t: int(t.split("_")[1])):
        spec = spec_exact_match(grader, outputs[tid], expected[tid])
        cells = [passes["deepeval"][tid], passes["dspy"][tid], spec]
        agree = len(set(cells)) == 1
        if not agree:
            failures.append(f"suite grader {gid}/{tid}: deepeval/dspy/spec disagree: {cells}")
        print(f"  {tid:<9} {expected[tid]:>8}  {json.dumps(outputs[tid]):<11}" +
              "".join(f"{'pass' if c else 'fail':>10}" for c in cells) +
              f"  {'yes' if agree else 'NO'}")
    rates = [sum(p.values()) / len(p) for p in (passes["deepeval"], passes["dspy"])]
    spec_rate = sum(spec_exact_match(grader, outputs[t], expected[t]) for t in expected) / len(expected)
    print("  pass rate" + " " * 30 + "".join(f"{r:>10.2f}" for r in rates + [spec_rate]))
    for note in notes:
        print(f"  note: {note}")


PRODUCERS = [
    ("deepeval", f"DeepEval {deepeval.__version__} ExactMatchMetric", run_deepeval),
    ("dspy", f"DSPy {dspy.__version__} answer_exact_match", run_dspy),
    ("haystack", f"Haystack {haystack.__version__} AnswerExactMatchEvaluator", run_haystack),
]


def main() -> int:
    failures: List[str] = []
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    suite["test_cases"] = suite["test_cases"][:10]
    outputs = json.loads(OUTPUTS.read_text(encoding="utf-8"))["outputs"]
    print("EvalPort interop demo 3: cross-framework comparison joined with ResultSet.group")

    # --- producer side: each framework emits its own ResultSet file ----------
    natives: Dict[str, Native] = {}
    with tempfile.TemporaryDirectory() as outdir:
        for seq, (role, label, run) in enumerate(PRODUCERS):
            rs, natives[role] = run(suite, outputs)
            rs["group"] = {"group_id": GROUP_ID, "role": role, "label": label, "sequence": seq}
            path = Path(outdir) / f"{rs['run_id']}.resultset.json"
            path.write_text(json.dumps(rs, indent=2), encoding="utf-8")
            v = validate_result_set(rs)
            print(f"  wrote {path.name:<34} role={role:<9} valid={v.valid}  ({label})")
            if not v.valid:
                failures.append(f"{role}: ResultSet invalid: {v.errors[:3]}")

        # --- consumer side: only the files, no framework knowledge ------------
        loaded = [json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted(Path(outdir).glob("*.resultset.json"))]

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rs in loaded:
        groups.setdefault(rs["group"]["group_id"], []).append(rs)
    members = sorted(groups.get(GROUP_ID, []), key=lambda rs: rs["group"]["sequence"])
    roles = [m["group"]["role"] for m in members]
    if roles != [r for r, _, _ in PRODUCERS] or len(groups) != 1:
        failures.append(f"group join failed: groups={list(groups)}, roles={roles}")
    if len({m["suite_id"] for m in members}) != 1:
        failures.append("group members disagree on suite_id")
    table: Dict[str, Dict[str, Tuple[float, bool]]] = {}
    for m in members:
        for r in m["results"]:
            g = r["grader_results"][0]
            table.setdefault(r["test_case_id"], {})[m["group"]["role"]] = (g["score"], r["passed"])
    for m in members:
        ids = {r["test_case_id"] for r in m["results"]}
        if ids != set(table):
            failures.append(f"{m['group']['role']}: test_case_ids differ from the other members")

    # Lossless check: every ResultSet score == the framework's own native score.
    for role, native in natives.items():
        for tid, score in native.items():
            if table.get(tid, {}).get(role, (None,))[0] != score:
                failures.append(f"{role}/{tid}: ResultSet score {table.get(tid, {}).get(role)} "
                                f"!= native score {score}")

    expected = {tc["id"]: tc["expected_output"] for tc in suite["test_cases"]}
    print(f"\n  group_id={GROUP_ID}: {len(members)} members joined on test_case_id\n")
    print(f"  {'test case':<9} {'expected':>8}  {'output':<11}" +
          "".join(f"{r:>10}" for r in roles) + "  verdict")
    disagreements = []
    for tid in sorted(table, key=lambda t: int(t.split("_")[1])):
        cells = [table[tid].get(r, (None, None)) for r in roles]
        passes = {c[1] for c in cells}
        verdict = "all pass" if passes == {True} else "all fail" if passes == {False} else "DISAGREE"
        if verdict == "DISAGREE":
            disagreements.append((tid, [r for r, c in zip(roles, cells) if c[1]]))
        print(f"  {tid:<9} {expected[tid]:>8}  {json.dumps(outputs[tid]):<11}" +
              "".join(f"{'pass' if c[1] else 'fail':>10}" for c in cells) + f"  {verdict}")
    rates = {m["group"]["role"]: m["summary"]["pass_rate"] for m in members}
    print("  pass rate" + " " * 30 + "".join(f"{rates[r]:>10.2f}" for r in roles))

    print(f"\n  {len(disagreements)}/{len(table)} test cases get different verdicts from different "
          "frameworks on identical outputs:")
    for tid, passed_by in disagreements:
        print(f"    {tid}: {json.dumps(outputs[tid])} vs {json.dumps(expected[tid])} -- "
              f"passed by {', '.join(passed_by)} only")
    grader = suite["graders"][0]
    print(f"  Why: DeepEval strips whitespace then compares; Haystack compares raw strings; DSPy "
          f"lowercases and drops punctuation/articles first. Converting a suite into a framework "
          f"does not carry the suite's own grader "
          f"({grader['type']} {json.dumps(grader.get('params', {}))}) with it (demo 1), so each "
          f"framework's built-in metric applies its own semantics.")

    run_suite_grader(suite, outputs, expected, failures)

    print(f"\nnetwork connections attempted: {len(NETWORK_ATTEMPTS)}")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS -- 3 valid ResultSets joined via group.group_id; every score in them "
          "equals the framework's native score; the suite's own grader gives identical verdicts "
          "in DeepEval and DSPy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
