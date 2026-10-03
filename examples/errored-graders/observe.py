"""Run real DeepEval and Inspect AI evaluations in which one grader raises.

No LLM, network or API key is involved: the "model" is canned (Inspect's ``mockllm``,
DeepEval test cases that already carry ``actual_output``) and the graders are plain
Python.  What is *real* is the framework: the code that decides what a row looks like
after a grader raised is DeepEval's and Inspect AI's own.

Each framework is reduced to the same neutral ``Row`` records, which
``errored_graders_to_evalport.py`` then writes as EvalPort documents.
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The four constructed cases. ``exact`` raises on q4, ``flaky`` raises on q2, q3 and q4.
#   q1: both graders run and the answer is right         -> clean pass
#   q2: exact passes, flaky raises                       -> one grader never ran, the other passed
#   q3: exact fails,  flaky raises                       -> one verified failure, one grader never ran
#   q4: both graders raise                               -> nothing ran
QUESTION = "What is 2+2?"
CASES = [
    {"id": "q1", "input": f"{QUESTION} (q1)", "expected": "4", "actual": "4"},
    {"id": "q2", "input": f"{QUESTION} (q2)", "expected": "4", "actual": "4"},
    {"id": "q3", "input": f"{QUESTION} (q3)", "expected": "4", "actual": "5"},
    {"id": "q4", "input": f"{QUESTION} (q4)", "expected": "4", "actual": "4"},
]
RAISES = {"exact": {"q4"}, "flaky": {"q2", "q3", "q4"}}
GRADERS = ["exact", "flaky"]


@dataclass
class GraderObs:
    name: str
    score: Optional[float]          # None when the grader produced no score
    success: Optional[bool]         # the framework's own pass flag for this grader, if it has one
    error: Optional[str]            # the framework's own error text, if any


@dataclass
class Row:
    case_id: str
    framework: str
    graders: List[GraderObs]
    native_row_success: Optional[bool]   # the framework's own row-level pass flag, None if it has none
    native_row_error: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    framework: str
    version: str
    rows: List[Row]
    run_level: Dict[str, Any]


def _case_id_of(text: str) -> str:
    """Case id from an input like ``"What is 2+2? (q3)"`` (DeepEval only hands the grader the input)."""
    return text[text.rindex("(") + 1 : -1]


def _expected_error(grader: str, case_id: str) -> bool:
    return case_id in RAISES[grader]


# --------------------------------------------------------------------------- DeepEval

def observe_deepeval(ignore_errors: bool = True) -> Observation:
    """With ``ignore_errors=False`` (DeepEval's default) the exception propagates out of ``evaluate``."""
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    import deepeval
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase

    class Metric(BaseMetric):
        threshold = 1.0
        async_mode = False
        evaluation_model = None
        strict_mode = False

        def __init__(self, name: str) -> None:
            self._name = name
            self.include_reason = True
            self.verbose_mode = False

        @property
        def __name__(self) -> str:  # DeepEval reads the metric name from here
            return self._name

        def measure(self, test_case, *a, **k):
            if _expected_error(self._name, _case_id_of(test_case.input)):
                raise RuntimeError(f"{self._name} grader unreachable")
            self.score = 1.0 if test_case.actual_output == test_case.expected_output else 0.0
            self.success = self.score >= self.threshold
            self.reason = "exact string comparison"
            self.error = None
            return self.score

        async def a_measure(self, test_case, *a, **k):
            return self.measure(test_case)

        def is_successful(self) -> bool:
            return bool(self.success)

    cases = [LLMTestCase(input=c["input"], actual_output=c["actual"], expected_output=c["expected"])
             for c in CASES]
    # DeepEval writes a local cache/results folder; keep it out of the working tree.
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
              result = evaluate(
                  cases, [Metric(g) for g in GRADERS],
                  error_config=ErrorConfig(ignore_errors=ignore_errors),
                  display_config=DisplayConfig(show_indicator=False, print_results=False),
                  async_config=AsyncConfig(run_async=False),
                  cache_config=CacheConfig(write_cache=False),
              )
        finally:
            os.chdir(old)

    rows: List[Row] = []
    for case, tr in zip(CASES, result.test_results):
        by_name = {m.name: m for m in tr.metrics_data}
        rows.append(Row(
            case_id=case["id"], framework="deepeval",
            graders=[GraderObs(g, by_name[g].score, by_name[g].success, by_name[g].error) for g in GRADERS],
            native_row_success=tr.success,
        ))
    return Observation("deepeval", deepeval.__version__, rows, {"ignore_errors": ignore_errors})


# --------------------------------------------------------------------------- Inspect AI

def observe_inspect(scorer_order: Optional[List[str]] = None, fail_on_error: bool = False) -> Observation:
    """``scorer_order`` is the order of the scorer list handed to Inspect (default: GRADERS).
    Rows always list their graders in GRADERS order, whatever the scorer order was."""
    order = list(scorer_order or GRADERS)
    assert sorted(order) == sorted(GRADERS)
    import importlib.metadata as md
    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer
    from inspect_ai.solver import generate

    def make(name: str):
        @scorer(metrics=[accuracy()], name=name)
        def build():
            async def score(state, target):
                if _expected_error(name, str(state.sample_id)):
                    raise RuntimeError(f"{name} grader unreachable")
                ok = state.output.completion.strip() == target.text
                return Score(value=CORRECT if ok else INCORRECT, answer=state.output.completion)
            return score
        return build()

    task = Task(
        dataset=[Sample(id=c["id"], input=c["input"], target=c["expected"]) for c in CASES],
        solver=[generate()],
        scorer=[make(g) for g in order],
    )
    with tempfile.TemporaryDirectory() as tmp:
        logs = inspect_eval(
            task, model="mockllm/model", fail_on_error=fail_on_error, display="none", log_dir=tmp,
            model_args={"custom_outputs": [ModelOutput.from_content("mockllm/model", c["actual"]) for c in CASES]},
        )
        # Samples are read lazily from the log file, so read them before the folder goes away.
        from inspect_ai.log import read_eval_log
        log = read_eval_log(logs[0].location)
    rows: List[Row] = []
    for case in CASES:
        s = next(x for x in log.samples if str(x.id) == case["id"])
        scores = s.scores or {}
        graders = []
        for g in GRADERS:
            sc = scores.get(g)
            if sc is None:
                graders.append(GraderObs(g, None, None, None))
            else:
                graders.append(GraderObs(g, 1.0 if sc.value == "C" else 0.0, None, None))
        rows.append(Row(case["id"], "inspect_ai", graders, native_row_success=None,
                        native_row_error=(s.error.message if s.error else None)))
    results = log.results
    run_level = {
        "status": log.status,
        "total_samples": results.total_samples if results else None,
        "completed_samples": results.completed_samples if results else None,
        "scores": {sc.name: {m: v.value for m, v in sc.metrics.items()} for sc in (results.scores if results else [])},
    }
    run_level["scorer_order"] = order
    return Observation("inspect_ai", md.version("inspect_ai"), rows, run_level)


if __name__ == "__main__":
    for obs in (observe_deepeval(), observe_inspect()):
        print(f"== {obs.framework} {obs.version}  run-level: {obs.run_level}")
        for r in obs.rows:
            print(" ", r.case_id, "native_row_success=", r.native_row_success,
                  "row_error=", (r.native_row_error or "")[:50],
                  [(g.name, g.score, g.success, (g.error or "")[:30]) for g in r.graders])
