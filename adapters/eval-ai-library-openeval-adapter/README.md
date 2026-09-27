# eval-ai-library-openeval-adapter

Convert [Eval-ai-library](https://github.com/meshkovQA/Eval-ai-library)
(PyPI `eval-ai-library`, import path `eval_lib`) test cases and evaluation
results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the
open interchange format for portable LLM evaluation datasets.

Filed and built following
[meshkovQA/Eval-ai-library#2](https://github.com/meshkovQA/Eval-ai-library/issues/2),
at the maintainer's request that it live here: the owner preferred the adapter
"live in the EvalPort repo alongside your other adapters", confirmed that
`eval_lib.evaluation_schema` (`TestCaseResult`, `MetricResult`) "is public and
stable, so you can import it from an external package without any changes on
our side", and noted "This isn't a 'no' to interoperability, it's just where
the code lives".

## Why a standalone package?

Eval-ai-library's result objects are plain dataclasses
(`eval_lib.evaluation_schema.MetricResult` / `TestCaseResult`) and its test
case is a pydantic model (`eval_lib.testcases_schema.EvalTestCase`). They are
public and stable, so nothing in `eval_lib` needs to change for interop: this
package reads those objects (or plain dicts with the same field names) and
emits EvalPort documents, and never imports `eval_lib` at module import time.
It was read directly from `eval-ai-library==0.7.26`'s source rather than
guessed; the field-by-field shape is in the module docstring.

## Install

```
pip install "eval-ai-library-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/eval-ai-library-openeval-adapter"
```

Not yet published to PyPI -- this installs directly from source via pip's
`git+`/`#subdirectory=` support. Add `[eval-ai-library]` to also install the
pinned upstream SDK (`eval-ai-library>=0.7.0,<0.8`), which is optional: the
adapter works on dicts shaped like `eval_lib`'s objects without it.

## Usage

### Test cases -> EvalPort Suite

```python
from eval_lib import EvalTestCase, ExactMatchMetric, AnswerRelevancyMetric
from eval_ai_library_openeval_adapter import to_openeval, from_openeval

test_cases = [
    EvalTestCase(
        name="capital_of_france",
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
        retrieval_context=["Paris is the capital and largest city of France."],
    ),
]
metrics = [
    ExactMatchMetric(threshold=0.5, case_sensitive=False),
    AnswerRelevancyMetric(model="gpt-4o", threshold=0.7),
]

suite = to_openeval(test_cases, metrics, suite_id="rag_smoke")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

recovered = from_openeval(suite)                    # EvalTestCase-shaped dicts
models = from_openeval(suite, as_models=True)       # real EvalTestCase objects
```

One `TestCase` per `EvalTestCase`; one shared `Grader` per distinct metric,
id `gr_<snake_case(metric.name)>` (`answerRelevancyMetric` ->
`gr_answer_relevancy_metric`). `input`, `expected_output`,
`retrieval_context`, `tools_called` and `expected_tools` map 1:1;
`actual_output` (which `eval_lib` stores on the test case) and the
reliability/`extra_fields` data go under `metadata.eval_ai_library` so
`from_openeval()` rebuilds a complete `EvalTestCase`. Test case ids are
`EvalTestCase.name`, falling back to positional `tc_0001`; pass `ids=[...]`
to override.

`metrics` may be real `MetricPattern` instances (best -- see the mapping table
below), `MetricResult`s, `{"name", "threshold", "model"}` dicts, or bare
names. If omitted, graders are derived from each item's `metrics_data`, so a
list of `TestCaseResult` needs nothing else; with no metric information at
all a single placeholder `custom` grader `gr_eval_ai_library` is declared,
because EvalPort requires at least one grader per test case.

### Evaluation results -> EvalPort ResultSet

```python
import asyncio
from eval_lib import evaluate
from eval_ai_library_openeval_adapter import results_to_openeval

# List[Tuple[None, List[TestCaseResult]]], exactly as eval_lib returns it
results = asyncio.run(evaluate(test_cases, metrics, verbose=False))

result_set = results_to_openeval(
    results,
    metrics,                       # same objects -> same grader ids/types as the suite
    suite_id="rag_smoke",
    run_id="run_2026_09_27",
    started_at="2026-09-27T10:00:00Z",
    test_cases=test_cases,         # so ids follow EvalTestCase.name, like the suite
    provider={"model": "gpt-4o"},
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

One `Result` per `TestCaseResult` (`passed` = its `success`, `actual_output`
carried through) and one `GraderResult` per `MetricResult`. Nothing is
recomputed: `passed` is the library's own `success` flag, and `score` is the
library's own 0..1 score. `summary` (`total`/`passed`/`failed`/`pass_rate`/
`avg_score`/`by_grader`) is computed from those carried-through values.

`eval_lib` has no error slot of its own (a failing metric raises out of
`evaluate()`), so `errors={index_or_id: exception_or_dict}` lets you record
test cases that never produced a `TestCaseResult`; they become `Result.error`
(`timeout` / `provider_error` / `runner_error`, classified from the exception
class name) with `passed=False`. `durations_ms={...}` fills
`Result.duration_ms` the same way.

## Mapping

| eval_lib | EvalPort | Notes |
|---|---|---|
| `EvalTestCase.name` | `TestCase.id` | Falls back to positional `tc_0001`; `TestCaseResult` has no name, so pass `test_cases=` (or `ids=`) to `results_to_openeval()` for matching ids |
| `EvalTestCase.input` | `TestCase.input` | 1:1 (string) |
| `EvalTestCase.expected_output` | `TestCase.expected_output` | 1:1; omitted when `None` |
| `EvalTestCase.retrieval_context` / `tools_called` / `expected_tools` | same names on `TestCase` | 1:1 |
| `EvalTestCase.actual_output` | `TestCase.metadata.eval_ai_library.actual_output` | EvalPort keeps outputs on `Result`, not `TestCase` |
| `reasoning`, `execution_trace`, `agent_confidence`, `perturbation_group`, `planning_steps`, `resource_usage`, `extra_fields` | `TestCase.metadata.eval_ai_library.*` | JSON-ified; pydantic sub-models via `model_dump()` |
| metric `name` | `Grader.id = gr_<snake_case(name)>`, `GraderResult.grader_id` | Stable across every input shape |
| `exactMatchMetric` | `exact_match` | Always; `case_sensitive` / `strip_whitespace` kept in `params` when known |
| `semanticSimilarityMetric` | `semantic_similarity` | `params.threshold` = the metric's threshold |
| `regexMatchMetric` with a known `pattern` | `regex` | Only from a metric *object* (or dict carrying `pattern`); otherwise `custom` |
| `jsonSchemaMetric` with a known `schema` | `json_schema` | Same rule |
| `containsMetric` with exactly one keyword, mode `any`/`all` | `contains` | `ignore_case = not case_sensitive`; multi-keyword or mode `none` -> `custom` (keywords preserved in `params.eval_ai_library`) |
| every other metric (all LLM-as-judge, agent, security, reliability, `gEval`, `customEval`, ...) | `custom`, `params.handler = "eval_ai_library:<name>"` | `llm_judge` requires `params.prompt`, which this adapter cannot honestly supply; `evaluation_model` kept in `params.eval_ai_library` |
| `MetricResult.score` (0..1) | `GraderResult.score` | Identity, clamped to `[0, 1]`; `None`/NaN -> `null`; raw value always in `metadata.eval_ai_library.score` |
| `MetricResult.success` | `GraderResult.passed` | Carried through, never recomputed |
| `MetricResult.reason` | `GraderResult.reason` | Omitted when `None` |
| `MetricResult.threshold` / `evaluation_cost` / `evaluation_model` / `evaluation_log` | `GraderResult.metadata.eval_ai_library.*` | Lossless; `evaluation_log.skipped` (the empty-output guard) sets `metadata.eval_ai_library.skipped = true` while keeping the reported `0.0` score |
| `TestCaseResult.success` | `Result.passed` | Carried through |
| `TestCaseResult.actual_output` | `Result.actual_output` | 1:1 |
| `TestCaseResult.input` / `expected_output` / `retrieval_context` / `tools_called` / `expected_tools` | `Result.metadata.eval_ai_library.*` | So a ResultSet is self-describing without its Suite |
| sum of `evaluation_cost` | `ResultSet.metadata.eval_ai_library.total_evaluation_cost` | |

Lossy or unsupported:

- `ConversationalEvalTestCase` / `ConversationalTestCaseResult` (multi-turn
  `dialogue`) are rejected with a clear `ValueError` in this version.
- `from_openeval()` on a suite this adapter did not produce yields
  `actual_output=""` (the field is required by `EvalTestCase`) and joins
  multi-turn `input` arrays with newlines.
- `GraderResult.type` for `regex` / `json_schema` / `contains` can only be
  typed when the metric objects are passed to `results_to_openeval()` too;
  from bare `MetricResult`s those metrics are `custom` in both documents,
  consistently.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 - see LICENSE.
