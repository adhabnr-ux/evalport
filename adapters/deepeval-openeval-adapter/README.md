# deepeval-openeval-adapter

Converts between [DeepEval](https://github.com/confident-ai/deepeval)'s `LLMTestCase` /
`TestResult` / `MetricData` objects and [EvalPort](https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM evaluation test cases, graders, suites, and
results.

```bash
pip install deepeval-openeval-adapter          # adapter only
pip install "deepeval-openeval-adapter[deepeval]"  # + the real deepeval SDK
```

```python
from deepeval.test_case import LLMTestCase
from deepeval_openeval_adapter import to_openeval, test_results_to_openeval

test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        context=["France is a country in Western Europe."],
    ),
]

# Before running: describe the inputs as an EvalPort suite
suite = to_openeval(test_cases, suite_id="geo_quiz", ids=["q1"])

# Run DeepEval's own metrics as usual
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
test_cases[0].actual_output = "Paris is the capital of France."
eval_result = evaluate(test_cases, [AnswerRelevancyMetric()])

# After running: describe the outputs + grades as an EvalPort ResultSet
result_set = test_results_to_openeval(
    eval_result, suite_id="geo_quiz", run_id="run-1",
    started_at="2026-08-22T00:00:00Z", ids=["q1"],
)
```

## Why this exists as a standalone package

See [confident-ai/deepeval#3067](https://github.com/confident-ai/deepeval/issues/3067),
opened after reading `LLMTestCase`, `TestResult`, and `MetricData` directly in DeepEval's
own source (`deepeval/test_case/llm_test_case.py`, `deepeval/evaluate/types.py`,
`deepeval/test_run/api.py` — not the docs). It follows the same "standalone package, zero
footprint on the target framework" shape as the AutoGen, CrewAI, Giskard, and Guardrails
adapters in this ecosystem, rather than presuming a maintainer wants an in-repo module —
DeepEval's team hasn't weighed in yet, and this way there's nothing to revert if they'd
rather it stay external.

## A genuinely close fit

DeepEval's `LLMTestCase` schema (`context`, `retrieval_context`, `tools_called`,
`expected_tools`, `tags`) lines up with EvalPort's `TestCase` schema almost field-for-field —
closer than most adapters in this ecosystem need to reach for, since both were independently
designed around the same shape: RAG context, agent tool-calls, and free-form tags.

| DeepEval (`LLMTestCase`) | EvalPort (`TestCase`) | Notes |
|---|---|---|
| `input` | `input` | direct |
| `expected_output` | `expected_output` | direct |
| `context` | `context` | direct |
| `retrieval_context` | `retrieval_context` | items stringified — see below |
| `tools_called` (`List[ToolCall]`) | `tools_called` (tool *names* only) | EvalPort's schema only carries names here |
| `expected_tools` (`List[ToolCall]`) | `expected_tools` (tool *names* only) | same |
| `tags` | `tags` | direct |
| `actual_output`, `comments`, `token_cost`, `completion_time`, `flaky`, `multimodal`, `name`, full `ToolCall` detail | — | no EvalPort `TestCase` field covers these — preserved under `metadata["deepeval"]` |

On the results side, each `MetricData` in `TestResult.metrics_data` becomes one EvalPort
`GraderResult`:

| DeepEval (`MetricData`) | EvalPort (`GraderResult`) |
|---|---|
| `name` | `grader_id` (slug-normalized, e.g. `"Answer Relevancy"` → `"answer_relevancy"`) |
| `score` | `score` (clamped to `[0, 1]`) |
| `success` | `passed` |
| `reason` | `reason` |
| `threshold`, `strict_mode`, `evaluation_model`, `error`, `evaluation_cost`, `input_tokens`, `output_tokens` | `metadata` |

## Design decisions, documented honestly

**Why every test case references one placeholder `custom` grader in `to_openeval()`, not a
grader per DeepEval metric.** DeepEval doesn't attach specific metrics to an `LLMTestCase` up
front — which of its dozens of metrics (built-in or community) run against a test case is
chosen separately, at `evaluate()` time. Guessing a grader list before that decision is made
would mean inventing information this adapter doesn't have. The suite-side grader is
explicitly labeled a placeholder (`description` field says so); the *real*, per-metric
grading shows up honestly once `test_results_to_openeval()` converts actual `MetricData`.

**Why `tools_called`/`expected_tools` map to plain tool-name strings, not full `ToolCall`
objects.** EvalPort's `TestCase` schema defines these fields as arrays of strings ("names of
tools called"), not objects — verified against `spec/schemas/testcase.json`, not assumed. A
`ToolCall`'s richer detail (`description`, `reasoning`, `output`, `input_parameters`) is
preserved under `metadata["deepeval"]["tools_called_full"]` / `["expected_tools_full"]`
rather than silently dropped, but the EvalPort-native fields only ever carry names, matching
what the schema actually allows.

**Why DeepEval test-case IDs are `name` → `tc_{index}`, not a stable field DeepEval
provides.** `LLMTestCase` has no public unique identifier — only an optional, user-chosen
`name` and a private `_identifier` UUID that, verified by reading `deepeval/evaluate/types.py`,
DeepEval itself does **not** propagate into `TestResult` (which carries `name` and `index`
only). So this adapter's id strategy — explicit `ids[i]` if supplied, else `name`, else
`tc_{i}` — mirrors exactly how DeepEval's own `evaluate()` correlates results back to test
cases: positionally, with `name` as an optional human label. Pass the same `ids` list to both
`to_openeval()` and `test_results_to_openeval()` for guaranteed correlation.

**Why `score` is clamped to `[0, 1]`, not passed through raw.** `MetricData.score` is a plain
`Optional[float]` with no enforced bound in DeepEval's own type — most built-in metrics are
documented as 0–1, but nothing stops a custom or community metric from returning outside that
range. EvalPort's schema requires `score` in `[0, 1]` or `null`. The clamp is a real, visible
information loss for any metric that scores outside that range — not an assumption that all
of them do.

**Why a `TestResult` with empty/`None` `metrics_data` becomes an explicit `runner_error`.**
DeepEval logs a metric failure rather than raising (verified by reading
`deepeval/evaluate/types.py` and the shape `TestResult.metrics_data` allows — `Union[List[MetricData], None]`),
so an empty result here is a real possible outcome, not a bug in this adapter. It's surfaced
as `error: {"type": "runner_error", ...}` in the `ResultSet`, not a silent pass or fail.

**Why `retrieval_context` items are stringified as `"source: context"`.** A
`retrieval_context` entry is either a plain `str` or a `RetrievedContextData`
(`context`, `source`) object. `RetrievedContextData` has its own `model_serializer` that
renders exactly as `f"{source}: {context}"` — this adapter matches that format precisely
(read from `deepeval/test_case/llm_test_case.py`, not guessed), so a value DeepEval itself
prints and a value round-tripped through this adapter read identically.

## What round-trips losslessly, and what doesn't

Round-trips cleanly: `input`, `expected_output`, `context`, `tags`, `name`, `comments`,
`token_cost`, `completion_time`, `flaky`, `multimodal`, full `ToolCall` detail (via
`metadata["deepeval"]["tools_called_full"]`/`["expected_tools_full"]`).

Does **not** round-trip losslessly:
- **`retrieval_context` items lose their `RetrievedContextData.source`/`context` split** once
  stringified — the combined `"source: context"` string comes back as a single string on
  `from_openeval()`, not a reconstructed `RetrievedContextData`.
- **`tools_called`/`expected_tools` lose everything but the tool name** going through
  EvalPort's native fields; `from_openeval()` returns plain name strings — construct real
  `ToolCall(name=...)` objects from them yourself if your `LLMTestCase` needs `ToolCall`
  instances rather than the loosely-typed strings deepeval's own `Union` type also accepts in
  some contexts.
- **Multi-turn `input` (a list of strings) is rejected outright by `from_openeval()`.**
  `LLMTestCase.input` is `str`-only single-turn; DeepEval's multi-turn shape is a separate
  `ConversationalTestCase.turns`, out of scope for this adapter.
- **A DeepEval-side `score` outside `[0, 1]`** is clamped, not preserved raw (see above).

## Testing

40 tests in `tests/test_adapter.py`, all passing against the real, installed
`deepeval==4.1.10` package (`LLMTestCase`, `ToolCall`, `RetrievedContextData`, `TestResult`,
`MetricData`, `EvaluationResult` are imported and constructed from `deepeval.test_case` /
`deepeval.evaluate.types` / `deepeval.test_run.api` directly, not reinvented) and the real
`openeval.validate.validate_suite()` / `validate_result_set()`. Covers: full field mapping,
metadata preservation, explicit vs. auto-generated test case IDs, score clamping at both
bounds, `None` scores, the `success`-is-`None` fallback, multi-metric results, the
empty/`None`-`metrics_data` → `runner_error` path, multimodal `actual_output` lists, the
`EvaluationResult` wrapper, and a full suite → simulated run → `ResultSet` end-to-end
round trip validated against the real spec.

```bash
pip install -e ".[test]"
pip install -e /path/to/evalport/sdk/python   # or: pip install evalport-sdk
pip install deepeval==4.1.10
pytest tests/
```
