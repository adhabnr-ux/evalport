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

Going the other way, `from_openeval()` returns `LLMTestCase` constructor kwargs, and the
suite's `exact_match` graders can run as real DeepEval metrics (opt-in):

```python
from deepeval_openeval_adapter import from_openeval, graders_to_deepeval_metrics

cases = [LLMTestCase(**kw, actual_output=my_app(kw["input"])) for kw in from_openeval(suite)]
metrics = graders_to_deepeval_metrics(suite, skip_unsupported=True)  # {grader_id: metric}
evaluate(cases, list(metrics.values()))
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
| `metadata` | `metadata` | key for key; the `"deepeval"` key is reserved for this adapter (a user key of that name is parked inside it and restored) |
| `actual_output`, `comments`, `token_cost`, `completion_time`, `flaky`, `multimodal`, `name`, full `ToolCall` detail | — | no EvalPort `TestCase` field covers these — preserved under `metadata["deepeval"]` |

Empty lists are kept as empty lists in both directions. That matters for `expected_tools=[]`,
which asserts "no tool should be called" — a different claim from an absent `expected_tools`
("no expectation").

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
what the schema actually allows. On the way back, `from_openeval()` wraps the names in
`deepeval.test_case.ToolCall` (deepeval 4.x's `LLMTestCase` rejects bare strings with a
`TypeError`), restoring the full detail from `metadata["deepeval"]` when it is there and still
matches the names. `tool_calls="names"` returns plain strings instead (the 0.1.x shape); wrap
those with `to_tool_calls()`. Without deepeval installed, the default (`"auto"`) returns
strings.

**Why `TestCase.metadata` maps onto `LLMTestCase.metadata` key for key.** `to_openeval()`
has always written `LLMTestCase.metadata` into `TestCase.metadata` at the top level, next to
the adapter's own `"deepeval"` key, so the inverse is the same mapping: every key except
`"deepeval"` goes into `LLMTestCase.metadata`. A metric then sees suite metadata where
DeepEval users expect it (e.g. TruthfulQA's `truthfulqa_all_correct` list of accepted answers).
DeepEval 4.2.6's `LLMTestCase.metadata` is a plain `Optional[Dict]` (`additional_metadata` is a
deprecated alias for it), so no nesting is needed to fit.

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

**Why a `None` score always has `passed: false`.** EvalPort Validation Rule 6 says a
`score: null` means "not verified" and MUST carry `passed: false`, and a `Result` whose graders
are all null-scored MUST be `passed: false` too. When DeepEval reports `success: True` with a
`None` score, the grader is exported with `passed: false` and DeepEval's own verdict is kept as
`metadata.native_success` (on the `GraderResult`, or on `Result.metadata.deepeval` for the
whole test case).

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

DeepEval → EvalPort → DeepEval round-trips cleanly: `input`, `expected_output`, `context`,
`tags`, `metadata`, `name`, `comments`, `token_cost`, `completion_time`, `flaky`,
`multimodal`, `tools_called`/`expected_tools` as `ToolCall` objects with their full detail
(via `metadata["deepeval"]["tools_called_full"]`/`["expected_tools_full"]`), and empty lists.

EvalPort → DeepEval → EvalPort: every `TestCase` data field comes back byte-identical (`id`,
`input`, `expected_output`, `context`, `retrieval_context` strings, `tools_called`,
`expected_tools` including `[]`, `tags`, `metadata`). `to_openeval()` adds its own
`metadata["deepeval"]` key (`name`, and the random `identifier` UUID each `LLMTestCase`
instance generates). `examples/interop/1_dataset_portability.py` checks this against real
benchmark suites.

Does **not** round-trip losslessly:
- **Graders.** A list of `LLMTestCase` has no slot for grader definitions, so `to_openeval()`
  writes one placeholder `custom` grader (see above). For `exact_match`, run the suite's own
  grader in DeepEval with `graders_to_deepeval_metrics()` (below).
- **Suite-level fields** (`name`, `description`, `config`, `metadata`, `version`): a list of
  `LLMTestCase` can't hold them; `to_openeval()` stamps its own name and version.
- **`retrieval_context` items lose their `RetrievedContextData.source`/`context` split** once
  stringified — the combined `"source: context"` string comes back as a single string on
  `from_openeval()`, not a reconstructed `RetrievedContextData`.
- **`tools_called`/`expected_tools` from a suite this adapter didn't write** carry names only,
  so `from_openeval()` builds `ToolCall(name=...)` with no other detail.
- **Multi-turn `input` (a list of strings) is rejected outright by `from_openeval()`.**
  `LLMTestCase.input` is `str`-only single-turn; DeepEval's multi-turn shape is a separate
  `ConversationalTestCase.turns`, out of scope for this adapter.
- **A DeepEval-side `score` outside `[0, 1]`** is clamped, not preserved raw (see above).

## Grader mapping (opt-in): `graders_to_deepeval_metrics()`

`graders_to_deepeval_metrics(suite)` returns `{grader_id: metric}` for every grader in the
suite that has a faithful DeepEval equivalent — one that computes exactly what the EvalPort
grader defines. Pass `skip_unsupported=True` to leave the others out instead of raising.

| EvalPort grader | DeepEval metric | Faithful? |
|---|---|---|
| `exact_match` (`ignore_case`, `trim_whitespace`) | `EvalPortExactMatchMetric`, a `deepeval.metrics.BaseMetric` subclass named after the grader id | yes: trims when `trim_whitespace` (default true), lowercases when `ignore_case` (default false), then `==`, the same steps as the reference runner's `gradeExactMatch` (`cli/src/run/graders/tier1.ts`). Score 1.0/0.0, threshold 1.0 |
| `semantic_similarity` | — | no: the score depends on the embedding model the runner picks, and DeepEval has no deterministic embedding-cosine metric to pin it to |
| `llm_judge` / `model graded` | — | no: DeepEval's `GEval` builds its own evaluation prompt around your criteria, so it would not run the grader's `prompt` verbatim |
| `contains`, `regex`, `json_schema`, `json_path`, `code`, `human`, `custom` | — | not mapped: no DeepEval built-in metric computes exactly these |

Why not DeepEval's own `ExactMatchMetric`: it always strips and is always case-sensitive, so it
can't express `ignore_case: true` or `trim_whitespace: false`. With the spec defaults the two
agree.

Details, so nothing is assumed:
- Python's `str.strip()`/`str.lower()` stand in for JavaScript's `trim()`/`toLowerCase()`.
  Their whitespace sets differ on exactly six code points: U+FEFF is trimmed by JS only;
  U+001C–U+001F and U+0085 are stripped by Python only.
- A missing `expected_output` compares as `""`, as in the reference runner.
- Params the spec doesn't define (e.g. `strip`, used by `benchmarks/gsm8k`) raise a
  `UserWarning` and are ignored, as the reference runner ignores them.
- The metric's name is the grader id, so `test_results_to_openeval()` reports
  `grader_id: "gr_exact_match"` (not `"exact_match"`), matching the suite. Its `type` in the
  ResultSet stays `"custom"`, like every DeepEval metric result.
- `deepeval.evaluate()` runs every metric on every test case. To honour per-test-case
  `graders` lists, group the test cases by the grader ids they reference.

`exact_match()` (the comparison itself, no deepeval needed) is exported too.

## Changelog

### 0.2.0

Fixes found by the cross-framework demos in `examples/interop/`:

- `from_openeval()` now sets `LLMTestCase.metadata` from `TestCase.metadata` (minus the
  adapter's `"deepeval"` key). Previously it dropped it, losing e.g. TruthfulQA's accepted
  answers.
- `to_openeval()` keeps empty lists (`expected_tools: []`, "no tool should be called") instead
  of dropping them. It now emits every list field that is set, including empty
  `context`/`retrieval_context`/`tags`/`tools_called`.
- **Behavior change:** `from_openeval()` returns `deepeval.test_case.ToolCall` objects for
  `tools_called`/`expected_tools` when deepeval is importable (new `tool_calls="auto"`
  default), restoring full `ToolCall` detail recorded by `to_openeval()`, so
  `LLMTestCase(**kwargs)` works. Its output is no longer plain JSON when tools are present. Pass
  `tool_calls="names"` for the 0.1.x return shape; `to_tool_calls()` wraps names later.
- `to_openeval()` parks a user `LLMTestCase.metadata["deepeval"]` key inside the adapter
  namespace instead of letting the namespace overwrite it.
- New opt-in `graders_to_deepeval_metrics()` and `exact_match()` (see above).

### 0.1.0

Initial release.

## Testing

72 tests in `tests/test_adapter.py`, all passing against the real, installed
`deepeval==4.2.6` package (`LLMTestCase`, `ToolCall`, `RetrievedContextData`, `TestResult`,
`MetricData`, `EvaluationResult`, `BaseMetric` are imported and constructed from
`deepeval.test_case` / `deepeval.evaluate.types` / `deepeval.test_run.api` /
`deepeval.metrics` directly, not reinvented) and the real
`openeval.validate.validate_suite()` / `validate_result_set()`. Without deepeval installed
(CI's min mode), the framework-free tests still run and the rest skip. Covers: full field
mapping, metadata preservation in both directions, empty vs absent tool lists, `ToolCall`
reconstruction (and the `names` mode), explicit vs. auto-generated test case IDs, score
clamping at both bounds, `None` scores, the `success`-is-`None` fallback, multi-metric
results, the empty/`None`-`metrics_data` → `runner_error` path, multimodal `actual_output`
lists, the `EvaluationResult` wrapper, the `exact_match` grader mapping, and a full suite →
simulated run → `ResultSet` end-to-end round trip validated against the real spec.

```bash
pip install -e ".[test]"
pip install -e /path/to/evalport/sdk/python   # or: pip install evalport-sdk
pip install deepeval==4.2.6
pytest tests/
```
