# athina-openeval-adapter

Converts between [Athina](https://github.com/athina-ai/athina-evals)'s LLM-evaluator
data model and [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation test cases, graders, suites, and results.

```bash
pip install athina-openeval-adapter          # adapter only
pip install "athina-openeval-adapter[athina]"  # + the real athina SDK
```

```python
from athina.loaders.loader import DataPoint
from athina.evals import DoesResponseAnswerQuery
from athina_openeval_adapter import to_openeval, result_to_openeval, from_openeval

data = [
    DataPoint(query="What is the capital of France?", response="Paris is the capital of France."),
    DataPoint(query="How do I reset my password?", response="The weather today is sunny."),
]

# Before running: describe the inputs as an EvalPort suite
suite = to_openeval(data, "does_response_answer_query")

# Run the real evaluator (requires ATHINA_API_KEY / OPENAI_API_KEY)
evaluator = DoesResponseAnswerQuery()
eval_results = evaluator.run_batch(data)

# After running: describe the outputs + grades as an EvalPort ResultSet
result_set = result_to_openeval(
    data, eval_results, "does_response_answer_query",
    suite_id=suite["id"], run_id="run-1", started_at="2026-08-22T00:00:00Z",
)
```

## What this adapter covers, and why it's scoped that way

Athina ships several eval families under `athina.evals`: LLM evals, function evals
(regex/contains/PII/...), Ragas-wrapping evals, safety evals, and conversation evals.
Reading the **installed, real** `athina==1.7.39` package's source (not the docs, not the
GitHub `main` branch — see the note below on why that distinction mattered) showed that
only the four evaluators actually exported from `athina.evals.__all__` share one verified
contract:

| Evaluator | `required_args()` (verified) |
|---|---|
| `DoesResponseAnswerQuery` | `["query", "response"]` |
| `ContextContainsEnoughInformation` | `["query", "context"]` |
| `Faithfulness` | `["context", "response"]` |
| `CustomGrader` | `["response"]` |

All four are built on the same `LlmEvaluator` base class, whose `.run()` /
`.run_batch()` return `LlmEvalResult` — a `TypedDict` with exactly
`{name, data, failure, reason, runtime, model}`. That shared, verified contract is what
this adapter converts. Athina's function/Ragas/safety/conversation eval families were
not found to share this same result shape in the installed package (different base
classes were referenced from `athina.evals.function`, `athina.evals.ragas`, etc.), and
rather than guess a mapping for surface this adapter never actually read and tested
against, it stops at the boundary of what it verified. A follow-up adapter module can
extend coverage once someone reads and tests those shapes directly.

### A note on what's real vs. what's aspirational in Athina's own repo

Athina's GitHub `main` branch contains source for a apparently-unreleased, more unified
architecture (`BaseEvaluator`, a single `EvalResult` TypedDict with a `metrics` list and
`display_name`, a `BatchRunResult` dataclass wrapper) that does **not** match what
`pip install athina-evals` actually installs today (`athina==1.7.39`, verified by
installing it in a clean venv and reading its installed source directly). This adapter
is built and tested against the real, installable, currently-shipping version — the one
an actual `pip install athina-evals` user gets — not the in-progress `main` branch. If/when
that refactor ships to PyPI, this adapter's `_KNOWN_INPUT_KEYS` / `LlmEvalResult` field
list will need a version-gated update; that's flagged here explicitly rather than left
as a silent trap for whoever hits it first.

## Design decisions, documented honestly

**Why the suite-level grader type is `custom`, not a made-up Athina-specific type.**
EvalPort's grader schema only standardizes params validation for a fixed list of types
(`exact_match`, `contains`, `regex`, `semantic_similarity`, `llm_judge`, `json_schema`,
`json_path`, `code`, `human`, `custom`); anything else — including a hypothetical
`athina_faithfulness` — is validated exactly like `custom` (`params.handler` required)
anyway. Using the standard `custom` type with `params.handler` set to the real Athina
evaluator name (`"faithfulness"`, `"does_response_answer_query"`, ...) is honest about
what this is: a third-party framework's own eval, not one of EvalPort's built-in grader
types, without inventing a type string a generic EvalPort runner wouldn't recognize.

**Why `score` is a booleanized 0/1, not a continuous confidence value.** `LlmEvalResult`
carries only `failure: bool` and a natural-language `reason` — verified against the
installed source, there is no separate numeric confidence field the underlying LLM judge
exposes for these four evaluators. `result_to_openeval()` reports `1.0` for a pass and
`0.0` for a fail. That's a real, documented information loss versus a grader that does
expose a graded score (Ragas's 0–1 metrics, for instance) — not a bug here, a fact about
what Athina's LLM evaluators actually return.

**Why entries with no `query` use `response` as the EvalPort `input`, flagged in metadata.**
EvalPort's `TestCase.input` is required and non-empty. `CustomGrader`'s `required_args()`
is just `["response"]` — there is no query at all for that evaluator, and forcing one
would mean fabricating data. `to_openeval()` falls back to using `response` as the input
and sets `metadata["athina.input_synthesized_from_response"] = True` so a consumer can
tell a real query from a borrowed one, rather than silently blurring the two.

**Why `from_openeval()` never returns a `response` key.** In EvalPort's model, the
generated output belongs to `ResultSet.Result.actual_output` — a suite's `TestCase` never
carries one, because a suite alone hasn't been run yet. `from_openeval()` reconstructs
only the input side (`query` / `context` / `expected_response`) and expects the caller to
add `response` after generating (or re-generating) it, then pass the completed dict to
`evaluator.run_batch()`. This is the real, honest shape of the cross-tool boundary, not
an oversight — see the module docstring for the full intended round-trip flow.

## What round-trips losslessly, and what doesn't

Round-trips cleanly: `query` / `response` / `context` / `expected_response`, any
evaluator-specific extra kwargs (e.g. `CustomGrader`'s `grading_criteria`, preserved
under `metadata["athina.extra_args"]`), `failure` → `passed`/`score`, `reason`, `model`,
`runtime` → `duration_ms`.

Does **not** round-trip losslessly:
- **EvalPort `context` is a list of strings; Athina's context-taking evaluators
  (`Faithfulness`, `ContextContainsEnoughInformation`) take a single string.**
  `to_openeval()` wraps the single string in a one-item list (lossless);
  `from_openeval()` joins a multi-item list back with `"\n\n"` (lossless as characters,
  but the original list *boundaries* aren't recoverable from the joined string alone if
  a caller round-trips a suite that wasn't produced by this adapter in the first place).
- **Multi-turn `input` (a list of strings) is rejected outright by `from_openeval()`.**
  None of the four covered evaluators has a concept of multi-turn conversation input —
  converting one would mean silently flattening a real structural difference into
  something wrong, so this raises instead.
- **A `None` entry in `eval_results`** (which athina's own `_run_batch_generator` yields
  when a row raises during evaluation, logged rather than re-raised) becomes an explicit
  `error: {type: "runner_error", ...}` in the `ResultSet`, not a silent pass or fail.

## Testing

38 tests in `tests/test_adapter.py`, all passing against the real, installed
`athina==1.7.39` package (`DataPoint` and `LlmEvalResult` are imported and constructed
from `athina.loaders.loader` / `athina.interfaces.result` directly, not reinvented) and
the real `openeval.validate.validate_suite()` / `validate_result_set()`. Every
`required_args()` combination above is exercised with data shaped exactly like what that
evaluator would actually receive.

```bash
pip install -e ".[test]"
pip install -e /path/to/evalport/sdk/python   # or: pip install evalport-sdk
pip install athina==1.7.39
pytest tests/
```
