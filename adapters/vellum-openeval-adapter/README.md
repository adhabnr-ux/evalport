# vellum-openeval-adapter

Convert [Vellum](https://www.vellum.ai/) Test Suite data to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

## Install

```
pip install "vellum-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/vellum-openeval-adapter"
```

Not yet published to PyPI — installs directly from source via pip's
`git+` / `#subdirectory=` support (verified working).

## Usage

### Test Suite → EvalPort suite

```python
import vellum
from vellum_openeval_adapter import to_openeval

client = vellum.Vellum(api_key="...")
page = client.test_suites.list_test_suite_test_cases(id="my-test-suite-id")

suite = to_openeval(page.results, id="my-test-suite-id")  # llm_judge grader (default)
# suite = to_openeval(page.results, grader_type="exact_match")  # for clean-string expected answers

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

### EvalPort suite → Vellum test cases

```python
from vellum_openeval_adapter import from_openeval

items = from_openeval(suite)
for item in items:
    # input_values/evaluation_values are plain dicts shaped like
    # NamedTestCase*VariableValueRequest ({"name", "type", "value"}) --
    # Vellum's Fern-generated client converts them via
    # convert_and_respect_annotation_metadata(), so raw dicts work directly,
    # no need to construct the typed request objects by hand.
    client.test_suites.upsert_test_suite_test_case(
        "my-test-suite-id",  # positional: this parameter is named `id_` (not `id`) to avoid shadowing the builtin
        input_values=item["input_values"],
        evaluation_values=item["evaluation_values"],
        label=item.get("label"),
    )
```

### Test Suite Run executions → EvalPort ResultSet

```python
from vellum_openeval_adapter import results_to_openeval

run = client.test_suite_runs.retrieve(id="my-run-id")
executions = client.test_suite_runs.list_executions(id="my-run-id")

result_set = results_to_openeval(
    executions,          # PaginatedTestSuiteRunExecutionList — .results is read automatically
    suite_id="my-test-suite-id",
    run_id=run.id,
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

## Design notes

Two translation challenges are handled explicitly:

### 1. Named, typed test-case variables

Vellum's `TestSuiteTestCase.input_values` / `.evaluation_values` are each a
`List[TestCase*VariableValue]` — a typed, *named* variable system
(`STRING`, `NUMBER`, `JSON`, `CHAT_HISTORY`, `ARRAY`, and more), not a flat
dict or a single string. EvalPort requires a plain `input: string | string[]`.

`variables_to_input()` handles this per variable rather than picking one and
dropping the rest: a single `STRING` variable becomes a plain string (the
common case — one prompt variable), and everything else (multiple
variables, or a single non-string variable) becomes an array of
`"{name}: {value}"` entries, preserving every named variable. Non-string
types (`JSON`, `ARRAY`, `CHAT_HISTORY`, ...) are stringified honestly —
`CHAT_HISTORY` renders as `role: text` lines, everything else as JSON.

The original named-variable list is always preserved under
`metadata.vellum.original_input_values` / `original_evaluation_values` so
`from_openeval()` can restore it losslessly.

### 2. Typed metric-output union on results

`TestSuiteRunExecutionMetricResult.outputs` is a typed union
(`TestSuiteRunMetricStringOutput` / `NumberOutput` / `JsonOutput` /
`ErrorOutput` / `ArrayOutput`). Only the `NUMBER` variant is a real
EvalPort `GraderResult.score` (which the spec requires to be `null` or in
`[0, 1]`) — `map_metric_output()` clamps it into range and preserves the
raw value under the spec's own reserved `metadata.openeval.raw_score` key.
Every other output type gets an honest `score: null`, never a fabricated
number.

`passed` has no universal semantic Vellum hands you directly (unlike a
numeric score), so it's derived with a documented, overridable heuristic:
`score >= pass_threshold` (default `0.5`) when a numeric score exists,
otherwise `False` only if the metric reported an `ERROR` output, `True`
otherwise. The full raw output list is preserved under `metadata.vellum`
so a caller who needs different pass/fail semantics can re-derive them.

`grader_id` uses `metric_label` when present (falling back to `metric_id`),
and `type` is set to `"custom"` — Vellum's `TestSuiteRunExecutionMetricDefinition`
doesn't expose a grader-type vocabulary that maps onto EvalPort's
well-known types (`exact_match`, `semantic_similarity`, `llm_judge`, ...),
so claiming one would be a guess.

### 3. Suite-level graders

Vellum test cases don't carry a grader/metric definition of their own —
metrics are configured on the Test Suite separately from its test cases,
invisible on `TestSuiteTestCase`. `to_openeval()` generates a single
default grader (`llm_judge` unless overridden via `grader_type`) that every
test case references, the same approach `literalai-openeval-adapter` takes
for the same reason.

## Spec

See the full EvalPort specification at
<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>

## License

Apache 2.0 — see [LICENSE](LICENSE).
