# freeplay-openeval-adapter

Convert [Freeplay](https://freeplay.ai/) `Dataset`/`DatasetTestCase` data,
and `TestSuiteRun` per-test-case `eval_results`, to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

## Install

```
pip install "freeplay-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/freeplay-openeval-adapter"
```

Not yet published to PyPI — installs directly from source via pip's
`git+` / `#subdirectory=` support (verified working).

## Usage

### Dataset → EvalPort suite

```python
from freeplay import Freeplay
from freeplay_openeval_adapter import to_openeval

fp = Freeplay(freeplay_api_key="...", api_base="https://your-instance.freeplay.ai/api")
dataset = fp.test_cases.get(project_id="...", dataset_id="my-dataset-id")  # -> DatasetResults

suite = to_openeval(dataset)           # llm_judge grader (default)
# suite = to_openeval(dataset, grader_type="exact_match")  # for clean-string expected answers

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

### EvalPort suite → Freeplay DatasetTestCase objects

```python
from freeplay.resources.test_cases import DatasetTestCase
from freeplay_openeval_adapter import from_openeval

items = from_openeval(suite)
test_cases = [
    DatasetTestCase(
        inputs=item["inputs"],
        output=item["output"],
        metadata=item["metadata"],
        id=item["id"],
    )
    for item in items
]
fp.test_cases.create_many(project_id="...", dataset_id="my-dataset-id", test_cases=test_cases)
```

### TestSuiteRun results → EvalPort ResultSet

**Note:** as of `freeplay` 0.6.0, `TestSuites`/`TestSuiteRun` (defined in
`freeplay/resources/test_suites.py`, and the only place in this SDK version
with a real, typed per-test-case `eval_results` parameter — see "Design
notes" below) isn't wired onto the top-level `Freeplay` client as
`fp.test_suites` the way `fp.test_cases`/`fp.test_runs` are — confirmed by
reading `freeplay/freeplay.py`'s `__init__` directly. It's still fully
usable, constructed from the client's own (non-underscore-prefixed)
`call_support`/`recordings` attributes:

```python
from freeplay.resources.test_suites import TestSuites
from freeplay_openeval_adapter import results_to_openeval

test_suites = TestSuites(fp.call_support, fp.recordings)
run = test_suites.run(project_id="...", suite_id="my-suite-id")

recorded = []
for test_case in run.test_cases:  # prompt-type suite; use run.trace_test_cases for agent-type
    formatted = run.format_prompt(test_case)
    # ... call your LLM with formatted.llm_prompt, get `output` and `all_messages` back ...
    eval_results = {"exact_match": output.strip() == test_case.output, "helpfulness": 0.82}
    run.record(test_case, all_messages, eval_results=eval_results)
    recorded.append({
        "test_case_id": test_case.id,
        "eval_results": eval_results,
        "output": output,
    })

result_set = results_to_openeval(
    suite_id=run.suite_id,
    run_id=run.run_id,
    recorded=recorded,
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

## Design notes

Two translation challenges are handled explicitly, both confirmed against
the real installed `freeplay` package (0.6.0) — `freeplay.resources.test_cases`
and `freeplay.resources.test_suites` — not the docs.

### 1. Named, schema-free `inputs`/`variables` mapping

`DatasetTestCase.inputs` (and `CompletionTestCase.variables`) is a
`Mapping[str, str | int | bool | float | dict | list]` — a schema-free,
named-variable row (the same shape Parea's and Vellum's adapters in this
repo had to flatten), not a single string. EvalPort requires a plain
`input: string | string[]`.

`flatten_inputs()` picks the value of the first key matching a preferred
name (`input`, `question`, `query`, `prompt`, `text`), then falls back to
the first string-valued entry, then to a stable JSON dump of the whole
mapping so nothing is silently dropped when no string field exists. The
original mapping (and any conversation `history`) is always preserved
under `metadata.freeplay.original_inputs` / `.history` so `from_openeval()`
restores it losslessly.

### 2. Results: recording-time `eval_results`, not the retrieved run summary

Freeplay's `TestSuiteRun` has two very differently-shaped result surfaces,
and only one of them is real, typed, per-test-case data:

- **`TestSuiteRun.record()` / `.record_trace()`** take an
  `eval_results: Optional[Dict[str, Union[bool, float]]]` parameter at the
  point you record a completion or trace — one dict per test case, keyed
  by evaluator name. This is real, typed, and already joined to the test
  case being recorded.
- **`TestSuiteRun.get_results() -> TestSuiteRunResults`** — the
  *retrieved* run result — only exposes an aggregate `summary_statistics`
  (`auto_evaluation` / `human_evaluation` / `client_evaluation`, each a
  bare `Dict[str, Any]`) plus a top-level `eval_results: Optional[Dict[str, Any]]`.
  Reading `freeplay/resources/test_suites.py`'s `_parse_run_results()`
  directly confirms both are passed through completely unvalidated, with
  no documented per-test-case breakdown anywhere in the SDK.

`results_to_openeval()` therefore converts the **recording-time** shape —
a list of `{"test_case_id", "eval_results", "output"?, "passed"?}` dicts
that callers accumulate as they iterate a run and call `record()` /
`record_trace()` — rather than inventing a per-test-case join the SDK
doesn't actually provide from the retrieved summary. If a future Freeplay
SDK release adds real per-test-case structure to the retrieved run result,
converting that shape is a natural, separate extension — documented here
rather than guessed at now.

Each `eval_results` entry becomes a `GraderResult` with `type: "custom"`
(Freeplay's evaluator names are free-form, with no vocabulary mapping onto
EvalPort's well-known grader types) and a clamped `score`: booleans map to
the natural `1.0`/`0.0`, and unbounded floats are clamped into EvalPort's
required `[0, 1]` range via `clamp_score()`, with the original raw value
always preserved under the spec's own reserved `metadata.openeval.raw_score`
key. Overall `Result.passed` defaults to *all* graders passing, or can be
overridden per test case via an explicit `"passed"` key.

### 3. Prompt-type vs. agent-type suites

A Freeplay test suite's `target_type` determines whether `TestSuiteRun`
yields `CompletionTestCase` (`.test_cases`, prompt-type) or `TraceTestCase`
(`.trace_test_cases`, agent-type) — the SDK itself enforces you can't mix
them. `results_to_openeval()` is agnostic to which one produced a given
`test_case_id`, since it only needs the id string, not the test case
object — the same `record()`/`record_trace()` → `recorded` list pattern
works for both suite types.

### 4. `TestSuites` isn't wired onto the top-level client (as of 0.6.0)

Worth flagging plainly, since it affects how the results-side example above
has to be written: `freeplay/freeplay.py`'s `Freeplay.__init__` wires up
`self.test_cases` (`TestCases`), `self.test_runs` (`TestRuns` — a simpler,
older resource with only `.create()`/`.get()` and no `.record()` at all),
`self.recordings`, `self.sessions`, `self.prompts`, `self.traces`,
`self.customer_feedback`, and `self.metadata` — but not `self.test_suites`.
`TestSuites`/`TestSuiteRun`, defined in `freeplay/resources/test_suites.py`
and re-exported nowhere in `freeplay/__init__.py` either, is the *only*
place in this SDK version with a real per-test-case `eval_results` — so
this adapter's results-side function is necessarily built against it, with
the construction workaround shown above (`TestSuites(fp.call_support, fp.recordings)`)
rather than an `fp.test_suites` attribute that doesn't currently exist.

## Spec

See the full EvalPort specification at
<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>

## License

Apache 2.0 — see [LICENSE](LICENSE).
