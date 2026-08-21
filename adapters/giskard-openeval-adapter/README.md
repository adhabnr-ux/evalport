# giskard-openeval-adapter

Convert [Giskard](https://github.com/Giskard-AI/giskard-oss) `giskard-checks` Suites, Scenarios, and SuiteResults to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "giskard-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/giskard-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

This installs `evalport-sdk` as its only hard dependency. `giskard-checks` itself is not installed automatically -- it's still pre-1.0 (currently `1.0.2rc1`, a release candidate; PyPI has published `1.0.1a1` through `1.0.2rc1` so far, no stable `1.0.x` yet) and requires Python >=3.12, so it's an opt-in extra rather than a hard dependency:

```bash
pip install "giskard-openeval-adapter[giskard] @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/giskard-openeval-adapter"
```

This was a git-source-only install as recently as this adapter's last README update -- `giskard-checks` has since been published to PyPI (still pre-release, hence the extra needing `giskard-checks>=1.0.2rc1` rather than a plain `>=1.0.0` to opt pip into resolving a pre-release version at all). If you already have an older git-source install, note that `1.0.2rc1`'s public API **renamed `key` to `target_key`** on every comparison check (`Equals`, `NotEquals`, `GreaterThan`, `LessThan`, `GreaterThanEquals`, `LessThanEquals`) and dropped a `text_key` parameter this adapter had (incorrectly) relied on for `StringMatching` -- this adapter's `to_openeval()`/`from_openeval()` were updated to match the real published API and re-verified against it (42/42 tests passing), so upgrading `giskard-checks` to `1.0.2rc1`+ alongside this adapter is safe; upgrading only `giskard-checks` while pinned to an older version of this adapter is not.

## Usage

### Suite definition → EvalPort suite, and back

```python
from giskard.checks import Scenario, Suite, Equals, StringMatching

scenario = (
    Scenario("geo_fact")
    .interact("What is the capital of France?", outputs="Paris is the capital of France.")
    .check(Equals(target_key="trace.last.outputs", expected_value="Paris is the capital of France."))
)
suite = Suite(name="geo_quiz")
suite.append(scenario)

from giskard_openeval_adapter import to_openeval, from_openeval

eval_suite = to_openeval(suite, suite_id="geo_quiz")

from openeval.validate import validate_suite
assert validate_suite(eval_suite).valid

# ...and back: rebuild runnable Scenarios from an EvalPort suite document.
# Every interact()'s outputs are left unbound (MISSING) -- bind your system
# under test before running.
scenarios = from_openeval(eval_suite)
new_suite = Suite(name="geo_quiz")
for s in scenarios:
    s.with_target(lambda inputs, trace=None: my_app(inputs))
    new_suite.append(s)

result = await new_suite.run()
```

`to_openeval()` reads scenario/step/check *definitions* -- it's meant to be called on a `Suite` you've built but not necessarily run yet, the same shape as the `Scenario(...).interact(...).check(...)` fluent API examples in `giskard-checks`' own README. Only steps whose interacts all have static (already-resolved) string inputs are exported; a step driven by a callable, generator, or `InputGenerator` has no fixed value to serialize and is silently skipped rather than exporting a fabricated one. A step with interacts but no checks is also skipped, since every EvalPort `TestCase` requires at least one grader.

A `Scenario` with more than one step (i.e. `.interact().check().interact().check()`, a genuine multi-turn conversation with an intermediate check) exports as multiple EvalPort `TestCase`s, one per step, with ids `"{scenario_name}::step_{i}"`. A single step with more than one `interact()` call before its `check()` (no intermediate check) exports as *one* `TestCase` whose `input` is the array of each turn's text -- EvalPort's `input` field is exactly `string | array of strings` for this reason, and `from_openeval()` reverses this precisely: an array `input` becomes that many `interact()` calls inside a single step.

### Executed results → EvalPort ResultSet

```python
from giskard_openeval_adapter import suite_result_to_openeval

result = await suite.run()

result_set = suite_result_to_openeval(
    result,
    suite_id="geo_quiz",
    run_id="run-2026-08-15",
    started_at="2026-08-15T00:00:00Z",
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Giskard's four-state `CheckStatus` (`PASS`/`FAIL`/`ERROR`/`SKIP`) is treated the way `giskard-checks` itself insists it be treated (see `result.py`'s module docstring: *"ERROR and SKIP mean no verdict was reached... keep the four states distinct"*) -- `PASS` becomes `score: 1.0, passed: true`; `FAIL` becomes `score: 0.0, passed: false`; both `ERROR` and `SKIP` become `score: null, passed: false`, since EvalPort's `GraderResult.score` is explicitly nullable for exactly this "no verdict reached" case, rather than collapsing an errored check into a falsely-confident `0.0`.

## Grader / Check mapping

| EvalPort grader `type` | Giskard `Check` | Direction |
|---|---|---|
| `exact_match` | `Equals(target_key="trace.last.outputs", ...)` | both |
| `contains` | `StringMatching` | both |
| `regex` | `RegexMatching` | both |
| `semantic_similarity` | `SemanticSimilarity` | both |
| `llm_judge` | `LLMJudge` | both |
| `json_schema` | `JsonValid(schema=...)` | both |
| `json_path` | `Equals`/`NotEquals`/`GreaterThan`/`LessThan`/`GreaterThanEquals`/`LessThanEquals` keyed on a sub-path of `trace.last.outputs`, or `StringMatching` for the `contains` operator | both |
| `custom` | any other check kind (`AllOf`, `AnyOf`, `Not`, `FnCheck`, `Readability`, `RegoPolicy`, or a comparison check keyed on something other than the last output) | export only |
| `code`, `human`, `"model graded"`, unrecognized types | -- | clean-skipped on import |

A giskard `Check` this adapter can't map onto a native EvalPort grader type is exported as a `"custom"` grader (`params.handler` names the giskard check kind) with the check's **full definition preserved verbatim** under `grader.metadata.giskard.check` -- nothing is silently dropped, even though this adapter doesn't attempt to reconstruct it back into a real `Check` on import (a `"custom"` grader is one of the types EvalPort's own clean-skip convention exists for). EvalPort grader types with no `giskard-checks` equivalent (`code`, `human`, `"model graded"`, and any inline grader object of an unrecognized type) are clean-skipped on import the same way `openeval run` skips a grader type it doesn't know how to execute -- the resulting `Scenario` is still returned, just with fewer checks than graders in the source `TestCase`, so callers can inspect that rather than the whole conversion silently failing.

## What round-trips losslessly, and what doesn't

Deterministic checks (`Equals`, `StringMatching`, `RegexMatching`, comparisons) round-trip cleanly in both directions -- the EvalPort grader's `params` capture everything the giskard `Check` needs to reconstruct.

Two spots are honestly lossy or approximate, and it's worth knowing about them rather than discovering them by surprise:

- **`llm_judge` prompts.** Giskard's `LLMJudge` prompts use Jinja2 templating (`{{ trace.last.outputs }}`, `{{ trace.last.inputs }}`) to reference trace data. EvalPort's `llm_judge` grader instead requires the literal substitution tokens `{output}`, `{input}`, or `{expected}` somewhere in `params.prompt` (enforced by `validate_suite()`, not just documented). If a giskard prompt doesn't already contain one of those tokens, `to_openeval()` appends `"\n\nResponse to evaluate: {output}"` to it -- the original prompt text is never rewritten, only appended to, and the original is *also* preserved verbatim under `grader.metadata.giskard.check.prompt` regardless. Similarly, EvalPort's `llm_judge` grader requires a `params.model` string, but giskard's `LLMJudge` has no per-check model field (the model is configured on the generator, globally or per-run) -- `to_openeval()` fills in the placeholder `"giskard-default"`, and `from_openeval()` ignores `params.model` entirely when building a `LLMJudge` back.
- **`json_path` translation.** EvalPort's `json_path` grader uses a generic JSONPath (`"$.field.nested"`); giskard's comparison checks use their own `JSONPathStr` rooted at `"trace.last.outputs"`. This adapter translates the common case -- plain dotted field access -- by swapping roots (`"$.count"` <-> `"trace.last.outputs.count"`). Wildcards, filters, and slice expressions pass through after the root swap, but aren't independently verified to translate correctly; giskard's own JSONPath validator (via `jsonpath_ng`) will raise a clear error at `Check` construction time if the result isn't valid syntax, rather than this adapter silently mistranslating it. A comparison check keyed on anything *other* than `trace.last.outputs` (e.g. `trace.last.inputs`, `trace.annotations`) has no EvalPort representation at all -- `json_path` graders only ever describe `actual_output` -- and exports as a `"custom"` grader instead.

Everything else -- the full original check definition (every field, not just the ones EvalPort's grader schema has room for), the scenario name, scenario-level annotations and tags, and giskard's per-check-result `details`/`metrics` -- is preserved under the `metadata.giskard.*` namespace on export (`openeval.*` is the only reserved metadata prefix), the same pattern every adapter in this repository uses (see the [UpTrain adapter](../uptrain-openeval-adapter) or [Langfuse adapter](../langfuse-openeval-adapter) for the same convention with their own framework-specific fields).

## Running the tests

```bash
cd adapters/giskard-openeval-adapter
python3.12 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest tests/ -v
```

42 tests, all running against the real, now-published `giskard-checks==1.0.2rc1` classes (`Scenario`, `Suite`, `Equals`, `StringMatching`, `RegexMatching`, `SemanticSimilarity`, `LLMJudge`, `JsonValid`, the comparison checks, `SuiteResult`/`ScenarioResult`/`TestCaseResult`/`CheckResult`, including two tests that `await suite.run()` for real against deterministic checks) and the real `openeval.validate.validate_suite()`/`validate_result_set()` -- not mocks.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
