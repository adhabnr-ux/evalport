# daimax-openeval-adapter

Convert [daimax-appbench](https://github.com/open-daimax/daimax-appbench)
(an open-source evaluation framework for AI app generators, Python package
`evalapp`) evaluation runs and benchmark items to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

Filed and built following
[open-daimax/daimax-appbench#9](https://github.com/open-daimax/daimax-appbench/issues/9),
where the daimax maintainer asked for this as a standalone adapter living in
the EvalPort repo rather than inside `evalapp`, and gave the constraints that
the "Mapping" section below quotes and follows.

## Why a standalone package?

The maintainer's own words in #9: "this adapter is very welcome. Your 'no
changes to `evalapp` core, standalone package' approach is the right fit for
us." `evalapp` is a full CLI + runner (ai-ui-test / Midscene E2E execution,
report UI, dataset tooling), and its result models are ordinary pydantic
models, so there is nothing to gain by hooking an export into the core. This
package reads those models' real field sets (`EvalRun`, `PromptResult`,
`TestCaseResult`, `SuccessRateMetrics`, `QualityMetrics`,
`ExperienceMetrics`, `EvalSample`, `TestCase` -- taken from the upstream
source, not guessed) and is duck-typed: it accepts the pydantic objects, or
plain dicts shaped like `model_dump()` / the `results.json` daimax writes, so
it never needs to import `evalapp` at runtime.

## Install

```
pip install "daimax-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/daimax-openeval-adapter"
```

Not yet published to PyPI -- this installs directly from source via pip's
`git+`/`#subdirectory=` support.

`evalapp` itself is not on PyPI either. If you want the real upstream models
alongside the adapter (they are optional), the `daimax` extra installs them
from the upstream repo:

```
pip install "daimax-openeval-adapter[daimax] @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/daimax-openeval-adapter"
```

## Usage

### EvalRun -> EvalPort ResultSet (primary, per #9)

```python
import json
from daimax_openeval_adapter import run_to_openeval, run_from_openeval

# An evalapp EvalRun (pydantic model) -- or the dict from its results.json:
eval_run = json.load(open("results/run_abc123/results.json"))

result_set = run_to_openeval(
    eval_run,
    suite_id="appbench_v2",     # default: EvalRun.sample_source
    # pass_threshold=None      # default: passed comes only from daimax's
    #                          # own booleans; see "Mapping" before setting one
    # test_cases=[...],        # optional daimax TestCase objects, to enrich
    #                          # E2E grader results with priority/name/category
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

prompt_results = run_from_openeval(result_set)  # PromptResult-shaped dicts
```

Each daimax `PromptResult` becomes one EvalPort `Result` whose
`grader_results` hold exactly three synthetic metric graders
(`gr_daimax_success_rate`, `gr_daimax_quality`, `gr_daimax_experience`, one
per top-level daimax metric, each carrying its own metric's full breakdown in
`metadata.daimax.breakdown`) plus one `gr_daimax_e2e` grader result per
`TestCaseResult` from the independent ai-ui-test / Midscene engine.

If you want an explicit cutoff (for example the 70-point figure from the
original sketch), opt into it and it is written down as adapter policy:

```python
result_set = run_to_openeval(eval_run, pass_threshold=70)  # daimax 0..100 scale
result_set["metadata"]["daimax"]["adapter_policy"]
# {'name': 'composite_threshold', 'pass_threshold': 70.0,
#  'pass_threshold_scale': '0..100 (daimax composite_score scale)',
#  'pass_threshold_normalized': 0.7, 'description': '... not a daimax-wide pass criterion.'}
```

### EvalSample / TestCase -> EvalPort Suite (bonus scope)

```python
from daimax_openeval_adapter import to_openeval, from_openeval

samples = [  # evalapp EvalSample objects, or dicts shaped like model_dump()
    {"sample_id": "s_todo_001", "requirement": "Build a todo app with add/complete/delete.",
     "app_type": "tool", "complexity": "medium", "requires_backend": True},
]
design = {  # optional: sample_id -> that sample's daimax TestCase objects
    "s_todo_001": [
        {"id": "TC_LAUNCH", "name": "Launch", "description": "App launches",
         "steps": ["open app"], "expected_result": "home visible", "priority": "P0"},
    ],
}

suite = to_openeval(samples, test_cases=design, suite_id="appbench_v2")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

samples_back = from_openeval(suite)  # EvalSample-shaped dicts (+ "test_cases")
```

`Suite.test_cases[].id` is the daimax `sample_id`, which is the same
`item_id` (`sample_id or prompt_id`) that `run_to_openeval()` uses for
`Result.test_case_id`, so a Suite and a ResultSet built by this package join
on it.

## Mapping

The maintainer's constraints from #9 are quoted where they apply.

### ResultSet

| daimax (`evalapp`) | EvalPort | Notes |
|---|---|---|
| `EvalRun.run_id` | `ResultSet.run_id` | Verbatim. |
| `EvalRun.sample_source` | `ResultSet.suite_id` | Unless `suite_id=` is passed; falls back to `"daimax_appbench"`. |
| `EvalRun.timestamp` | `ResultSet.started_at` | daimax writes `datetime.now().isoformat()`, which is naive (no offset). The adapter attaches `assume_timezone` (default UTC) and records `metadata.daimax.timestamp_raw` and `timestamp_assumed_timezone`; an offset-bearing input is never rewritten; `assume_timezone=None` leaves a naive string untouched. |
| `EvalRun.generator_name` | `ResultSet.provider.model`, `metadata.daimax.generator_name` | The generator is the system under test. |
| `EvalRun.summary` (`EvalSummary`) | `metadata.daimax.summary` | Kept whole. `ResultSet.summary` is schema-shaped (`total`/`passed`/`failed`/`pass_rate`/`avg_score`/`duration_ms`) and computed from the emitted results, because `summary` has `additionalProperties: false`. |
| `PromptResult` | one `Result` | |
| `PromptResult.item_id` (`sample_id or prompt_id`) | `Result.test_case_id` | Both raw ids are kept in `metadata.daimax.prompt_id` / `sample_id`. |
| `PromptResult.generation_duration` (float seconds) | `Result.duration_ms` | `int(round(seconds * 1000))`; raw seconds in `metadata.daimax.generation_duration_s_raw`. |
| `PromptResult.generation_success == False` | `Result.error` | Only then. `type` is `provider_error` (the generator is the EvalPort-analogue of the provider that should have produced the output; the harness did not fail, so `runner_error` is not used), or `timeout` when `process_data.error_type` contains "timeout". `message` = `error_message` only when non-empty (never fabricated); `code` = `process_data.error_type` when set. |
| `PromptResult.success_rate` / `quality` / `experience` | three `GraderResult`s, `type: "custom"` | "`usecase_completeness`, `stability_deduction`, and `backend_deduction` belong specifically to `QualityMetrics`. Each of the three synthetic graders should preserve its own metric's breakdown". `gr_daimax_quality.metadata.daimax.breakdown` holds the whole `QualityMetrics` (usecase_completeness, e2e counts, stability_score/deduction, backend_completeness/deduction, compliance, ...); `gr_daimax_success_rate` holds the three rates + weights + reasons; `gr_daimax_experience` holds duration, package size, tokens and aesthetics. Nothing is cross-mixed or flattened. |
| `*.composite_score` (0..100) | `GraderResult.score` | `composite_score / 100`, clamped to [0, 1]. Raw value in `metadata.daimax.composite_score_raw`. |
| `usecase_completeness`, `stability_score`, `backend_completeness`, `compliance_score`, `initial_generation_rate`, `issue_fix_rate`, `requirement_extension_rate`, `duration_score`, `package_size_score` (0..100) | `metadata.daimax.normalized.*` | `/ 100`; raw values stay in `breakdown`. |
| `ExperienceMetrics.aesthetics_score` (0..10, `None` = not scored) | `metadata.daimax.normalized.aesthetics_score` | `/ 10`; raw in `breakdown.aesthetics_score`. |
| metric is `None` | `score: null` | "EvalPort's existing `score: null` convention fits unexecuted checks well." Plus `metadata.daimax.skip_reason: "metric_not_computed"`. |
| `TestCaseResult` | one `GraderResult` per entry, `grader_id: "gr_daimax_e2e"` | `type` is the namespaced `org.daimax.vision_e2e_step` (maintainer request). EvalPort's `grader.json` treats `type` as an open string -- any type outside the well-known list is validated exactly like `custom` and only requires `params.handler` -- so the namespaced type is used directly and the shared grader definition keeps `params.handler: "daimax:vision_e2e_step"`. No `type: "custom"` fallback was needed. |
| `TestCaseResult.passed` | `GraderResult.passed` | Verbatim ("please keep the original scores and statuses intact"). |
| `TestCaseResult.status` `"PASS"` / `"FAIL"` | `score` `1.0` / `0.0` | Status string kept in `metadata.daimax.status`. |
| `TestCaseResult.status` `"SKIPPED"` | `score: null` | The unexecuted-check convention; native `passed` (`False`) is still kept verbatim; `metadata.daimax.skip_reason: "SKIPPED"`. |
| `TestCaseResult.details` | `GraderResult.reason` | |
| `TestCaseResult.duration` (seconds) | `metadata.daimax.duration_ms`, `duration_s_raw` | |
| `report_path`, `report_started_at`, `report_generated_at`, `verifications` | `metadata.daimax.*` | When present. |
| `TestCase.priority` `P0`/`P1`/`P2` | `metadata.daimax.priority_weight` `3`/`2`/`1` | **Project convention, not a daimax-native quantity.** The maintainer accepted the mapping on the condition that it be documented as such; it is applied only when you pass `test_cases=`, it is never used to compute any score, and the table is exported as `PRIORITY_WEIGHTS` and repeated in `metadata.daimax.priority_weight_convention`. |
| (none) | `GraderResult.passed` for the three metric graders | **Adapter policy, explicit.** "The sketch's 70-point cutoff isn't a native daimax-wide pass threshold, so any derived verdict should be clearly documented as an adapter policy." Default `pass_threshold=None` (policy `native_booleans`): `passed` is daimax's own `generation_success`, the only per-item boolean daimax computes -- no verdict is derived from any score. With `pass_threshold=<0..100>` (policy `composite_threshold`): `passed = composite_score >= pass_threshold`; a missing metric is `passed: false`. The chosen policy is written to `metadata.daimax.adapter_policy` on the ResultSet and on every Result. |
| (derived) | `Result.passed` | `all(grader.passed)`, so under the default policy it equals `generation_success and all(TestCaseResult.passed)` -- native booleans only. |
| `platform`, `sample_title`, `sample_complexity`, `sample_top_category`, `requirement`, `session_id`, `project_id`, `project_path`, `e2e_report_path`, `requires_backend`, `error_message`, `result_data.{build,install,launch,generation}_status`, `process_data.error_type`, `process_data.token_*` | `Result.metadata.daimax.*` | |

No `actual_output` is emitted: daimax's output is a generated app (a project
directory / build artifact), not text. `project_path` and `e2e_report_path`
in metadata are the pointers to it.

### Suite

| daimax | EvalPort | Notes |
|---|---|---|
| `EvalSample.sample_id` | `TestCase.id` | Same key as `Result.test_case_id`. |
| `EvalSample.requirement` | `TestCase.input` | Must be non-empty (schema `minLength: 1`); an empty requirement raises. |
| `app_type`, `top_category`, `complexity` | `TestCase.tags` | When non-empty. |
| `title`, `platforms`, `app_type`, `game_category`, `complexity`, `top_category`, `core_functions`, `constraints`, `notes`, `requires_backend`, `requires_auth`, `dataset_version`, `status`, `deprecated_reason`, `pages`, `mock_resources` | `TestCase.metadata.daimax.*` | |
| daimax `TestCase` objects for that sample (`test_cases={sample_id: [...]}`) | `TestCase.metadata.daimax.test_cases[]` | Each with `priority_weight` added per the convention above. EvalPort's grader list is the four shared daimax graders; daimax test cases are the *inputs* to the `gr_daimax_e2e` grader, not graders of their own. |
| (shared) | `Suite.graders` | `gr_daimax_success_rate`, `gr_daimax_quality`, `gr_daimax_experience` (`custom`, handlers `daimax:*_metrics`) and `gr_daimax_e2e` (`org.daimax.vision_e2e_step`, handler `daimax:vision_e2e_step`). |

### Lossy fields

- `run_from_openeval()` recovers `PromptResult`-shaped dicts that validate as
  real `PromptResult` models, and the three metrics round-trip exactly (their
  full dumps live in `breakdown`). `process_data` comes back with only
  `error_type` and the token counts; `result_data` with only the four status
  strings; `EvalSummary` is not re-hydrated into a dict of `PromptResult`s
  (it lives in `ResultSet.metadata.daimax.summary`).
- Suite direction: `from_openeval()` recovers every `EvalSample` field the
  adapter wrote; the deprecated `platforms` default is preserved only if it
  was present in the input.
- The naive timestamp's true offset is unknowable from the data; the
  adapter's assumption is always recorded next to the raw string rather than
  silently applied.

## Tests

```
pip install -e ".[test]"
python -m pytest -q
```

Fully offline. Fixtures are plain dicts shaped like `model_dump()`; when
`evalapp` is importable (`pip install -e ".[daimax]"`, or an editable install
of the upstream clone), two extra tests feed the same fixtures through the
real `EvalRun` / `PromptResult` / `TestCaseResult` / `QualityMetrics` /
`EvalSample` / `TestCase` models and assert identical output, otherwise they
skip.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 - see LICENSE.
