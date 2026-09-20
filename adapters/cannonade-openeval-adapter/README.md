# cannonade-openeval-adapter

Convert [Cannonade](https://github.com/cannonade-ai/cannonade)'s on-disk
`TestSuite` and `TestRun` JSON files to and from the
[EvalPort](https://github.com/adhabnr-ux/evalport) open interchange format.

## Why this is a standalone package, not a Cannonade core change

Cannonade's maintainer ([@BekirUzun](https://github.com/BekirUzun)) passed on
adding an EvalPort dependency or a `TestRun.version` field, but confirmed the
on-disk JSON contract is deliberate and documented ("Local first: prompts,
test suites, and test runs are all stored locally in JSON files") and said
"You're welcome to build the adapter, of course." Full discussion:
[cannonade-ai/cannonade#69](https://github.com/cannonade-ai/cannonade/issues/69).

This adapter reads the plain JSON files Cannonade already writes:

- `~/.cannonade/suites/<id>.json` — one `TestSuite` per file
- `~/.cannonade/runs/<id>-<slug>.json` — one `TestRun` per file

grounded directly against Cannonade's real TypeScript source (not guessed):
`src/shared/app/test-suite.ts` (`TestSuite`, `TestCase`, `TestInput`,
`EvaluationConfig` — 14 evaluation types), `src/shared/app/test-run.ts`
(`TestRun`, `PerModelRun`, `TestCaseRun`, `ModelRef`), and
`src/shared/app/judge.ts` (`JudgeUsage`).

## An explicit, unresolved caveat (not glossed over)

The maintainer was direct about this: "the layout of `~/.cannonade/*` is an
internal implementation detail, not a published contract... anything reading
those files should expect to break and validate defensively." `TestRun` also
carries no `version` field (unlike `TestSuite.version`), so this adapter has
**no reliable signal** for detecting an incompatible on-disk schema change in
a future Cannonade release. Every field access uses `.get()` with safe
defaults specifically because of this, and a real Cannonade upgrade may still
require an update here. Pin your Cannonade version if you depend on this
pipeline for anything long-running.

## Install

```bash
pip install -e .
```

Requires `evalport-sdk>=1.0.0`.

## Usage

```python
import json
from cannonade_openeval_adapter import suite_to_openeval, run_to_openeval, to_openeval
from openeval.validate import validate_suite, validate_result_set

suite = json.loads(open("~/.cannonade/suites/abc123.json").read())
openeval_suite = suite_to_openeval(suite)
assert validate_suite(openeval_suite).valid

run = json.loads(open("~/.cannonade/runs/abc123-my-run.json").read())
result_sets = run_to_openeval(run)  # one per model in run["modelRuns"]
for rs in result_sets:
    assert validate_result_set(rs).valid

# Or let to_openeval() dispatch by shape:
converted = to_openeval(suite)      # -> dict (a suite)
converted = to_openeval(run)        # -> list[dict] (ResultSets)
```

## Why two conversion entry points instead of one `to_openeval()`

Unlike most adapters in this repo, Cannonade produces two structurally
different artifacts that map onto two different EvalPort documents: a
`TestSuite` (maps to an EvalPort Suite) and a `TestRun` (maps to N EvalPort
ResultSets, one per model). Forcing both through one `to_openeval()` would
require guessing the caller's intent from shape alone, so this adapter
exposes `suite_to_openeval()` and `run_to_openeval()` explicitly, plus a
`to_openeval()` dispatcher for callers who don't care which they're passing.

## Mapping notes

- `TestRun.modelRuns[]` (`PerModelRun`) each become one EvalPort `ResultSet`,
  all sharing `suite_id = TestRun.suiteId` — exactly the "compare N models
  against one suite" shape EvalPort's grouped/sibling ResultSets feature
  ([RFC Discussion #45](https://github.com/adhabnr-ux/evalport/discussions/45),
  merged in [PR #54](https://github.com/adhabnr-ux/evalport/pull/54)) was
  built for. Each ResultSet's `group` field ties all of one TestRun's
  PerModelRuns together via `group_id = TestRun.id`, with `role`/`label` set
  to a human-readable model identifier (`"<source>:<modelKey or modelId>"`)
  and `sequence` set to the PerModelRun's position in `modelRuns[]`.
- `EvaluationConfig.type` maps directly to an EvalPort standard grader type
  for `exact_match`, `contains`, `regex`, and `cosine_similarity` (→
  `semantic_similarity`). The other nine types (`json_match`, `bleu`,
  `rouge`, `levenshtein`, `f1`, `custom`, `code_execution`,
  `html_validation`, `llm_rubric`, `g_eval`) have no first-class EvalPort
  grader type in v1 and are mapped to `type="custom"` with
  `params.handler="cannonade:<type>"`, the same pattern the
  [ragas-openeval-adapter](../ragas-openeval-adapter) uses for Ragas metrics
  without a 1:1 EvalPort type, rather than forcing a fake direct mapping.
  **`llm_rubric`/`g_eval` are deliberately included in this "custom" bucket
  rather than `llm_judge`** because Cannonade's suite files don't record
  which judge model will score a given rubric (that's a run-time/app-level
  setting, not per-test-case), and `llm_judge`'s EvalPort schema requires a
  concrete `params.model` — putting a fabricated model name there would
  misrepresent something this adapter doesn't actually know. The real judge
  model *used*, when there is one, is preserved verbatim per-result via
  `JudgeUsage` (see below).
- `TestCase.passingLogic` (`'all'` | `'any'`) has no EvalPort equivalent —
  EvalPort graders score independently. It rides along as
  `TestCase.metadata["cannonade"]["passing_logic"]`, informational only, and
  the already-computed `Result.passed` (from Cannonade's own
  `TestCaseResult.passed`) is **carried through as-is rather than
  re-derived**, so this adapter never second-guesses Cannonade's own
  pass/fail logic.
- `AggregateMetrics` (Cannonade's own per-model summary) is preserved
  verbatim as `ResultSet.summary` rather than recomputed.
- `JudgeUsage` (LLM-judge cost/token info, when a rubric/g_eval grader used
  one) is preserved verbatim in each `GraderResult.metadata["judge_usage"]`.
- Cannonade's `TestSuite.version` (the suite CONTENT's own version, not a
  spec version) is preserved in `metadata["cannonade"]["suite_version"]`;
  EvalPort's own top-level `version` field is always the EvalPort spec
  version this document targets (`OPENEVAL_VERSION`), matching every other
  adapter in this repo.
- A `TestCaseRun` whose `result` is `null` (run cancelled or failed before
  completion) produces the most honest `Result` possible from just the run's
  own `status` field, rather than fabricating grader results that were never
  computed.
- A `PerModelRun` with zero `caseRuns` (e.g. it failed before running
  anything) still needs `validate_result_set()` to pass, which requires a
  non-empty `results` list — this adapter emits one placeholder `Result`
  (`test_case_id="cannonade_no_case_runs"`, `passed=False`) in that case
  rather than crashing or emitting an invalid document. The same applies to
  `suite_to_openeval()`: a `TestCase` with zero configured `evaluations`
  gets one placeholder grader scoped to that test case
  (`id="<test_case_id>__gr_default"`), so `validate_suite()`'s "every
  grader reference resolves" rule holds even when other test cases in the
  same suite *do* have real graders — a suite-wide shared `"gr_default"` id
  would dangle in exactly that mixed case, which is why it's per-test-case
  rather than global.

## Lossiness of `openeval_to_suite()`

The reverse direction can only round-trip what this adapter itself put into
`metadata["cannonade"]`. It is **not** a general EvalPort → Cannonade
importer for suites built by other producers: EvalPort's schema has no
equivalent for `RunConfig` sampling parameters, `promptRef`, or `timeoutMs`,
so those are recovered only when this adapter's own `raw_input`/`cannonade`
metadata block is present. `evaluations` is always returned empty — a
grader's EvalPort shape (`type`, `params`) is not reversible back to the
original Cannonade `EvaluationConfig` without more information than EvalPort
carries.

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v
```

All tests run against the real `openeval.validate.validate_suite()` /
`validate_result_set()` from `evalport-sdk` — nothing is mocked.
