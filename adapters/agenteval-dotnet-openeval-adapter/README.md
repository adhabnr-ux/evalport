# EvalPort.AgentEval — AgentEval (.NET) ⇄ EvalPort

> **Status: standalone package, not an upstream integration.** This lives in the EvalPort repository
> and depends on the published [`AgentEval`](https://www.nuget.org/packages/AgentEval) NuGet package
> (0.43.0-beta). It was built after AgentEval's maintainer, in
> [AgentEvalHQ/AgentEval#203](https://github.com/AgentEvalHQ/AgentEval/issues/203), named
> `IResultExporter` and `IDatasetLoader` as the intended extension points and suggested "a small
> package on your side (for example `EvalPort.AgentEval`) implementing both interfaces", with an offer to
> link and review it. He reviewed the 0.1.0-beta version end to end on 2026-10-04 (source, tests, sample
> and CI) and found one real problem, the `test_cases_file` confinement fixed in 0.1.1-beta; the three
> observations this package made about AgentEval led to AgentEval 0.43.0-beta's verdict changes
> ([AgentEvalHQ/AgentEval#279](https://github.com/AgentEvalHQ/AgentEval/pull/279), see
> "What changed between 0.42 and 0.43" below). It is not on NuGet; the maintainer has offered to link a
> tagged release from AgentEval's `docs/export.md` as a third-party package.
>
> **Two different projects are called AgentEval.** This package is for
> [AgentEvalHQ/AgentEval](https://github.com/AgentEvalHQ/AgentEval), the .NET toolkit for Microsoft
> Agent Framework. The Python package [`agenteval-openeval-adapter`](../agenteval-openeval-adapter/) in
> this repository is for [lokesh75-kank/agenteval](https://github.com/lokesh75-kank/agenteval), a
> TypeScript determinism-sampling tool. They share nothing but the name.

## What it does

| Entry point | AgentEval type | Direction | What it carries |
|---|---|---|---|
| `EvalPortDocuments.Build(EvalPortRun)` | `EvalResult` trees (`CompositeEval`, `AtomicLlmEval`, `AtomicCodeEval`, …) | AgentEval → EvalPort `Suite` + `ResultSet` | **The result model.** `MeasurementState`, the `skipped` / `error` / `inapplicable` labels, composite verdicts with their aggregation strategy, judge provenance. |
| `EvalPortResultExporter` (`IResultExporter`, format name `evalport`) | `EvaluationReport` | AgentEval → EvalPort `ResultSet` | What the flat report has: per test a 0–100 score, `Passed`, `Skipped`, an error string, assertions, metric scores. **No measurement state** (see below). |
| `EvalPortDatasetLoader` (`IDatasetLoader`, format name `evalport`, extension `.evalport.json`) | `DatasetTestCase` | EvalPort `Suite` → AgentEval | `id`, `input` (multi-turn joined, turns kept), `expected_output`, `context`, `expected_tools`, `tags`, `metadata`; the case's resolved graders as data. A `test_cases_file` is followed only inside the suite's own directory (no absolute paths, no `..`, no links leading out), since suites can come from other people and those lines become model inputs. |
| `services.AddEvalPortAgentEval()` | DI | | Registers both; `AddAgentEvalDataLoaders()` puts them in `IExporterRegistry` and `IDatasetLoaderFactory`. |

```csharp
using AgentEval.DependencyInjection;
using EvalPort.AgentEval;

services.AddEvalPortAgentEval();
services.AddAgentEvalDataLoaders();   // not AddAgentEval(): only this one (or AddAgentEvalAll()) builds the registries

var exporter = provider.GetRequiredService<IExporterRegistry>().GetRequired("evalport");
await exporter.ExportAsync(report, File.Create("run.evalport.json"));

var cases = await provider.GetRequiredService<IDatasetLoaderFactory>()
    .CreateFromExtension(".evalport.json").LoadAsync("suite.evalport.json");
```

For the result model, build the documents from the results you already have:

```csharp
var run = new EvalPortRun
{
    SuiteId = "suite_my_evals", RunId = "run_2026_10_04",
    Cases = gradedCases.Select(c => new EvalPortCaseRun(c.DatasetCase, c.AgentOutput, new[] { c.CompositeResult })).ToList(),
};
var (suite, resultSet) = EvalPortDocuments.Build(run);
File.WriteAllText("suite.json", EvalPortJson.ToJsonText(suite));
File.WriteAllText("results.json", EvalPortJson.ToJsonText(resultSet));
```

Both outputs validate against the EvalPort 1.0.0 JSON Schemas (`schema/`) and both SDK validators on
`main` (the tests check the schemas; CI also runs the Python SDK over `sample_output/`).

## How AgentEval's three states become EvalPort

AgentEval keeps "measured", "the case could not test this" (`NotApplicable`) and "the instrument did
not run" (`NotMeasured`) apart, by enum and by label, and its own predicate for "does this score count"
is `EvalScoreExtensions.CountsTowardAggregate()`. EvalPort has one word for both non-measurements,
Validation Rule 6: `"score": null` with `"passed": false`, excluded from aggregation. So:

| AgentEval `EvalScore` | EvalPort `GraderResult` |
|---|---|
| `CountsTowardAggregate()` is true | `score` = `Value` (clamped to [0, 1]), `passed` = `Passed` |
| anything else: `Measurement` is `NotApplicable` or `NotMeasured`, or the label is `skipped`, `error` or `inapplicable` | `score: null`, `passed: false` |

Which state it was is kept next to the null, under `metadata.agenteval`: `label`, `severity`,
`measurement` (`measured` / `notApplicable` / `notMeasured`, computed with AgentEval's `CensusBucket()`,
spelled as AgentEval's own `eval-result.schema.json` spells them), the raw `value`, and the
`provenance` (`type`, `judge_model`, `prompt_id`, `prompt_hash`, tokens, cost, cache hit). A grader's
`type` is `agenteval_<provenance type>` (`agenteval_atomic_llm`, `agenteval_atomic_code`, …),
a framework-specific type in EvalPort's sense, because the result does not carry the prompt text or
source that EvalPort's well-known `llm_judge` or `code` types require.

**Composites.** Only leaves become `GraderResult`s. `Result.passed` is the composite's own `Score.Passed`
(the AND over the case's top-level results when there are several). The composite's label, score,
severity, aggregation strategy and coverage note, and the same for every component, are written under
`Result.metadata.agenteval.verdicts`, with a three-way `census` over the leaves (AgentEval's
`Census()`). Whether a component was `Required` is not on the result (it is on the `EvalComponent`),
so it is not written; the composite's label already accounts for it. This package does **not**
re-aggregate the leaves, so `Result.passed` is AgentEval's verdict and not necessarily what EvalPort's
default `all` aggregation would compute (next section).

## What AgentEval does when one of two graders cannot measure a case

`sample_output/` is written by `samples/EvalPort.AgentEval.Samples` from AgentEval's real
`CompositeEval`, `AtomicLlmEval` and `ChatClientEvaluator` on seven constructed cases. The judge is
AgentEval's own `FakeChatClient` with canned replies; there is no LLM, network or API key. **The
cases are constructed and their proportions mean nothing**; the framework's handling of them is real,
and the first test class fails if a newer AgentEval changes it.

| Case | `exact` | `judge` | Composite, judge **required** | Composite, judge **optional** |
|---|---|---|---|---|
| q1 | pass | pass | `pass` | `pass` |
| q2 | pass | **error** (no JSON from the judge, twice) | **`error`**, passed false ("a required component errored, so no pass/fail verdict is reported") | **`pass`**, passed true, note "Measured 1 of 2" |
| q3 | fail | error | **`fail`** ("a required part produced no verdict, but a high failure the measured parts show decides it") | `fail` |
| q4 | skipped | error | `error` (nothing measured, a required component errored) | `skipped` (nothing measured; the errored component is optional, so it decides nothing) |
| q5 | inapplicable (no ground truth) | pass | `pass` | `pass` |
| q6 | — | chat client **throws** | the exception leaves `EvaluateAsync`; no result | same |
| q7 | pass | skipped | **`warn`**, passed false, `measurement: notMeasured` ("required component(s) that did not run … a pass cannot rest on them") | `pass`, note "Measured 1 of 2" |

Summary over the seven rows: **2 / 3 / 2 passed / failed / skipped, pass rate 0.2857** with the judge
required; **4 / 1 / 2, pass rate 0.5714** with it optional. Same leaves, both documents valid, the
difference is one flag on the eval definition.

### What changed between 0.42 and 0.43

The first version of this package pinned 0.42.0-beta and reported three things to AgentEval's maintainer
in #203. AgentEval 0.43.0-beta (released 2026-10-06; its CHANGELOG entry "A required component that did
not run no longer lets a composite pass" cites #203) changed the framework, and the same seven cases
now read differently. The first test class pins the new behavior and keeps the old expectations in
comments.

| Case | 0.42.0-beta | 0.43.0-beta | Why (from the maintainer's #203 follow-up and the 0.43 CHANGELOG) |
|---|---|---|---|
| q7, judge required | `pass` | `warn`, `measurement: notMeasured` | A required component that returned `skipped` no longer lets the composite pass. The composite's own score is marked not measured, so `CountsTowardAggregate()` is false for it; a parent reads that, not the label. |
| q3, judge required | `error` | `fail` | A required component's `error` is the verdict *unless* the composite would fail even if the errored parts had passed; a measured failure at `high` severity is such a case, so the established failure is reported instead of being hidden behind the error. |
| q4, judge optional | `error` | `skipped` | An optional component that errors no longer decides the verdict; with nothing measured, the composite is `skipped`. |
| q2, both | unchanged | unchanged | Required → `error`; optional → `pass` with the measured share noted. |

Two consequences for this package. `ProposedVerdict` now distinguishes a measured `warn` (a soft fail,
`failed`) from a `warn` whose `CensusBucket()` is `NotMeasured` (the pass was held back because
something did not run: `unverified`). And the `DependencyInjectionTests` pin that `AddAgentEval()` alone
does not build the exporter/loader registries still holds in 0.43; the maintainer confirmed it and the
AgentEval docs now say which call builds which registry.

Also in 0.43, per the maintainer's #203 follow-up and the CHANGELOG, **not exercised by this package's
sample** (which only drives `CompositeEval` / `AtomicLlmEval`), so recorded here rather than tested:
`EvalComponent.OnFailure` (`Averaged`, the old default, `Warn`, `Fail`, `FailUnlessPass`), which lets a
composite read `warn` or `fail` while its score is above threshold (the label is the verdict, not
`score.value`; the mapping already treats it that way); `MultiJudgeWrapper` and
`AdjudicatedMultiJudgeWrapper` now respect each judge's `Required`; `EvalScore.Label` is stored
lower-case; the flat `IMetric` results gained `MetricResult.Measured` / `MetricResult.NotMeasured`, and
`TestResultSummary.MetricsNotMeasured` (name → reason) carries them in the flat report, which
`EvalPortResultExporter` does not yet read; a required component's `error` is no longer always the
verdict (see q3 above); and through the MAF bridges a not-measured metric fails its item, because MAF has
no third state. The exporter's handling of `MetricsNotMeasured` is the next thing to add.

What this says, and does not say, about EvalPort:

- **q2 is the row Rule 6 cannot name.** One grader scored a pass and the other produced nothing.
  EvalPort's default aggregation (null-scored graders excluded, AND of the rest) makes it `passed: true`;
  that is AgentEval's *optional* path. DeepEval's row flag (see
  [`examples/errored-graders/`](../../examples/errored-graders/)) is false; that is AgentEval's
  *required* path. AgentEval's answer to "what should `passed` be when a grader did not run" is "it
  depends on whether that grader was required", declared per component, and either way it writes
  *how much of the verdict was measured* into the result. EvalPort has no slot for either fact; this
  package puts both in `metadata.agenteval`.
- **AgentEval has "no verdict" at the result level, which EvalPort does not.** q2 and q4 with the
  judge required (`error`) and q7 (`warn` + `notMeasured`) are all `passed: false` on the EvalPort
  side, which a reader of `main` must take as "verified failing" when any grader scored (q2, q7) or
  as "not verified" only when every grader is null (q4). AgentEval's own reading of all three is that
  no pass/fail verdict was established. That is a case for a result-level "not established" marker
  ([Discussion #49](https://github.com/adhabnr-ux/evalport/discussions/49)), and for the alternative it
  has to beat, a `metadata` convention, which is what this package does.
- **An established failure is reported even when a required judge errored (0.43).** q3 with the judge
  required is `fail`, not `error`: the exact-match failure decides. On the EvalPort side that row is
  `passed: false` with one scored failing grader, which Rule 6 reads correctly as "verified failing".
  This is a framework choosing "the measured failure wins over the error". The errored part is a
  component, not the whole run, so it is not literally #83's `error` + `failed` case, but it is the
  same question answered the other way, and a concrete input to #49's open question on it.
- **Two kinds of "judge unreachable".** A judge that answers with no usable JSON becomes a grader-level
  `error` label and the composite still returns; a judge whose transport throws takes the whole case
  down (q6), so the only EvalPort representation is `Result.error` (`runner_error`) with nothing graded.
  The same outage produces a null-scored grader or an errored row depending on where it fails.
- **Severity-driven verdicts.** With no threshold, a `CompositeEval`'s pass/fail is read from the
  required components' severity, not from their pass flags. `exact` therefore fails with severity
  `high` in the sample so that a wrong answer fails the composite; a `low`-severity failure would not.
  This is AgentEval's design and is recorded in `metadata.agenteval.verdicts[].severity`.

## What `IResultExporter` cannot carry (a finding for AgentEval)

AgentEval's `IResultExporter.ExportAsync(EvaluationReport, Stream)` receives the flat report, whose
`TestResultSummary` has `Score` (0–100), `Passed`, `Skipped`, `Error`, `StackTrace`, `Output`,
`MetricScores` (0–100, no per-metric pass flag) and `Assertions`. It has no `EvalResult`, so no
`MeasurementState`, no labels, no composite tree and no provenance. Point 1 of the maintainer's list
("not measured is not failed") cannot be honored through that interface, only through
`EvalPortDocuments`. What the exporter does with what it has:

| `TestResultSummary` | EvalPort `Result` |
|---|---|
| `Name` | `test_case_id`; repeated names get a 1-based `attempt` |
| `Score`, `Passed` | one grader `agenteval_test`: `score` = `Score / 100`, `passed` = `Passed` |
| `Assertions[i]` | grader `assertion_<i+1>` (`agenteval_assertion`): 1.0 / 0.0, or `score: null`, `passed: false` for `AssertionOutcome.Inconclusive` |
| `Skipped` | `grader_results: []`, `passed: false`, `metadata.openeval.aggregation_status: "unscored"` (the report's own `Passed` is kept in metadata) |
| `Error` **with** `StackTrace` | `Result.error` `{type: "runner_error", message}` |
| `Error` **without** `StackTrace` | the test grader's `reason` (AgentEval's `BuildErrorMessage` writes `FailureReport.WhyItFailed` into the same field, and the report does not distinguish the two any other way) |
| `MetricScores` | `metadata.agenteval.metric_scores_0_100`, not graders: they have no pass flag |
| `TotalTests` … `OverallScore` | kept verbatim under `metadata.agenteval.report`; `summary` is recomputed from the rows written so the two always agree |

`Format` returns `ExportFormat.Json` because the enum is closed (AgentEval's docs prescribe the closest
built-in); `FormatName` is `evalport` and `FileExtension` is `.evalport.json`.

## The two open EvalPort RFCs, tested against this output

Both are opt-in through `EvalPortExportOptions` and **neither is in the spec**.

- `EmitProposedVerdict` adds `Result.verdict` ([#49](https://github.com/adhabnr-ux/evalport/discussions/49),
  reference PR #83): `passed` when every top-level label is `pass`; `failed` for `fail`, or for a `warn`
  that was measured (AgentEval calls that a soft fail); `unverified` for `error`, `skipped`,
  `inapplicable`, a `warn` whose measurement is `notMeasured` (0.43's "pass held back"), a run error or
  no results. The JSON Schema on `main` rejects the field (`additionalProperties: false` on `Result`;
  the tests assert exactly seven such errors), while both SDK validators on `main` accept it, which is
  the gap [#108](https://github.com/adhabnr-ux/evalport/discussions/108) is about. Checked against the
  #83 validator: the documents validate; flipping q2 (judge optional, `passed: true`) to `unverified`
  gives `VERDICT_PASSED_MISMATCH`, so the proposal cannot mark that partly-measured pass; `failed` on
  q6 gives `VERDICT_ERROR_CONFLICT`; q7 (judge required) is `unverified` with `passed: false`, which #83
  accepts, and it is the first row in this repository where a framework's own result-level state maps
  onto `unverified` without this package inventing anything.
- `EmitProposedJudgeIdentity` adds `metadata.openeval.judge.model` and `observed_at` on graders whose
  provenance names a judge model ([#118](https://github.com/adhabnr-ux/evalport/discussions/118),
  reference PR #119). Only `model` is written, because AgentEval records the judge it was configured
  with, not the model that answered, and no fingerprint is invented. AgentEval's `PromptHash` is the
  first 16 hex characters of a SHA-256 over its judge-input framing version, the prompt material and
  the criteria, so it is **not** written as the proposed `prompt_sha256`; it stays
  `metadata.agenteval.provenance.prompt_hash`. That is one real producer whose prompt digest does not
  fit the proposed `prompt_sha256` / `prompt_basis` pair.

**On the #49 alternative-B branch only (declared aggregation, not in the spec, not on `main`):** every
ResultSet this package writes also carries `metadata.openeval.aggregation: {"strategy": "producer"}`.
That branch requires a row with some null-scored and some scored graders (q2, q3, q4, q7 here) to say how
its `passed` was derived, and `producer` is the proposed name for what `passed_semantics` already says:
`passed` is AgentEval's own verdict and is not re-derived from the graders. The branch's tests also
teach the JsonSchema.Net helper to ignore a failed `if` subschema, which that schema is the first in this
repository to use on `resultset.json`. On `main` the key is free-form metadata and nothing reads it.

## What is lost

`EvalComponent.Required` and `Weight` (definition-side, not on the result); the prompt text and the
judge's raw reply (AgentEval does not keep them on the result either); `EvalProvenance` for the composite
itself beyond its type; `ComparabilityFacts`, `StimulusHash` and `JudgeFingerprint`, which live on
AgentEval's `ScenarioResult`, not on `EvalResult`; and on the flat path everything listed above. The
loader turns no EvalPort grader into an AgentEval metric: a suite's graders arrive as data under
`Metadata["evalport.graders"]` for the caller to map.

## Build, test, regenerate

```bash
# .NET 10 SDK (global.json: 10.0.100, rollForward latestFeature)
dotnet test                                                            # 68 tests
dotnet run --project samples/EvalPort.AgentEval.Samples -- --out-dir sample_output --deterministic
dotnet run --project samples/EvalPort.AgentEval.Samples -- --out-dir sample_output --deterministic --proposed-verdict
dotnet pack src/EvalPort.AgentEval -c Release                          # EvalPort.AgentEval.0.1.1-beta.nupkg (not published)
```

The tests validate every document against this repository's `schema/*.json` with JsonSchema.Net 7.3.4
(the version AgentEval itself depends on) and compare the regenerated `sample_output/` byte for byte
with the checked-in files. `.github/workflows/agenteval-dotnet-adapter.yml` does the same on CI and then
runs the Python SDK validator over `sample_output/`, and the #83 branch's validator over
`sample_output/proposed-verdict/`.

## Credit

Direction from [@joslat](https://github.com/joslat) (AgentEval's maintainer) on
[AgentEvalHQ/AgentEval#203](https://github.com/AgentEvalHQ/AgentEval/issues/203): the three things a
flat export loses (measurement state, composite trees, provenance), the two extension points, the
package name, and the review of 0.1.0-beta that found the `test_cases_file` problem. Built by Sahi; not
affiliated with AgentEval.

## Changelog

- **0.1.1-beta** (2026-10-07): AgentEval 0.43.0-beta; `test_cases_file` confined to the suite's
  directory (review finding, #203); `ProposedVerdict` reads `warn` + `notMeasured` as `unverified`;
  sample output regenerated (q3 required `error` → `fail`, q4 optional `error` → `skipped`, q7 required
  `pass` → `warn`).
- **0.1.0-beta** (2026-10-04): first version, against AgentEval 0.42.0-beta.

## License

Apache 2.0 — see [LICENSE](LICENSE). AgentEval is MIT-licensed and is used here as a NuGet dependency.
