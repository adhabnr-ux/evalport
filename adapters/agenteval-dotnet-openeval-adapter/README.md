# EvalPort.AgentEval — AgentEval (.NET) ⇄ EvalPort

> **Status: standalone package, not an upstream integration.** This lives in the EvalPort repository
> and depends on the published [`AgentEval`](https://www.nuget.org/packages/AgentEval) NuGet package
> (0.42.0-beta). It was built after AgentEval's maintainer, in
> [AgentEvalHQ/AgentEval#203](https://github.com/AgentEvalHQ/AgentEval/issues/203), named
> `IResultExporter` and `IDatasetLoader` as the intended extension points and suggested "a small
> package on your side (for example `EvalPort.AgentEval`) implementing both interfaces", with an offer to
> link and review it. As of this writing it has not been reviewed by AgentEval's maintainers, is not
> linked from AgentEval's docs, and is not on NuGet.
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
| `EvalPortDatasetLoader` (`IDatasetLoader`, format name `evalport`, extension `.evalport.json`) | `DatasetTestCase` | EvalPort `Suite` → AgentEval | `id`, `input` (multi-turn joined, turns kept), `expected_output`, `context`, `expected_tools`, `tags`, `metadata`; the case's resolved graders as data. |
| `services.AddEvalPortAgentEval()` | DI | | Registers both; `AddAgentEvalDataLoaders()` puts them in `IExporterRegistry` and `IDatasetLoaderFactory`. |

```csharp
using AgentEval.DependencyInjection;
using EvalPort.AgentEval;

services.AddEvalPortAgentEval();
services.AddAgentEvalDataLoaders();   // not AddAgentEval(): in 0.42.0-beta only this one builds the registries

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
| q2 | pass | **error** (no JSON from the judge, twice) | **`error`**, passed false | **`pass`**, passed true, note "Measured 1 of 2" |
| q3 | fail | error | `error`, passed false (the measured failure is not the verdict) | `fail` |
| q4 | skipped | error | `error` (nothing measured, one errored) | `error` |
| q5 | inapplicable (no ground truth) | pass | `pass` | `pass` |
| q6 | — | chat client **throws** | the exception leaves `EvaluateAsync`; no result | same |
| q7 | pass | skipped | `pass` (a required but *skipped* component does not block the verdict; only `error` does) | `pass` |

Summary over the seven rows: **3 / 2 / 2 passed / failed / skipped, pass rate 0.4286** with the judge
required; **4 / 1 / 2, pass rate 0.5714** with it optional. Same leaves, both documents valid, the
difference is one flag on the eval definition.

What this says, and does not say, about EvalPort:

- **q2 is the row Rule 6 cannot name.** One grader scored a pass and the other produced nothing.
  EvalPort's default aggregation (null-scored graders excluded, AND of the rest) makes it `passed: true`;
  that is AgentEval's *optional* path. DeepEval's row flag (see
  [`examples/errored-graders/`](../../examples/errored-graders/)) is false; that is AgentEval's
  *required* path. AgentEval's answer to "what should `passed` be when a grader did not run" is "it
  depends on whether that grader was required", declared per component, and either way it writes
  *how much of the verdict was measured* into the result. EvalPort has no slot for either fact; this
  package puts both in `metadata.agenteval`.
- **AgentEval has an "error" verdict at the result level, which EvalPort does not.** q2/q3/q4 with the
  judge required are `passed: false` on the EvalPort side, which a reader of `main` must take as
  "verified failing" when any grader scored (q3) or as "not verified" only when every grader is null
  (q4). AgentEval's own word for all three is `error`: no verdict. That is a case for a result-level
  "not established" marker ([Discussion #49](https://github.com/adhabnr-ux/evalport/discussions/49)),
  and for the alternative it has to beat, a `metadata` convention, which is what this package does.
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
  reference PR #83): `passed` when every top-level label is `pass`, `failed` for `fail` or `warn`
  (AgentEval calls `warn` a soft fail), otherwise `unverified`. The JSON Schema on `main` rejects the
  field (`additionalProperties: false` on `Result`; the tests assert exactly seven such errors), while
  both SDK validators on `main` accept it, which is the gap
  [#108](https://github.com/adhabnr-ux/evalport/discussions/108) is about. Checked against the #83
  validator: the documents validate; flipping q2 (judge optional, `passed: true`) to `unverified` gives
  `VERDICT_PASSED_MISMATCH`, so the proposal cannot mark that partly-measured pass; `failed` on q6 gives
  `VERDICT_ERROR_CONFLICT`; `failed` on q3 (judge required) is accepted by #83 although AgentEval's own
  label is `error`, so the mapping writes `unverified` there and drops a measured failure, by the
  framework's choice.
- `EmitProposedJudgeIdentity` adds `metadata.openeval.judge.model` and `observed_at` on graders whose
  provenance names a judge model ([#118](https://github.com/adhabnr-ux/evalport/discussions/118),
  reference PR #119). Only `model` is written, because AgentEval records the judge it was configured
  with, not the model that answered, and no fingerprint is invented. AgentEval's `PromptHash` is the
  first 16 hex characters of a SHA-256 over its judge-input framing version, the prompt material and
  the criteria, so it is **not** written as the proposed `prompt_sha256`; it stays
  `metadata.agenteval.provenance.prompt_hash`. That is one real producer whose prompt digest does not
  fit the proposed `prompt_sha256` / `prompt_basis` pair.

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
dotnet test                                                            # 67 tests
dotnet run --project samples/EvalPort.AgentEval.Samples -- --out-dir sample_output --deterministic
dotnet run --project samples/EvalPort.AgentEval.Samples -- --out-dir sample_output --deterministic --proposed-verdict
dotnet pack src/EvalPort.AgentEval -c Release                          # EvalPort.AgentEval.0.1.0-beta.nupkg (not published)
```

The tests validate every document against this repository's `schema/*.json` with JsonSchema.Net 7.3.4
(the version AgentEval itself depends on) and compare the regenerated `sample_output/` byte for byte
with the checked-in files. `.github/workflows/agenteval-dotnet-adapter.yml` does the same on CI and then
runs the Python SDK validator over `sample_output/`, and the #83 branch's validator over
`sample_output/proposed-verdict/`.

## Credit

Direction from [@joslat](https://github.com/joslat) (AgentEval's maintainer) on
[AgentEvalHQ/AgentEval#203](https://github.com/AgentEvalHQ/AgentEval/issues/203): the three things a
flat export loses (measurement state, composite trees, provenance), the two extension points, and the
package name. Built by Sahi; not affiliated with AgentEval.

## License

Apache 2.0 — see [LICENSE](LICENSE). AgentEval is MIT-licensed and is used here as a NuGet dependency.
