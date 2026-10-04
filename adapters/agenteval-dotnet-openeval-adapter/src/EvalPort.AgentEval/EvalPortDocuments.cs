// SPDX-License-Identifier: Apache-2.0

using System.Text.Json.Nodes;
using AgentEval.Evals;
using AgentEval.Evals.Meta;
using AgentEval.Models;

namespace EvalPort.AgentEval;

/// <summary>
/// One graded case of an AgentEval run, as the input to <see cref="EvalPortDocuments"/>.
/// </summary>
/// <param name="Case">The dataset case that was run (its <c>Id</c> becomes <c>test_case_id</c>).</param>
/// <param name="ActualOutput">What the agent produced, if captured (<c>Result.actual_output</c>).</param>
/// <param name="Results">
/// The top-level <see cref="EvalResult"/>s for this case: one per eval the case was graded with. A
/// composite counts as one; its leaves become the <c>GraderResult</c>s and its own verdict is kept on
/// the <c>Result</c>.
/// </param>
/// <param name="Error">
/// A failure of the run itself (the agent or the harness raised before or during grading), written as
/// <c>Result.error</c> with type <c>runner_error</c>. A judge that errored is <em>not</em> this: that
/// is a leaf with AgentEval's <c>"error"</c> label and stays a grader with a <c>null</c> score.
/// </param>
/// <param name="DurationMs">Wall-clock time of the case, if measured.</param>
/// <param name="Attempt">1-based attempt number when the same case was run more than once in this run.</param>
public sealed record EvalPortCaseRun(
    DatasetTestCase Case,
    string? ActualOutput,
    IReadOnlyList<EvalResult> Results,
    Exception? Error = null,
    long? DurationMs = null,
    int? Attempt = null);

/// <summary>
/// Everything that describes one AgentEval run for the document builder.
/// </summary>
public sealed record EvalPortRun
{
    /// <summary>EvalPort <c>Suite.id</c> and <c>ResultSet.suite_id</c>.</summary>
    public required string SuiteId { get; init; }

    /// <summary>EvalPort <c>ResultSet.run_id</c>.</summary>
    public required string RunId { get; init; }

    /// <summary>The graded cases, in order.</summary>
    public required IReadOnlyList<EvalPortCaseRun> Cases { get; init; }

    /// <summary>Optional <c>Suite.name</c>.</summary>
    public string? SuiteName { get; init; }

    /// <summary>Optional <c>Suite.description</c>.</summary>
    public string? SuiteDescription { get; init; }

    /// <summary>When the run started (<c>ResultSet.started_at</c>). Defaults to now.</summary>
    public DateTimeOffset StartedAt { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>When the run finished (<c>ResultSet.completed_at</c>); omitted when null.</summary>
    public DateTimeOffset? CompletedAt { get; init; }

    /// <summary>The model under test, if known (<c>ResultSet.provider.model</c>).</summary>
    public string? SubjectModel { get; init; }

    /// <summary>Free-form run metadata, written under <c>ResultSet.metadata.agenteval.run</c>.</summary>
    public IReadOnlyDictionary<string, object?>? Metadata { get; init; }
}

/// <summary>
/// Writes AgentEval <see cref="EvalResult"/> trees as an EvalPort <c>Suite</c> plus <c>ResultSet</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is the path that keeps AgentEval's result model: <see cref="MeasurementState"/>, the
/// <c>"skipped"</c> / <c>"error"</c> / <c>"inapplicable"</c> labels, composite trees with their
/// aggregation strategy and per-component <c>Required</c>/<c>Weight</c>, and judge provenance.
/// <see cref="EvalPortResultExporter"/> (AgentEval's <c>IResultExporter</c>) cannot carry any of
/// that, because the interface hands it a flat <see cref="EvaluationReport"/> that has none of it.
/// </para>
/// <para>
/// <c>Result.passed</c> is AgentEval's own verdict: the AND of <c>Score.Passed</c> over the case's
/// top-level results. For a composite that is <c>CompositeEval</c>'s label (<c>"pass"</c> only), which
/// is <c>false</c> whenever a <c>Required</c> component errored, when nothing was measured, or when a
/// pass rested on fewer than <c>MinimumMeasuredShare</c> of the components (<c>"warn"</c>). This package
/// does not re-aggregate the leaves; the spec's default <c>all</c> strategy and AgentEval's verdict
/// can disagree on a row with an unmeasured leaf, and the disagreement is recorded, not resolved
/// (see <c>metadata.agenteval.verdicts</c> and the README).
/// </para>
/// </remarks>
public static class EvalPortDocuments
{
    /// <summary>Build both documents.</summary>
    public static (JsonObject Suite, JsonObject ResultSet) Build(EvalPortRun run, EvalPortExportOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(run);
        options ??= EvalPortExportOptions.Default;
        return (BuildSuite(run), BuildResultSet(run, options));
    }

    /// <summary>
    /// The <c>Suite</c>: one <c>TestCase</c> per distinct case id, one shared grader per distinct eval
    /// key seen in the results. A case whose results reference no grader at all is rejected, because
    /// EvalPort requires at least one grader per test case and inventing one would be a lie.
    /// </summary>
    public static JsonObject BuildSuite(EvalPortRun run)
    {
        ArgumentNullException.ThrowIfNull(run);

        var graders = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        var testCases = new List<JsonObject>();
        var seenCaseIds = new HashSet<string>(StringComparer.Ordinal);

        foreach (var caseRun in run.Cases)
        {
            var leaves = caseRun.Results.SelectMany(EvalResultMapping.Leaves).ToList();
            foreach (var leaf in leaves)
            {
                var key = leaf.Metric.Key;
                if (graders.TryGetValue(key, out var existing))
                {
                    // Prefer a type learned from a result that ran over "agenteval_eval" from a skipped one.
                    if ((string?)existing["type"] == "agenteval_eval" && EvalResultMapping.GraderTypeOf(leaf) != "agenteval_eval")
                        existing["type"] = EvalResultMapping.GraderTypeOf(leaf);
                    continue;
                }
                graders[key] = new JsonObject
                {
                    ["id"] = key,
                    ["type"] = EvalResultMapping.GraderTypeOf(leaf),
                    // Framework-specific types are validated like "custom": params.handler is required.
                    ["params"] = new JsonObject { ["handler"] = key },
                    ["description"] = $"{leaf.Metric.Name} ({leaf.Metric.Category} v{leaf.Metric.Version})",
                };
            }

            var id = RequireId(caseRun.Case);
            if (!seenCaseIds.Add(id))
                continue; // repeated attempts of the same case share one TestCase

            var graderIds = leaves.Select(l => l.Metric.Key).Distinct(StringComparer.Ordinal).ToList();
            if (graderIds.Count == 0 && caseRun.Error is null)
                throw new InvalidOperationException(
                    $"Case '{id}' has no eval results and no error, so it has no graders; EvalPort requires at least one.");
            if (graderIds.Count == 0)
                throw new InvalidOperationException(
                    $"Case '{id}' only errored, so no grader is known for it. Pass the evals' results (even skipped ones) " +
                    "so the Suite can list them, or leave the case out of the Suite.");

            testCases.Add(ToTestCase(caseRun.Case, graderIds));
        }

        var suite = new JsonObject
        {
            ["$schema"] = EvalPortJson.SuiteSchema,
            ["version"] = EvalPortJson.SpecVersion,
            ["id"] = run.SuiteId,
        };
        if (run.SuiteName is not null) suite["name"] = run.SuiteName;
        if (run.SuiteDescription is not null) suite["description"] = run.SuiteDescription;
        suite["graders"] = new JsonArray(graders.Values.Select(g => (JsonNode?)g).ToArray());
        suite["test_cases"] = new JsonArray(testCases.Select(t => (JsonNode?)t).ToArray());
        suite["metadata"] = new JsonObject
        {
            [EvalPortJson.MetadataKey] = new JsonObject
            {
                ["version"] = EvalPortJson.AgentEvalVersion,
                ["exporter"] = "EvalPort.AgentEval",
                ["exporter_version"] = EvalPortJson.PackageVersion,
            },
        };
        return suite;
    }

    /// <summary>One EvalPort <c>TestCase</c> from an AgentEval <see cref="DatasetTestCase"/>.</summary>
    public static JsonObject ToTestCase(DatasetTestCase testCase, IReadOnlyList<string> graderIds)
    {
        ArgumentNullException.ThrowIfNull(testCase);
        ArgumentNullException.ThrowIfNull(graderIds);
        if (graderIds.Count == 0)
            throw new ArgumentException("A TestCase needs at least one grader.", nameof(graderIds));
        if (string.IsNullOrEmpty(testCase.Input))
            throw new ArgumentException($"Case '{testCase.Id}' has an empty Input; EvalPort requires a non-empty input.", nameof(testCase));

        var tc = new JsonObject
        {
            ["id"] = RequireId(testCase),
            ["input"] = testCase.Input,
        };
        if (testCase.ExpectedOutput is not null) tc["expected_output"] = testCase.ExpectedOutput;
        if (testCase.Context is { Count: > 0 }) tc["context"] = Strings(testCase.Context);
        if (testCase.ExpectedTools is { Count: > 0 }) tc["expected_tools"] = Strings(testCase.ExpectedTools);
        tc["graders"] = Strings(graderIds);
        if (testCase.Tags is { Count: > 0 }) tc["tags"] = Strings(testCase.Tags);

        // Everything AgentEval knows about a case that EvalPort has no typed slot for.
        var ae = new JsonObject();
        if (testCase.Category is not null) ae["category"] = testCase.Category;
        if (testCase.PassingScore is { } ps) ae["passing_score"] = ps;
        if (testCase.EvaluationCriteria is { Count: > 0 }) ae["evaluation_criteria"] = Strings(testCase.EvaluationCriteria);
        if (testCase.GroundTruth is { } gt)
            ae["ground_truth"] = new JsonObject { ["name"] = gt.Name, ["arguments"] = EvalPortJson.ToObject(gt.Arguments) };
        if (testCase.Metadata is { Count: > 0 }) ae["metadata"] = EvalPortJson.ToObject(testCase.Metadata);
        if (ae.Count > 0)
            tc["metadata"] = new JsonObject { [EvalPortJson.MetadataKey] = ae };
        return tc;
    }

    /// <summary>The <c>ResultSet</c>: one <c>Result</c> per case run, plus a producer-side <c>summary</c>.</summary>
    public static JsonObject BuildResultSet(EvalPortRun run, EvalPortExportOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(run);
        options ??= EvalPortExportOptions.Default;

        var results = run.Cases.Select(c => ToResult(c, options)).ToList();

        var rs = new JsonObject
        {
            ["$schema"] = EvalPortJson.ResultSetSchema,
            ["version"] = EvalPortJson.SpecVersion,
            ["suite_id"] = run.SuiteId,
            ["run_id"] = run.RunId,
            ["started_at"] = EvalPortJson.Timestamp(options.Deterministic ? options.FixedTimestamp : run.StartedAt),
        };
        if (!options.Deterministic && run.CompletedAt is { } completed)
            rs["completed_at"] = EvalPortJson.Timestamp(completed);
        if (run.SubjectModel is not null)
            rs["provider"] = new JsonObject { ["model"] = run.SubjectModel };
        rs["runner"] = new JsonObject { ["name"] = "AgentEval", ["version"] = EvalPortJson.AgentEvalVersion };
        rs["results"] = new JsonArray(results.Select(r => (JsonNode?)r).ToArray());
        rs["summary"] = Summarize(results);

        var ae = new JsonObject
        {
            ["version"] = EvalPortJson.AgentEvalVersion,
            ["exporter"] = "EvalPort.AgentEval",
            ["exporter_version"] = EvalPortJson.PackageVersion,
            ["passed_semantics"] = "Result.passed is AgentEval's own verdict (AND of Score.Passed over the case's top-level results); " +
                                   "null-scored graders are not re-aggregated by this exporter",
        };
        if (options.EmitProposedVerdict) ae["proposed_verdict"] = "Discussion #49 (not in the spec)";
        if (options.EmitProposedJudgeIdentity) ae["proposed_judge_identity"] = "Discussion #118 (not in the spec)";
        if (run.Metadata is { Count: > 0 }) ae["run"] = EvalPortJson.ToObject(run.Metadata);
        rs["metadata"] = new JsonObject { [EvalPortJson.MetadataKey] = ae };
        return rs;
    }

    /// <summary>One EvalPort <c>Result</c> for one case run.</summary>
    public static JsonObject ToResult(EvalPortCaseRun caseRun, EvalPortExportOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(caseRun);
        options ??= EvalPortExportOptions.Default;

        var leaves = caseRun.Results.SelectMany(EvalResultMapping.Leaves).ToList();
        var graderResults = leaves.Select(l => EvalResultMapping.ToGraderResult(l, options)).ToList();

        // AgentEval's verdict. A case with no results and no error has nothing to say; it is not a pass.
        var passed = caseRun.Error is null
                     && caseRun.Results.Count > 0
                     && caseRun.Results.All(r => r.Score.Passed);

        var result = new JsonObject { ["test_case_id"] = RequireId(caseRun.Case) };
        if (caseRun.ActualOutput is not null) result["actual_output"] = caseRun.ActualOutput;
        if (caseRun.Attempt is { } attempt) result["attempt"] = attempt;
        result["grader_results"] = new JsonArray(graderResults.Select(g => (JsonNode?)g).ToArray());
        result["passed"] = passed;
        if (caseRun.DurationMs is { } ms) result["duration_ms"] = ms;
        if (caseRun.Error is { } error)
        {
            result["error"] = new JsonObject
            {
                ["type"] = "runner_error",
                ["message"] = error.Message,
                ["retryable"] = false,
            };
        }
        if (options.EmitProposedVerdict)
            result["verdict"] = ProposedVerdict(caseRun);

        result["metadata"] = ResultMetadata(caseRun, leaves, options);
        return result;
    }

    /// <summary>
    /// The <c>Result.verdict</c> this package would write under Discussion #49, derived from AgentEval's
    /// labels: <c>"passed"</c> when every top-level result is <c>"pass"</c>; <c>"failed"</c> when any is
    /// <c>"fail"</c> or <c>"warn"</c> (AgentEval calls <c>"warn"</c> a soft fail); otherwise
    /// <c>"unverified"</c> (<c>"error"</c>, <c>"skipped"</c>, <c>"inapplicable"</c>, a run error, or no results).
    /// </summary>
    public static string ProposedVerdict(EvalPortCaseRun caseRun)
    {
        ArgumentNullException.ThrowIfNull(caseRun);
        if (caseRun.Error is not null || caseRun.Results.Count == 0)
            return "unverified";
        var labels = caseRun.Results.Select(r => r.Score.Label).ToList();
        if (labels.Any(l => l is "fail" or "warn"))
            return "failed";
        if (labels.All(l => l == "pass"))
            return "passed";
        return "unverified";
    }

    /// <summary>
    /// Producer-side summary, with the same choices as the other examples in this repository: a row
    /// whose graders all have <c>null</c> scores (or that has no graders) is <c>skipped</c> and stays in
    /// the <c>pass_rate</c> denominator; <c>null</c> scores are left out of averages.
    /// </summary>
    public static JsonObject Summarize(IReadOnlyList<JsonObject> results)
    {
        ArgumentNullException.ThrowIfNull(results);
        var total = results.Count;
        var passed = results.Count(r => r["passed"]!.GetValue<bool>());
        var skipped = results.Count(r => !HasScoredGrader(r));
        var summary = new JsonObject
        {
            ["total"] = total,
            ["passed"] = passed,
            ["failed"] = total - passed - skipped,
            ["skipped"] = skipped,
            ["pass_rate"] = total == 0 ? 0.0 : EvalPortJson.Round4((double)passed / total),
        };

        var byGrader = new Dictionary<string, GraderTally>(StringComparer.Ordinal);
        var all = new List<double>();
        foreach (var r in results)
        {
            foreach (var g in r["grader_results"]!.AsArray())
            {
                var score = g!["score"];
                if (score is null) continue;
                var v = score.GetValue<double>();
                var id = (string)g["grader_id"]!;
                if (!byGrader.TryGetValue(id, out var slot))
                    byGrader[id] = slot = new GraderTally();
                if (g["passed"]!.GetValue<bool>()) slot.Passed++; else slot.Failed++;
                slot.Scores.Add(v);
                all.Add(v);
            }
        }
        if (all.Count > 0)
        {
            summary["avg_score"] = EvalPortJson.Round4(all.Average());
            var bg = new JsonObject();
            foreach (var (id, s) in byGrader)
                bg[id] = new JsonObject { ["passed"] = s.Passed, ["failed"] = s.Failed, ["avg_score"] = EvalPortJson.Round4(s.Scores.Average()) };
            summary["by_grader"] = bg;
        }
        return summary;

        static bool HasScoredGrader(JsonObject r) =>
            r["grader_results"]!.AsArray().Any(g => g!["score"] is not null);
    }

    private sealed class GraderTally
    {
        public int Passed;
        public int Failed;
        public List<double> Scores { get; } = new();
    }

    private static JsonObject ResultMetadata(EvalPortCaseRun caseRun, IReadOnlyList<EvalResult> leaves, EvalPortExportOptions options)
    {
        var metadata = new JsonObject();

        // Rule 6's recommended marker for a row none of whose graders produced a score.
        if (leaves.All(l => EvalResultMapping.ScoreOf(l.Score) is null))
            metadata["openeval"] = new JsonObject { ["aggregation_status"] = "unscored" };

        var ae = new JsonObject();

        // AgentEval's own verdict per top-level result, with the composite facts EvalPort has no slot for.
        var verdicts = new JsonArray();
        foreach (var r in caseRun.Results)
            verdicts.Add(VerdictNode(r, depth: 0, options));
        ae["verdicts"] = verdicts;

        // The three-way census over the leaves, computed by AgentEval's own extension method.
        var census = leaves.Select(l => l.Score).Census();
        ae["census"] = new JsonObject
        {
            ["measured"] = census.Measured,
            ["not_applicable"] = census.NotApplicable,
            ["not_measured"] = census.NotMeasured,
        };

        if (caseRun.Error is { } error)
            ae["error_type"] = error.GetType().FullName;

        metadata[EvalPortJson.MetadataKey] = ae;
        return metadata;
    }

    private static JsonObject VerdictNode(EvalResult r, int depth, EvalPortExportOptions options)
    {
        if (depth > EvalTreeLimits.MaxTreeWalkDepth)
            throw new InvalidOperationException($"EvalResult tree deeper than {EvalTreeLimits.MaxTreeWalkDepth}.");

        var node = new JsonObject
        {
            ["key"] = r.Metric.Key,
            ["label"] = r.Score.Label,
            ["passed"] = r.Score.Passed,
            ["severity"] = r.Score.Severity,
            ["measurement"] = EvalResultMapping.MeasurementOf(r.Score),
            ["value"] = r.Score.Value,
        };
        if (r.Score.Threshold is { } t) node["threshold"] = t;
        var summary = EvalResultMapping.ReasonOf(r);
        if (summary is not null) node["summary"] = summary;

        if (EvalResultMapping.IsComposite(r))
        {
            node["provenance_type"] = r.Provenance?.Type;
            if (r.Details?.AggregationStrategy is { } strategy) node["aggregation_strategy"] = strategy;
            var children = new JsonArray();
            foreach (var sub in r.Details!.SubResults!)
                children.Add(VerdictNode(sub, depth + 1, options));
            node["components"] = children;
        }
        return node;
    }

    private static string RequireId(DatasetTestCase testCase)
    {
        if (string.IsNullOrWhiteSpace(testCase.Id))
            throw new ArgumentException("A DatasetTestCase needs a non-empty Id to become an EvalPort test_case_id.", nameof(testCase));
        return testCase.Id;
    }

    private static JsonArray Strings(IEnumerable<string> values) =>
        new(values.Select(v => (JsonNode?)JsonValue.Create(v)).ToArray());
}
