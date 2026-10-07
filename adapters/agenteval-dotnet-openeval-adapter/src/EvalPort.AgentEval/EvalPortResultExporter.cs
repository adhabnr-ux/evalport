// SPDX-License-Identifier: Apache-2.0

using System.Text;
using System.Text.Json.Nodes;
using AgentEval.Core;
using AgentEval.Models;
using AgentEval.Output;

namespace EvalPort.AgentEval;

/// <summary>
/// AgentEval <see cref="IResultExporter"/> that writes an <see cref="EvaluationReport"/> as an EvalPort
/// <c>ResultSet</c>. Registered under the format name <c>"evalport"</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>What this interface can and cannot carry.</b> <see cref="IResultExporter.ExportAsync"/> receives
/// AgentEval's flat <see cref="EvaluationReport"/>: per test a name, a 0–100 score, <c>Passed</c>,
/// <c>Skipped</c>, an error string, an output string, metric scores (0–100, no per-metric pass flag) and
/// assertion results. It does not receive <c>EvalResult</c>, so it has no <c>MeasurementState</c>, no
/// <c>"error"</c> / <c>"inapplicable"</c> labels, no composite tree and no judge provenance. Everything
/// below is therefore what the report says, and only that. For the result model, use
/// <see cref="EvalPortDocuments"/>.
/// </para>
/// <para>
/// Mapping, per <see cref="TestResultSummary"/>:
/// <list type="bullet">
/// <item><c>test_case_id</c> is <c>Name</c>; when a name repeats, every occurrence gets a 1-based <c>attempt</c>.</item>
/// <item>One grader <c>agenteval_test</c> carries the test's own <c>Score / 100</c> and <c>Passed</c>.</item>
/// <item>One grader <c>assertion_&lt;n&gt;</c> per <see cref="AssertionResult"/>: score 1 or 0, or
/// <c>null</c> with <c>passed: false</c> for <see cref="AssertionOutcome.Inconclusive"/> (Rule 6).</item>
/// <item><c>Skipped</c>: <c>grader_results: []</c>, <c>passed: false</c>,
/// <c>metadata.openeval.aggregation_status: "unscored"</c>. The report's own <c>Passed</c> is kept in metadata.</item>
/// <item><c>Error</c> with a <c>StackTrace</c> is an exception: <c>Result.error</c> of type <c>runner_error</c>.
/// <c>Error</c> without one is AgentEval's failure explanation (its <c>BuildErrorMessage</c> writes
/// <c>FailureReport.WhyItFailed</c> there too), so it becomes the test grader's <c>reason</c>. The report
/// does not distinguish the two any other way.</item>
/// <item><c>MetricScores</c> have no pass flag, so they are not graders; they are kept raw under
/// <c>metadata.agenteval.metric_scores_0_100</c>.</item>
/// </list>
/// </para>
/// </remarks>
public sealed class EvalPortResultExporter : IResultExporter
{
    /// <summary>The registry name of this exporter.</summary>
    public const string Name = "evalport";

    private readonly EvalPortExportOptions _options;

    /// <summary>Create an exporter with default options.</summary>
    public EvalPortResultExporter() : this(EvalPortExportOptions.Default) { }

    /// <summary>Create an exporter with the given options.</summary>
    public EvalPortResultExporter(EvalPortExportOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    /// <remarks><see cref="ExportFormat"/> is a closed enum; <c>Json</c> is the closest built-in, as AgentEval's docs prescribe.</remarks>
    public ExportFormat Format => ExportFormat.Json;

    /// <inheritdoc/>
    public string FormatName => Name;

    /// <inheritdoc/>
    public string FileExtension => ".evalport.json";

    /// <inheritdoc/>
    public string ContentType => "application/json";

    /// <inheritdoc/>
    public async Task ExportAsync(EvaluationReport report, Stream output, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(report);
        ArgumentNullException.ThrowIfNull(output);
        var text = EvalPortJson.ToJsonText(ToResultSet(report));
        var bytes = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false).GetBytes(text);
        await output.WriteAsync(bytes, ct).ConfigureAwait(false);
        await output.FlushAsync(ct).ConfigureAwait(false);
    }

    /// <summary>Export to a string (same bytes as <see cref="ExportAsync"/>).</summary>
    public string ExportToString(EvaluationReport report)
    {
        ArgumentNullException.ThrowIfNull(report);
        return EvalPortJson.ToJsonText(ToResultSet(report));
    }

    /// <summary>The <c>ResultSet</c> document for a report, as a JSON object.</summary>
    public JsonObject ToResultSet(EvaluationReport report)
    {
        ArgumentNullException.ThrowIfNull(report);

        var nameCounts = report.TestResults
            .GroupBy(t => t.Name, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal);
        var attemptSoFar = new Dictionary<string, int>(StringComparer.Ordinal);

        var results = new List<JsonObject>(report.TestResults.Count);
        foreach (var test in report.TestResults)
        {
            int? attempt = null;
            if (nameCounts[test.Name] > 1)
            {
                attemptSoFar[test.Name] = attemptSoFar.TryGetValue(test.Name, out var n) ? n + 1 : 1;
                attempt = attemptSoFar[test.Name];
            }
            results.Add(ToResult(test, attempt));
        }

        var rs = new JsonObject
        {
            ["$schema"] = EvalPortJson.ResultSetSchema,
            ["version"] = EvalPortJson.SpecVersion,
            ["suite_id"] = string.IsNullOrWhiteSpace(report.Name) ? "agenteval" : report.Name,
            ["run_id"] = string.IsNullOrWhiteSpace(report.RunId) ? "agenteval-run" : report.RunId,
            ["started_at"] = EvalPortJson.Timestamp(_options.Deterministic ? _options.FixedTimestamp : report.StartTime),
        };
        if (!_options.Deterministic && report.EndTime > report.StartTime)
            rs["completed_at"] = EvalPortJson.Timestamp(report.EndTime);
        if (report.Agent?.Model is { } model)
            rs["provider"] = new JsonObject { ["model"] = model };
        rs["runner"] = new JsonObject { ["name"] = "AgentEval", ["version"] = EvalPortJson.AgentEvalVersion };
        rs["results"] = new JsonArray(results.Select(r => (JsonNode?)r).ToArray());

        // The summary is recomputed from the rows written, so it always agrees with them; the report's own
        // counters are kept verbatim next to it.
        var summary = EvalPortDocuments.Summarize(results);
        if (!_options.Deterministic && report.Duration > TimeSpan.Zero)
            summary["duration_ms"] = (long)report.Duration.TotalMilliseconds;
        rs["summary"] = summary;

        var ae = new JsonObject
        {
            ["version"] = EvalPortJson.AgentEvalVersion,
            ["exporter"] = "EvalPort.AgentEval",
            ["exporter_version"] = EvalPortJson.PackageVersion,
            ["source"] = "EvaluationReport (IResultExporter); no MeasurementState, labels, composite tree or provenance are available on this path",
            ["report"] = new JsonObject
            {
                ["run_id"] = report.RunId,
                ["total_tests"] = report.TotalTests,
                ["passed_tests"] = report.PassedTests,
                ["failed_tests"] = report.FailedTests,
                ["skipped_tests"] = report.SkippedTests,
                ["pass_rate_percent"] = report.PassRate,
                ["overall_score_0_100"] = report.OverallScore,
            },
        };
        if (report.Agent is { } agent)
        {
            var a = new JsonObject();
            if (agent.Name is not null) a["name"] = agent.Name;
            if (agent.Model is not null) a["model"] = agent.Model;
            if (agent.Version is not null) a["version"] = agent.Version;
            if (agent.Endpoint is not null) a["endpoint"] = agent.Endpoint;
            ae["agent"] = a;
        }
        if (report.Metadata is { Count: > 0 })
            ae["report_metadata"] = EvalPortJson.ToObject(report.Metadata);
        rs["metadata"] = new JsonObject
        {
            [EvalPortJson.MetadataKey] = ae,
            // PROPOSED (evalport Discussion #49, alternative B), NOT in the spec on main; see
            // EvalPortDocuments.ToResultSet for why `producer` is the honest declaration here.
            ["openeval"] = new JsonObject { ["aggregation"] = new JsonObject { ["strategy"] = "producer" } },
        };
        return rs;
    }

    /// <summary>One EvalPort <c>Result</c> for one <see cref="TestResultSummary"/>.</summary>
    public JsonObject ToResult(TestResultSummary test, int? attempt = null)
    {
        ArgumentNullException.ThrowIfNull(test);
        if (string.IsNullOrWhiteSpace(test.Name))
            throw new ArgumentException("A TestResultSummary needs a non-empty Name to become an EvalPort test_case_id.", nameof(test));

        var result = new JsonObject { ["test_case_id"] = test.Name };
        if (test.Output is not null) result["actual_output"] = test.Output;
        if (attempt is { } a) result["attempt"] = a;

        var isException = test.Error is not null && !string.IsNullOrEmpty(test.StackTrace);
        var graders = new List<JsonObject>();
        if (!test.Skipped)
        {
            var testGrader = new JsonObject
            {
                ["grader_id"] = "agenteval_test",
                ["type"] = "agenteval_test",
                ["score"] = EvalPortJson.Clamp01(test.Score / 100.0),
                ["passed"] = test.Passed,
            };
            if (test.Error is not null && !isException)
                testGrader["reason"] = test.Error;
            testGrader["metadata"] = new JsonObject
            {
                [EvalPortJson.MetadataKey] = new JsonObject { ["score_0_100"] = test.Score },
            };
            graders.Add(testGrader);

            for (var i = 0; i < test.Assertions.Count; i++)
                graders.Add(ToAssertionGrader(test.Assertions[i], i));
        }

        result["grader_results"] = new JsonArray(graders.Select(g => (JsonNode?)g).ToArray());
        // A skipped test has no graded outcome; Rule 6 forbids reading that as a pass.
        result["passed"] = !test.Skipped && test.Passed;
        if (test.DurationMs > 0) result["duration_ms"] = test.DurationMs;
        if (isException)
        {
            result["error"] = new JsonObject
            {
                ["type"] = "runner_error",
                ["message"] = test.Error!,
                ["retryable"] = false,
            };
        }
        if (_options.EmitProposedVerdict)
            result["verdict"] = test.Skipped || isException ? "unverified" : test.Passed ? "passed" : "failed";

        var metadata = new JsonObject();
        if (graders.All(g => g["score"] is null))
            metadata["openeval"] = new JsonObject { ["aggregation_status"] = "unscored" };
        var ae = new JsonObject
        {
            ["native_passed"] = test.Passed,
            ["skipped"] = test.Skipped,
            ["score_0_100"] = test.Score,
        };
        if (test.Category is not null) ae["category"] = test.Category;
        if (test.MetricScores is { Count: > 0 }) ae["metric_scores_0_100"] = EvalPortJson.ToObject(test.MetricScores);
        if (isException && test.StackTrace is not null) ae["stack_trace"] = test.StackTrace;
        if (test.Skipped && test.Error is not null) ae["error_text"] = test.Error;
        metadata[EvalPortJson.MetadataKey] = ae;
        result["metadata"] = metadata;
        return result;
    }

    private static JsonObject ToAssertionGrader(AssertionResult assertion, int index)
    {
        var inconclusive = assertion.Outcome == AssertionOutcome.Inconclusive;
        var gr = new JsonObject
        {
            ["grader_id"] = $"assertion_{index + 1}",
            ["type"] = "agenteval_assertion",
            ["score"] = inconclusive ? null : (assertion.Passed ? 1.0 : 0.0),
            ["passed"] = !inconclusive && assertion.Passed,
        };
        var reason = assertion.Message ?? (inconclusive ? "assertion could not decide" : null);
        if (reason is not null) gr["reason"] = reason;
        gr["metadata"] = new JsonObject
        {
            [EvalPortJson.MetadataKey] = new JsonObject
            {
                ["assertion"] = assertion.Assertion,
                ["outcome"] = assertion.Outcome.ToString(),
            },
        };
        return gr;
    }
}
