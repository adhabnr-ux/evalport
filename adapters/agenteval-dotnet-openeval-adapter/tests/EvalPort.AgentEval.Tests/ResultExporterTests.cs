// SPDX-License-Identifier: Apache-2.0

using System.Text;
using System.Text.Json.Nodes;
using AgentEval.Core;
using AgentEval.Models;
using AgentEval.Output;

namespace EvalPort.AgentEval.Tests;

/// <summary>The flat path: <see cref="EvalPortResultExporter"/> on an <see cref="EvaluationReport"/>.</summary>
public class ResultExporterTests
{
    private static EvaluationReport SampleReport() => new()
    {
        RunId = "run_42",
        Name = "suite_flat",
        StartTime = new DateTimeOffset(2026, 10, 4, 0, 0, 0, TimeSpan.Zero),
        EndTime = new DateTimeOffset(2026, 10, 4, 0, 0, 5, TimeSpan.Zero),
        TotalTests = 5, PassedTests = 1, FailedTests = 3, SkippedTests = 1, OverallScore = 48.0,
        Agent = new AgentInfo { Name = "demo-agent", Model = "canned-model", Version = "1.2.3" },
        Metadata = { ["ci"] = "true" },
        TestResults =
        {
            new TestResultSummary { Name = "t_pass", Score = 92.5, Passed = true, DurationMs = 12, Output = "4",
                MetricScores = { ["llm_relevance"] = 90, ["code_exact"] = 100 },
                Assertions = { AssertionResult.Pass("answer is 4", "matched") } },
            new TestResultSummary { Name = "t_fail", Score = 20, Passed = false, DurationMs = 8, Output = "5",
                Error = "Expected 4 but got 5 | Suggestions: check arithmetic",
                Assertions = { AssertionResult.Fail("answer is 4", "got 5") } },
            new TestResultSummary { Name = "t_inconclusive", Score = 70, Passed = false, Output = "4?",
                Assertions = { AssertionResult.Undecidable("answer is 4", "judge returned no verdict") } },
            new TestResultSummary { Name = "t_skipped", Score = 0, Passed = false, Skipped = true, Error = "no API key" },
            new TestResultSummary { Name = "t_crash", Score = 0, Passed = false,
                Error = "Object reference not set to an instance of an object.",
                StackTrace = "   at Demo.Agent.Run() in Agent.cs:line 12" },
        },
    };

    private static JsonObject Result(JsonObject rs, string id) =>
        rs["results"]!.AsArray().Cast<JsonObject>().Single(r => (string)r["test_case_id"]! == id);

    [Fact]
    public void Exporter_IdentifiesItself()
    {
        var exporter = new EvalPortResultExporter();
        Assert.Equal(ExportFormat.Json, exporter.Format);
        Assert.Equal("evalport", exporter.FormatName);
        Assert.Equal(".evalport.json", exporter.FileExtension);
        Assert.Equal("application/json", exporter.ContentType);
    }

    [Fact]
    public void Report_BecomesASchemaValidResultSet()
    {
        var rs = (JsonObject)EvalPortSchemas.Reparse(new EvalPortResultExporter().ToResultSet(SampleReport()));
        EvalPortSchemas.Instance.AssertValidResultSet(rs);
        Assert.Equal("suite_flat", (string)rs["suite_id"]!);
        Assert.Equal("run_42", (string)rs["run_id"]!);
        Assert.Equal("2026-10-04T00:00:00Z", (string)rs["started_at"]!);
        Assert.Equal("2026-10-04T00:00:05Z", (string)rs["completed_at"]!);
        Assert.Equal("canned-model", (string)rs["provider"]!["model"]!);
        Assert.Equal("AgentEval", (string)rs["runner"]!["name"]!);
        Assert.Equal("0.43.0-beta", (string)rs["runner"]!["version"]!);
        Assert.Equal(5, (int)rs["summary"]!["total"]!);
        Assert.Equal(1, (int)rs["summary"]!["passed"]!);
        Assert.Equal(1, (int)rs["summary"]!["skipped"]!);
        Assert.Equal(3, (int)rs["summary"]!["failed"]!);
        Assert.Equal(5000, (long)rs["summary"]!["duration_ms"]!);
        Assert.Equal(48.0, rs["metadata"]!["agenteval"]!["report"]!["overall_score_0_100"]!.GetValue<double>());
        Assert.Equal("true", (string)rs["metadata"]!["agenteval"]!["report_metadata"]!["ci"]!);
    }

    [Fact]
    public void TestScore_IsDividedBy100_AndMetricScoresAreNotGraders()
    {
        var rs = new EvalPortResultExporter().ToResultSet(SampleReport());
        var t = Result(rs, "t_pass");
        var graders = t["grader_results"]!.AsArray();
        Assert.Equal(2, graders.Count);
        Assert.Equal("agenteval_test", (string)graders[0]!["grader_id"]!);
        Assert.Equal(0.925, graders[0]!["score"]!.GetValue<double>());
        Assert.True(graders[0]!["passed"]!.GetValue<bool>());
        Assert.Equal("assertion_1", (string)graders[1]!["grader_id"]!);
        Assert.Equal(1.0, graders[1]!["score"]!.GetValue<double>());
        Assert.Equal("answer is 4", (string)graders[1]!["metadata"]!["agenteval"]!["assertion"]!);
        Assert.Equal(90.0, t["metadata"]!["agenteval"]!["metric_scores_0_100"]!["llm_relevance"]!.GetValue<double>());
        Assert.Equal(12, (long)t["duration_ms"]!);
        Assert.Equal("4", (string)t["actual_output"]!);
    }

    [Fact]
    public void FailureExplanation_IsTheReason_NotAnError()
    {
        var t = Result(new EvalPortResultExporter().ToResultSet(SampleReport()), "t_fail");
        Assert.Null(t["error"]);
        Assert.False(t["passed"]!.GetValue<bool>());
        Assert.StartsWith("Expected 4 but got 5", (string)t["grader_results"]![0]!["reason"]!);
        Assert.Equal(0.0, t["grader_results"]![1]!["score"]!.GetValue<double>());
    }

    [Fact]
    public void InconclusiveAssertion_IsNullScore_NotPassed()
    {
        var t = Result(new EvalPortResultExporter().ToResultSet(SampleReport()), "t_inconclusive");
        var assertion = t["grader_results"]![1]!;
        Assert.Null(assertion["score"]);
        Assert.False(assertion["passed"]!.GetValue<bool>());
        Assert.Equal("Inconclusive", (string)assertion["metadata"]!["agenteval"]!["outcome"]!);
        Assert.Equal("judge returned no verdict", (string)assertion["reason"]!);
        // The test's own grader did score, so the row is not "unscored".
        Assert.Null(t["metadata"]!["openeval"]);
    }

    [Fact]
    public void SkippedTest_HasNoGraders_IsNotPassed_AndIsMarkedUnscored()
    {
        var t = Result(new EvalPortResultExporter().ToResultSet(SampleReport()), "t_skipped");
        Assert.Empty(t["grader_results"]!.AsArray());
        Assert.False(t["passed"]!.GetValue<bool>());
        Assert.Null(t["error"]);
        Assert.Equal("unscored", (string)t["metadata"]!["openeval"]!["aggregation_status"]!);
        Assert.True(t["metadata"]!["agenteval"]!["skipped"]!.GetValue<bool>());
        Assert.Equal("no API key", (string)t["metadata"]!["agenteval"]!["error_text"]!);
    }

    [Fact]
    public void SkippedButPassedTest_IsStillNotPassed_Rule6()
    {
        var report = new EvaluationReport { TestResults = { new TestResultSummary { Name = "odd", Passed = true, Skipped = true } } };
        var t = Result(new EvalPortResultExporter().ToResultSet(report), "odd");
        Assert.False(t["passed"]!.GetValue<bool>());
        Assert.True(t["metadata"]!["agenteval"]!["native_passed"]!.GetValue<bool>());
    }

    [Fact]
    public void ErrorWithStackTrace_IsARunnerError()
    {
        var t = Result(new EvalPortResultExporter().ToResultSet(SampleReport()), "t_crash");
        Assert.Equal("runner_error", (string)t["error"]!["type"]!);
        Assert.StartsWith("Object reference", (string)t["error"]!["message"]!);
        Assert.Null(t["grader_results"]![0]!["reason"]);
        Assert.Contains("Agent.cs", (string)t["metadata"]!["agenteval"]!["stack_trace"]!);
    }

    [Fact]
    public void RepeatedTestNames_GetAttemptNumbers()
    {
        var report = new EvaluationReport
        {
            TestResults =
            {
                new TestResultSummary { Name = "same", Score = 100, Passed = true },
                new TestResultSummary { Name = "same", Score = 0, Passed = false },
                new TestResultSummary { Name = "other", Score = 100, Passed = true },
            },
        };
        var rs = (JsonObject)EvalPortSchemas.Reparse(new EvalPortResultExporter().ToResultSet(report));
        EvalPortSchemas.Instance.AssertValidResultSet(rs);
        var results = rs["results"]!.AsArray();
        Assert.Equal(1, (int)results[0]!["attempt"]!);
        Assert.Equal(2, (int)results[1]!["attempt"]!);
        Assert.Null(results[2]!["attempt"]);
    }

    [Fact]
    public void ProposedVerdict_FollowsSkippedAndExceptions()
    {
        var rs = new EvalPortResultExporter(new EvalPortExportOptions { EmitProposedVerdict = true }).ToResultSet(SampleReport());
        Assert.Equal("passed", (string)Result(rs, "t_pass")["verdict"]!);
        Assert.Equal("failed", (string)Result(rs, "t_fail")["verdict"]!);
        Assert.Equal("failed", (string)Result(rs, "t_inconclusive")["verdict"]!);
        Assert.Equal("unverified", (string)Result(rs, "t_skipped")["verdict"]!);
        Assert.Equal("unverified", (string)Result(rs, "t_crash")["verdict"]!);
    }

    [Fact]
    public async Task ExportAsync_WritesTheSameBytesAsExportToString_Utf8NoBom()
    {
        var exporter = new EvalPortResultExporter(new EvalPortExportOptions { Deterministic = true });
        using var stream = new MemoryStream();
        await exporter.ExportAsync(SampleReport(), stream);
        var bytes = stream.ToArray();
        Assert.Equal(Encoding.UTF8.GetBytes(exporter.ExportToString(SampleReport())), bytes);
        Assert.NotEqual(0xEF, bytes[0]);
        Assert.EndsWith("}\n", Encoding.UTF8.GetString(bytes));
    }

    [Fact]
    public void EmptyName_IsRejected()
    {
        var report = new EvaluationReport { TestResults = { new TestResultSummary { Name = "" } } };
        Assert.Throws<ArgumentException>(() => new EvalPortResultExporter().ToResultSet(report));
    }
}
