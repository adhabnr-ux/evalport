// SPDX-License-Identifier: Apache-2.0

using System.Text.Json.Nodes;
using AgentEval.Evals;
using AgentEval.Models;
using EvalPort.AgentEval.Samples;

namespace EvalPort.AgentEval.Tests;

/// <summary>The EvalResult-tree path: <see cref="EvalPortDocuments"/> on the scenario's real outcomes.</summary>
public class DocumentsTests
{
    private static readonly DateTimeOffset s_started = new(2026, 10, 4, 0, 0, 0, TimeSpan.Zero);
    private static readonly Lazy<Task<(JsonObject Suite, JsonObject ResultSet, IReadOnlyList<Scenario.CaseOutcome> Outcomes)>> s_required = new(() => Build(true, EvalPortExportOptions.Default));
    private static readonly Lazy<Task<(JsonObject Suite, JsonObject ResultSet, IReadOnlyList<Scenario.CaseOutcome> Outcomes)>> s_optional = new(() => Build(false, EvalPortExportOptions.Default));

    private static async Task<(JsonObject, JsonObject, IReadOnlyList<Scenario.CaseOutcome>)> Build(bool judgeRequired, EvalPortExportOptions options)
    {
        var outcomes = await Scenario.RunAsync(judgeRequired);
        var run = Scenario.ToRun(outcomes, judgeRequired, s_started);
        var (suite, rs) = EvalPortDocuments.Build(run, options);
        return ((JsonObject)EvalPortSchemas.Reparse(suite), (JsonObject)EvalPortSchemas.Reparse(rs), outcomes);
    }

    private static JsonObject Result(JsonObject rs, string id) =>
        rs["results"]!.AsArray().Cast<JsonObject>().Single(r => (string)r["test_case_id"]! == id);

    private static JsonObject Grader(JsonObject result, string id) =>
        result["grader_results"]!.AsArray().Cast<JsonObject>().Single(g => (string)g["grader_id"]! == id);

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task BothDocuments_AreSchemaValid(bool judgeRequired)
    {
        var (suite, rs, _) = await (judgeRequired ? s_required : s_optional).Value;
        EvalPortSchemas.Instance.AssertValidSuite(suite);
        EvalPortSchemas.Instance.AssertValidResultSet(rs);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Rule6_EveryUnmeasuredLeaf_IsNullScoreAndNotPassed(bool judgeRequired)
    {
        var (_, rs, outcomes) = await (judgeRequired ? s_required : s_optional).Value;
        var nullCount = 0;
        foreach (var o in outcomes.Where(o => o.Result is not null))
        {
            var result = Result(rs, o.Spec.Id);
            foreach (var leaf in EvalResultMapping.Leaves(o.Result!))
            {
                var g = Grader(result, leaf.Metric.Key);
                if (leaf.Score.CountsTowardAggregate())
                {
                    Assert.Equal(leaf.Score.Value, g["score"]!.GetValue<double>());
                    Assert.Equal(leaf.Score.Passed, g["passed"]!.GetValue<bool>());
                    Assert.Equal("measured", (string)g["metadata"]!["agenteval"]!["measurement"]!);
                }
                else
                {
                    nullCount++;
                    Assert.Null(g["score"]);
                    Assert.False(g["passed"]!.GetValue<bool>());
                    Assert.NotEqual("measured", (string)g["metadata"]!["agenteval"]!["measurement"]!);
                    Assert.Equal(leaf.Score.Label, (string)g["metadata"]!["agenteval"]!["label"]!);
                }
            }
        }
        Assert.True(nullCount >= 5, $"expected the scenario to produce several null-scored graders, got {nullCount}");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ResultPassed_IsAgentEvalsOwnVerdict(bool judgeRequired)
    {
        var (_, rs, outcomes) = await (judgeRequired ? s_required : s_optional).Value;
        foreach (var o in outcomes)
        {
            var result = Result(rs, o.Spec.Id);
            var expected = o.Result is not null && o.Result.Score.Passed;
            Assert.Equal(expected, result["passed"]!.GetValue<bool>());
            var verdicts = result["metadata"]!["agenteval"]!["verdicts"]!.AsArray();
            if (o.Result is null)
            {
                // q6: the sample substitutes two skipped placeholders (one per component) so the Suite can list the graders.
                Assert.Equal(2, verdicts.Count);
                Assert.All(verdicts, v => Assert.Equal("skipped", (string)v!["label"]!));
            }
            else
            {
                Assert.Single(verdicts);
                Assert.Equal(o.Result.Score.Label, (string)verdicts[0]!["label"]!);
                Assert.Equal("WeightedSum", (string)verdicts[0]!["aggregation_strategy"]!);
                Assert.Equal(2, verdicts[0]!["components"]!.AsArray().Count);
            }
        }
    }

    [Fact]
    public async Task TheMixedRow_IsPassedTrueWithOptionalJudge_AndFalseWithRequiredJudge()
    {
        var (_, required, _) = await s_required.Value;
        var (_, optional, _) = await s_optional.Value;
        var q2Required = Result(required, "q2");
        var q2Optional = Result(optional, "q2");
        // Same leaves in both: exact scored 1.0, judge produced nothing.
        foreach (var r in new[] { q2Required, q2Optional })
        {
            Assert.Equal(1.0, Grader(r, "exact")["score"]!.GetValue<double>());
            Assert.Null(Grader(r, "judge")["score"]);
            Assert.Null(r["error"]);
            Assert.Null(r["metadata"]!["openeval"]); // not unscored: one grader did score
        }
        Assert.False(q2Required["passed"]!.GetValue<bool>());
        Assert.True(q2Optional["passed"]!.GetValue<bool>());
        Assert.Equal(1, (int)q2Required["metadata"]!["agenteval"]!["census"]!["measured"]!);
        Assert.Equal(1, (int)q2Required["metadata"]!["agenteval"]!["census"]!["not_measured"]!);
    }

    [Fact]
    public async Task PassRate_DiffersByAFactorOfRequired()
    {
        var (_, required, _) = await s_required.Value;
        var (_, optional, _) = await s_optional.Value;
        Assert.Equal(7, (int)required["summary"]!["total"]!);
        Assert.Equal(3, (int)required["summary"]!["passed"]!);
        Assert.Equal(2, (int)required["summary"]!["failed"]!);
        Assert.Equal(2, (int)required["summary"]!["skipped"]!);
        Assert.Equal(0.4286, required["summary"]!["pass_rate"]!.GetValue<double>());
        Assert.Equal(4, (int)optional["summary"]!["passed"]!);
        Assert.Equal(1, (int)optional["summary"]!["failed"]!);
        Assert.Equal(0.5714, optional["summary"]!["pass_rate"]!.GetValue<double>());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task AllNullRows_CarryTheUnscoredMarker(bool judgeRequired)
    {
        var (_, rs, _) = await (judgeRequired ? s_required : s_optional).Value;
        foreach (var id in new[] { "q4", "q6" })
        {
            var r = Result(rs, id);
            Assert.All(r["grader_results"]!.AsArray(), g => Assert.Null(g!["score"]));
            Assert.False(r["passed"]!.GetValue<bool>());
            Assert.Equal("unscored", (string)r["metadata"]!["openeval"]!["aggregation_status"]!);
        }
        Assert.Null(Result(rs, "q1")["metadata"]!["openeval"]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task TheThrownCase_HasARunnerError_AndNoVerdictFromAgentEval(bool judgeRequired)
    {
        var (_, rs, _) = await (judgeRequired ? s_required : s_optional).Value;
        var q6 = Result(rs, "q6");
        Assert.Equal("runner_error", (string)q6["error"]!["type"]!);
        Assert.Equal("judge endpoint unreachable", (string)q6["error"]!["message"]!);
        Assert.False(q6["error"]!["retryable"]!.GetValue<bool>());
        Assert.Equal("System.InvalidOperationException", (string)q6["metadata"]!["agenteval"]!["error_type"]!);
        // The placeholders the sample adds so the Suite can list the graders are visibly "not run".
        Assert.All(q6["grader_results"]!.AsArray(), g => Assert.Contains("not run", (string)g!["reason"]!));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Suite_ListsTwoGradersWithFrameworkTypes_AndSevenCases(bool judgeRequired)
    {
        var (suite, _, _) = await (judgeRequired ? s_required : s_optional).Value;
        var graders = suite["graders"]!.AsArray().Cast<JsonObject>().ToDictionary(g => (string)g["id"]!, g => g);
        Assert.Equal(new[] { "exact", "judge" }, graders.Keys.OrderBy(k => k));
        Assert.Equal("agenteval_atomic_code", (string)graders["exact"]["type"]!);
        Assert.Equal("agenteval_atomic_llm", (string)graders["judge"]["type"]!);
        Assert.Equal("exact", (string)graders["exact"]["params"]!["handler"]!);
        var cases = suite["test_cases"]!.AsArray().Cast<JsonObject>().ToList();
        Assert.Equal(7, cases.Count);
        Assert.All(cases, c => Assert.Equal(new[] { "exact", "judge" }, c["graders"]!.AsArray().Select(g => (string)g!).OrderBy(k => k)));
        Assert.Null(cases.Single(c => (string)c["id"]! == "q5")["expected_output"]);
        Assert.Equal("4", (string)cases.Single(c => (string)c["id"]! == "q1")["expected_output"]!);
    }

    [Fact]
    public async Task JudgeProvenance_IsKeptOnTheGrader_AndPromptHashIsNotCalledASha256()
    {
        var (_, rs, _) = await s_required.Value;
        var prov = Grader(Result(rs, "q1"), "judge")["metadata"]!["agenteval"]!["provenance"]!;
        Assert.Equal("atomic-llm", (string)prov["type"]!);
        Assert.Equal(Scenario.JudgeModel, (string)prov["judge_model"]!);
        Assert.Equal("agenteval.judge.default-system.v1", (string)prov["prompt_id"]!);
        Assert.Equal(16, ((string)prov["prompt_hash"]!).Length);
        Assert.Null(Grader(Result(rs, "q1"), "judge")["metadata"]!["openeval"]);
        Assert.Null(Grader(Result(rs, "q1"), "exact")["metadata"]!["agenteval"]!["provenance"]!["judge_model"]);
    }

    [Fact]
    public async Task ProposedJudgeIdentity_WritesOnlyTheModel_OnlyOnJudges_AndStaysSchemaValid()
    {
        var (suite, rs, _) = await Build(true, new EvalPortExportOptions { EmitProposedJudgeIdentity = true, Deterministic = true });
        EvalPortSchemas.Instance.AssertValidSuite(suite);
        EvalPortSchemas.Instance.AssertValidResultSet(rs);
        var judge = Grader(Result(rs, "q1"), "judge")["metadata"]!["openeval"]!["judge"]!;
        Assert.Equal(Scenario.JudgeModel, (string)judge["model"]!);
        Assert.Equal("2026-10-04T00:00:00Z", (string)judge["observed_at"]!);
        Assert.Null(judge["prompt_sha256"]);
        Assert.Null(judge["fingerprint"]);
        Assert.Null(Grader(Result(rs, "q1"), "exact")["metadata"]!["openeval"]);
    }

    [Fact]
    public async Task ProposedVerdict_IsRejectedByTheSchemaOnMain_AndFollowsAgentEvalsLabels()
    {
        var (_, required, _) = await Build(true, new EvalPortExportOptions { EmitProposedVerdict = true, Deterministic = true });
        var (_, optional, _) = await Build(false, new EvalPortExportOptions { EmitProposedVerdict = true, Deterministic = true });

        // Result has additionalProperties: false on main, so the field is a schema error there (as intended: it is a proposal).
        var errors = EvalPortSchemas.Instance.Errors(EvalPortSchemas.Instance.ResultSet, required);
        Assert.Equal(7, errors.Count);
        Assert.All(errors, e => Assert.Matches(@"^/results/\d+/verdict: ", e)); // the only thing the schema objects to

        string V(JsonObject rs, string id) => (string)Result(rs, id)["verdict"]!;
        Assert.Equal("passed", V(required, "q1"));
        Assert.Equal("unverified", V(required, "q2")); // composite "error": no verdict, passed false
        Assert.Equal("unverified", V(required, "q3")); // a measured failure, but AgentEval's own label is "error"
        Assert.Equal("unverified", V(required, "q4"));
        Assert.Equal("passed", V(required, "q5"));
        Assert.Equal("unverified", V(required, "q6"));
        Assert.Equal("passed", V(required, "q7"));

        Assert.Equal("passed", V(optional, "q2")); // passed: true with a null-scored grader
        Assert.Equal("failed", V(optional, "q3"));

        // The #83 rule "unverified requires passed: false" holds for every unverified row here.
        foreach (var rs in new[] { required, optional })
        {
            foreach (var r in rs["results"]!.AsArray().Cast<JsonObject>())
            {
                var verdict = (string)r["verdict"]!;
                var passed = r["passed"]!.GetValue<bool>();
                Assert.Equal(verdict == "passed", passed);
                if (r["error"] is not null) Assert.Equal("unverified", verdict);
            }
        }
    }

    [Fact]
    public void Builder_RefusesACaseWithNoGraders_AndAnEmptyInput()
    {
        var noResults = new EvalPortRun
        {
            SuiteId = "s", RunId = "r",
            Cases = new[] { new EvalPortCaseRun(new DatasetTestCase { Id = "a", Input = "x" }, null, Array.Empty<EvalResult>()) },
        };
        Assert.Throws<InvalidOperationException>(() => EvalPortDocuments.BuildSuite(noResults));

        var emptyInput = new EvalPortRun
        {
            SuiteId = "s", RunId = "r",
            Cases = new[] { new EvalPortCaseRun(new DatasetTestCase { Id = "a", Input = "" }, null,
                new[] { EvalResult.Skipped(new Scenario.ExactAnswerEval(), "x") }) },
        };
        Assert.Throws<ArgumentException>(() => EvalPortDocuments.BuildSuite(emptyInput));

        // A ResultSet for the no-results case is still writable: passed false, no graders, unscored.
        var rs = EvalPortDocuments.BuildResultSet(noResults, new EvalPortExportOptions { Deterministic = true });
        var r = rs["results"]![0]!;
        Assert.False(r["passed"]!.GetValue<bool>());
        Assert.Empty(r["grader_results"]!.AsArray());
        Assert.Equal("unscored", (string)r["metadata"]!["openeval"]!["aggregation_status"]!);
        EvalPortSchemas.Instance.AssertValidResultSet(EvalPortSchemas.Reparse(rs));
    }

    [Fact]
    public void ToGraderResult_RefusesAComposite()
    {
        var composite = new EvalResult(
            new EvalMetadata("c", "C", "quality", "1.0.0"),
            new EvalScore(1, null, "pass", true, null, "none", null),
            new EvalDetails(null, null, null, new[] { EvalResult.Skipped(new Scenario.ExactAnswerEval(), "x") }, "WeightedSum"),
            new EvalProvenance("composite", null, null, null, null, 0, false),
            DateTimeOffset.UtcNow);
        Assert.Throws<ArgumentException>(() => EvalResultMapping.ToGraderResult(composite));
        Assert.Single(EvalResultMapping.Leaves(composite));
    }
}
