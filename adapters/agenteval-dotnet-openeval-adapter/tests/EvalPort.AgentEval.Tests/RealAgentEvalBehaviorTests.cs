// SPDX-License-Identifier: Apache-2.0

using AgentEval.Evals;
using AgentEval.Evals.Meta;
using EvalPort.AgentEval.Samples;

namespace EvalPort.AgentEval.Tests;

/// <summary>
/// What AgentEval 0.43.0-beta itself does when one of two graders cannot measure a case. These tests
/// run the real <c>CompositeEval</c>, <c>AtomicLlmEval</c> and <c>ChatClientEvaluator</c>; only the chat
/// client is canned. If a newer AgentEval changes any of this, these fail first, which is the signal to
/// re-read the README. That already happened once: 0.42.0-beta → 0.43.0-beta (AgentEvalHQ/AgentEval#279,
/// which cites this package's report in #203) flipped three of them; the old expectations are kept in
/// comments so the change stays visible.
/// </summary>
public class RealAgentEvalBehaviorTests
{
    private static readonly Lazy<Task<IReadOnlyList<Scenario.CaseOutcome>>> s_required = new(() => Scenario.RunAsync(judgeRequired: true));
    private static readonly Lazy<Task<IReadOnlyList<Scenario.CaseOutcome>>> s_optional = new(() => Scenario.RunAsync(judgeRequired: false));

    private static async Task<Scenario.CaseOutcome> Case(bool judgeRequired, string id)
    {
        var outcomes = await (judgeRequired ? s_required : s_optional).Value;
        return outcomes.Single(o => o.Spec.Id == id);
    }

    private static EvalResult Leaf(EvalResult composite, string key) =>
        composite.Details!.SubResults!.Single(s => s.Metric.Key == key);

    [Fact]
    public void AgentEval_PackageVersion_IsThePinnedOne()
    {
        Assert.Equal("0.43.0-beta", EvalPortJson.AgentEvalVersion);
    }

    [Fact]
    public async Task Judge_WithNoJsonReply_ProducesTheErrorLabel_NotAZero()
    {
        var q2 = await Case(judgeRequired: true, "q2");
        var judge = Leaf(q2.Result!, "judge");
        Assert.Equal("error", judge.Score.Label);
        Assert.False(judge.Score.Passed);
        Assert.Equal("none", judge.Score.Severity);
        Assert.Equal(0.0, judge.Score.Value);
        // The enum still says Measured; the label is what carries the state. CensusBucket reads the label.
        Assert.Equal(MeasurementState.Measured, judge.Score.Measurement);
        Assert.Equal(MeasurementState.NotMeasured, judge.Score.CensusBucket());
        Assert.False(judge.Score.CountsTowardAggregate());
        Assert.Equal("atomic-llm", judge.Provenance.Type);
        Assert.Equal(Scenario.JudgeModel, judge.Provenance.JudgeModel);
        Assert.NotNull(judge.Provenance.PromptHash);
    }

    [Fact]
    public async Task RequiredJudgeErrored_OtherGraderPassed_CompositeIsError_NotPass()
    {
        var q2 = await Case(judgeRequired: true, "q2");
        Assert.Equal("pass", Leaf(q2.Result!, "exact").Score.Label);
        Assert.Equal("error", q2.Result!.Score.Label);
        Assert.False(q2.Result.Score.Passed);
        // The aggregate score itself is 1.0: the errored leaf is excluded from the weighted sum.
        Assert.Equal(1.0, q2.Result.Score.Value);
        Assert.Contains("required component errored", q2.Result.Details.Summary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task OptionalJudgeErrored_OtherGraderPassed_CompositeIsPass()
    {
        var q2 = await Case(judgeRequired: false, "q2");
        Assert.Equal("pass", q2.Result!.Score.Label);
        Assert.True(q2.Result.Score.Passed);
        Assert.Equal(1.0, q2.Result.Score.Value);
        // AgentEval says how much of the pass was measured, in the result itself.
        Assert.Contains("Measured 1 of 2", q2.Result.Details.Summary);
    }

    [Fact]
    public async Task RequiredJudgeErrored_OtherGraderFailed_CompositeIsFail_TheMeasuredFailureDecides()
    {
        // 0.42.0-beta: "error" (the required judge's error was the verdict and the measured failure was
        // not reported). 0.43.0-beta: a required component's error is the verdict unless the composite
        // would fail even if the errored parts had passed; a measured high-severity failure is such a case.
        var q3 = await Case(judgeRequired: true, "q3");
        Assert.Equal("fail", Leaf(q3.Result!, "exact").Score.Label);
        Assert.Equal("high", Leaf(q3.Result!, "exact").Score.Severity);
        Assert.Equal("error", Leaf(q3.Result!, "judge").Score.Label);
        Assert.Equal("fail", q3.Result!.Score.Label);
        Assert.False(q3.Result.Score.Passed);
        Assert.Equal(MeasurementState.Measured, q3.Result.Score.CensusBucket());
        Assert.Contains("Decided by severity", q3.Result.Details.Summary);
    }

    [Fact]
    public async Task OptionalJudgeErrored_OtherGraderFailed_CompositeIsFail()
    {
        var q3 = await Case(judgeRequired: false, "q3");
        Assert.Equal("fail", q3.Result!.Score.Label);
        Assert.False(q3.Result.Score.Passed);
    }

    [Theory]
    [InlineData(true, "error")]
    [InlineData(false, "skipped")]
    public async Task NothingMeasured_OneErrored_CompositeHasNoVerdict(bool judgeRequired, string expectedLabel)
    {
        // 0.42.0-beta: "error" in both variants. 0.43.0-beta: an optional component's error no longer
        // decides anything, so with nothing measured the optional variant is "skipped" instead. Either
        // way there is no pass/fail verdict and nothing was measured.
        var q4 = await Case(judgeRequired, "q4");
        Assert.Equal("skipped", Leaf(q4.Result!, "exact").Score.Label);
        Assert.Equal("error", Leaf(q4.Result!, "judge").Score.Label);
        Assert.Equal(expectedLabel, q4.Result!.Score.Label);
        Assert.False(q4.Result.Score.Passed);
        Assert.Equal(0.0, q4.Result.Score.Value);
        Assert.Equal(MeasurementState.NotMeasured, q4.Result.Score.CensusBucket());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task InapplicableExact_JudgePassed_CompositeIsPass(bool judgeRequired)
    {
        var q5 = await Case(judgeRequired, "q5");
        var exact = Leaf(q5.Result!, "exact");
        Assert.Equal("inapplicable", exact.Score.Label);
        Assert.Equal(MeasurementState.NotApplicable, exact.Score.Measurement);
        Assert.False(exact.Score.CountsTowardAggregate());
        Assert.Equal("pass", q5.Result!.Score.Label);
        Assert.True(q5.Result.Score.Passed);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ChatClientThrows_TheExceptionLeavesTheComposite_NoResultExists(bool judgeRequired)
    {
        var q6 = await Case(judgeRequired, "q6");
        Assert.Null(q6.Result);
        Assert.IsType<InvalidOperationException>(q6.Error);
        Assert.Equal("judge endpoint unreachable", q6.Error!.Message);
    }

    [Fact]
    public async Task SkippedRequiredJudge_HoldsBackThePass_AsWarnNotMeasured()
    {
        // 0.42.0-beta: "pass" (only a required component with the "error" label blocked the verdict; a
        // required one that returned EvalResult.Skipped did not). This package reported that in
        // AgentEvalHQ/AgentEval#203; 0.43.0-beta holds the pass back as "warn" and marks the composite's
        // own score notMeasured, which is what a parent (and this package) reads rather than the label.
        var q7 = await Case(judgeRequired: true, "q7");
        var judge = Leaf(q7.Result!, "judge");
        Assert.Equal("skipped", judge.Score.Label);
        Assert.Equal("skipped", judge.Provenance.Type);
        Assert.Equal("warn", q7.Result!.Score.Label);
        Assert.False(q7.Result.Score.Passed);
        Assert.Equal(MeasurementState.NotMeasured, q7.Result.Score.CensusBucket());
        Assert.False(q7.Result.Score.CountsTowardAggregate());
        Assert.Contains("did not run", q7.Result.Details.Summary);
    }

    [Fact]
    public async Task SkippedOptionalJudge_CompositeIsPass_AndSaysHowMuchWasMeasured()
    {
        // One of two measured is exactly MinimumMeasuredShare (0.5), which is not below it, so the pass is
        // not downgraded to "warn".
        var q7 = await Case(judgeRequired: false, "q7");
        Assert.Equal("skipped", Leaf(q7.Result!, "judge").Score.Label);
        Assert.Equal("pass", q7.Result!.Score.Label);
        Assert.True(q7.Result.Score.Passed);
        Assert.Contains("Measured 1 of 2", q7.Result.Details.Summary);
    }

    [Fact]
    public async Task OneOfThreeMeasured_PassBecomesWarn()
    {
        // The under-coverage rule needs a third component to show: 1 of 3 < 0.5.
        var spec = Scenario.Cases.Single(c => c.Id == "q7");
        var composite = Scenario.BuildComposite(spec, judgeRequired: true);
        var skippedExtra = new Scenario.SkippingEval(new Scenario.ExactAnswerEval(), "third component skipped");
        var three = new CompositeEval("answer_quality", "Answer quality", "quality", "1.0.0",
            new[] { composite.Components[0], composite.Components[1], new EvalComponent(skippedExtra, 1.0, Required: false) },
            WeightedSumAggregation.Instance);
        var result = await three.EvaluateAsync(new EvalInput($"{Scenario.Question} (q7)", Response: "4", GroundTruth: "4"));
        Assert.Equal("warn", result.Score.Label);
        Assert.False(result.Score.Passed);
        Assert.Equal(1.0, result.Score.Value);
    }

    [Fact]
    public async Task WithoutAThreshold_TheVerdictIsReadFromSeverity_NotFromPassFlags()
    {
        // A leaf that FAILS with severity "low" does not fail a threshold-less composite; the same leaf with
        // severity "high" does. This is why the sample's ExactAnswerEval fails with "high".
        async Task<string> LabelFor(string failureSeverity)
        {
            var leaf = new FixedLeaf("fixed", 0.0, passed: false, failureSeverity);
            var composite = new CompositeEval("c", "C", "quality", "1.0.0",
                new[] { new EvalComponent(leaf, 1.0, Required: true) }, WeightedSumAggregation.Instance);
            return (await composite.EvaluateAsync(new EvalInput("q", Response: "x"))).Score.Label;
        }
        Assert.Equal("pass", await LabelFor("low"));
        Assert.Equal("warn", await LabelFor("medium"));
        Assert.Equal("fail", await LabelFor("high"));

        // With a threshold, the score decides instead.
        var leafLow = new FixedLeaf("fixed", 0.0, passed: false, "low");
        var thresholded = new CompositeEval("c", "C", "quality", "1.0.0",
            new[] { new EvalComponent(leafLow, 1.0, Required: true) }, WeightedSumAggregation.Instance, threshold: 0.5);
        Assert.Equal("fail", (await thresholded.EvaluateAsync(new EvalInput("q", Response: "x"))).Score.Label);
    }

    private sealed class FixedLeaf : AtomicCodeEval
    {
        private readonly double _value;
        private readonly bool _passed;
        private readonly string _severity;
        public FixedLeaf(string key, double value, bool passed, string severity) : base(key, key, "quality", "1.0.0")
        {
            _value = value; _passed = passed; _severity = severity;
        }
        protected override EvalResult Evaluate(EvalInput input) => Build(_value, _passed, _severity);
    }

    [Fact]
    public void EvalScore_RefusesPassedTrue_WhenNotMeasured()
    {
        var ex = Assert.ThrowsAny<ArgumentException>(() =>
            new EvalScore(1.0, null, "pass", true, null, "none", null) { Measurement = MeasurementState.NotApplicable });
        Assert.Contains("NotApplicable", ex.Message);
    }

    [Fact]
    public void Skipped_AndNotApplicable_AreNotPassedAndDoNotCount()
    {
        var skipped = EvalResult.Skipped(new Scenario.ExactAnswerEval(), "because");
        Assert.False(skipped.Score.Passed);
        Assert.False(skipped.Score.CountsTowardAggregate());
        Assert.Equal(MeasurementState.NotMeasured, skipped.Score.CensusBucket());

        var na = EvalScore.NotApplicable();
        Assert.False(na.Passed);
        Assert.False(na.CountsTowardAggregate());
        Assert.Equal(MeasurementState.NotApplicable, na.CensusBucket());
    }
}
