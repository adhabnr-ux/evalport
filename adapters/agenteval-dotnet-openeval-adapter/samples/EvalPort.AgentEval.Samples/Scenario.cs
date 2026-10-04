// SPDX-License-Identifier: Apache-2.0

using AgentEval.Core;
using AgentEval.Evals;
using AgentEval.Models;
using AgentEval.Testing;
using EvalPort.AgentEval;

namespace EvalPort.AgentEval.Samples;

/// <summary>
/// Runs AgentEval's real <see cref="CompositeEval"/>, <see cref="AtomicLlmEval"/> and
/// <see cref="ChatClientEvaluator"/> on seven constructed cases, with a canned chat client standing
/// in for the judge model. No LLM, network or API key. What is real is the framework: the code that
/// decides what a case looks like after one of its two graders could not measure it is AgentEval's own.
/// </summary>
/// <remarks>
/// The seven cases, graded by a composite of <c>exact</c> (an <see cref="AtomicCodeEval"/>) and
/// <c>judge</c> (an <see cref="AtomicLlmEval"/>):
/// <list type="table">
/// <item><term>q1</term><description>right answer; judge replies with a passing verdict.</description></item>
/// <item><term>q2</term><description>right answer; judge replies with no JSON twice, so <see cref="ChatClientEvaluator"/> reports <c>EvaluationFailed</c> and <see cref="AtomicLlmEval"/> writes its <c>"error"</c> label.</description></item>
/// <item><term>q3</term><description>wrong answer; judge errors as in q2.</description></item>
/// <item><term>q4</term><description>right answer; <c>exact</c> is told to skip (<see cref="EvalResult.Skipped"/>) and the judge errors: nothing is measured.</description></item>
/// <item><term>q5</term><description>no ground truth, so <c>exact</c> is <see cref="AtomicCodeEval.NotApplicable"/>; judge passes.</description></item>
/// <item><term>q6</term><description>the chat client throws (a transport failure): the exception leaves <see cref="CompositeEval.EvaluateAsync"/> and the case has no result.</description></item>
/// <item><term>q7</term><description>right answer; judge is skipped by its wrapper (<see cref="EvalResult.Skipped"/>, a budget filter).</description></item>
/// </list>
/// Two variants differ in one flag: whether the <c>judge</c> component is <c>Required</c>.
/// </remarks>
public static class Scenario
{
    /// <summary>The question every case asks; the id is appended so a grader can tell the cases apart.</summary>
    public const string Question = "What is 2+2?";

    /// <summary>The judge model name written into provenance (no such model is called).</summary>
    public const string JudgeModel = "canned-judge";

    /// <summary>A passing judge reply in the shape <see cref="ChatClientEvaluator"/> parses.</summary>
    public const string JudgePassReply =
        """{"overallScore": 100, "summary": "The answer is correct.", "criteriaResults": [{"criterion": "answers the question correctly", "met": true, "explanation": "4 is right"}]}""";

    /// <summary>A reply with no JSON in it; the evaluator retries once and then reports failure.</summary>
    public const string JudgeGarbageReply = "I am unable to help with that request.";

    /// <summary>The seven cases.</summary>
    public static IReadOnlyList<CaseSpec> Cases { get; } = new[]
    {
        new CaseSpec("q1", "4", "4", JudgeBehavior.Pass, ExactBehavior.Run, HasGroundTruth: true),
        new CaseSpec("q2", "4", "4", JudgeBehavior.Garbage, ExactBehavior.Run, HasGroundTruth: true),
        new CaseSpec("q3", "4", "5", JudgeBehavior.Garbage, ExactBehavior.Run, HasGroundTruth: true),
        new CaseSpec("q4", "4", "4", JudgeBehavior.Garbage, ExactBehavior.Skip, HasGroundTruth: true),
        new CaseSpec("q5", "4", "4", JudgeBehavior.Pass, ExactBehavior.Run, HasGroundTruth: false),
        new CaseSpec("q6", "4", "4", JudgeBehavior.Throw, ExactBehavior.Run, HasGroundTruth: true),
        new CaseSpec("q7", "4", "4", JudgeBehavior.Skipped, ExactBehavior.Run, HasGroundTruth: true),
    };

    /// <summary>How the canned judge behaves for a case.</summary>
    public enum JudgeBehavior
    {
        /// <summary>Replies with <see cref="JudgePassReply"/>.</summary>
        Pass,
        /// <summary>Replies with <see cref="JudgeGarbageReply"/> on both the call and the retry.</summary>
        Garbage,
        /// <summary>The chat client throws <see cref="InvalidOperationException"/>.</summary>
        Throw,
        /// <summary>The judge eval is wrapped and returns <see cref="EvalResult.Skipped"/> without calling the judge.</summary>
        Skipped,
    }

    /// <summary>How the deterministic grader behaves for a case.</summary>
    public enum ExactBehavior
    {
        /// <summary>Compares the response with the ground truth.</summary>
        Run,
        /// <summary>Returns <see cref="EvalResult.Skipped"/>.</summary>
        Skip,
    }

    /// <summary>One constructed case.</summary>
    public sealed record CaseSpec(string Id, string Expected, string Actual, JudgeBehavior Judge, ExactBehavior Exact, bool HasGroundTruth);

    /// <summary>The outcome of running one case: the composite result, or the exception that escaped.</summary>
    public sealed record CaseOutcome(CaseSpec Spec, EvalResult? Result, Exception? Error);

    /// <summary>Run all cases with the given <c>Required</c> flag on the judge component.</summary>
    public static async Task<IReadOnlyList<CaseOutcome>> RunAsync(bool judgeRequired, CancellationToken ct = default)
    {
        var outcomes = new List<CaseOutcome>(Cases.Count);
        foreach (var spec in Cases)
            outcomes.Add(await RunCaseAsync(spec, judgeRequired, ct).ConfigureAwait(false));
        return outcomes;
    }

    /// <summary>Run one case through a freshly built composite.</summary>
    public static async Task<CaseOutcome> RunCaseAsync(CaseSpec spec, bool judgeRequired, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        var composite = BuildComposite(spec, judgeRequired);
        var input = new EvalInput(
            Query: $"{Question} ({spec.Id})",
            Response: spec.Actual,
            GroundTruth: spec.HasGroundTruth ? spec.Expected : null)
        { CaseId = spec.Id };
        try
        {
            var result = await composite.EvaluateAsync(input, ct).ConfigureAwait(false);
            return new CaseOutcome(spec, result, null);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return new CaseOutcome(spec, null, ex);
        }
    }

    /// <summary>The composite for a case: <c>exact</c> (always required) and <c>judge</c>, weighted-sum, severity-driven verdict (no threshold).</summary>
    public static CompositeEval BuildComposite(CaseSpec spec, bool judgeRequired)
    {
        ArgumentNullException.ThrowIfNull(spec);
        IEval exact = new ExactAnswerEval();
        if (spec.Exact == ExactBehavior.Skip)
            exact = new SkippingEval(exact, "exact grader disabled for this case");

        IEval judge = BuildJudge(spec.Judge);
        if (spec.Judge == JudgeBehavior.Skipped)
            judge = new SkippingEval(judge, "judge skipped by budget filter");

        return new CompositeEval(
            key: "answer_quality",
            name: "Answer quality",
            category: "quality",
            version: "1.0.0",
            components: new[]
            {
                new EvalComponent(exact, Weight: 1.0, Required: true),
                new EvalComponent(judge, Weight: 1.0, Required: judgeRequired),
            },
            aggregation: WeightedSumAggregation.Instance,
            threshold: null);
    }

    private static AtomicLlmEval BuildJudge(JudgeBehavior behavior)
    {
        var chat = behavior switch
        {
            JudgeBehavior.Pass => new FakeChatClient(JudgePassReply, JudgePassReply),
            JudgeBehavior.Garbage => new FakeChatClient(JudgeGarbageReply, JudgeGarbageReply),
            JudgeBehavior.Throw => new FakeChatClient { ThrowOnNextCall = true, ThrowMessage = "judge endpoint unreachable" },
            JudgeBehavior.Skipped => new FakeChatClient(JudgePassReply),
            _ => throw new ArgumentOutOfRangeException(nameof(behavior)),
        };
        var evaluator = new ChatClientEvaluator(chat);
        return new AtomicLlmEval(
            evaluator,
            key: "judge",
            name: "LLM judge",
            category: "quality",
            version: "1.0.0",
            criteria: new[] { "answers the question correctly" },
            passThreshold: 0.70,
            judgeModel: JudgeModel);
    }

    /// <summary>Convert the outcomes to the builder's input, so the documents can be written.</summary>
    public static EvalPortRun ToRun(IReadOnlyList<CaseOutcome> outcomes, bool judgeRequired, DateTimeOffset startedAt)
    {
        ArgumentNullException.ThrowIfNull(outcomes);
        var variant = judgeRequired ? "judge_required" : "judge_optional";
        var cases = outcomes.Select(o => new EvalPortCaseRun(
            Case: new DatasetTestCase
            {
                Id = o.Spec.Id,
                Input = $"{Question} ({o.Spec.Id})",
                ExpectedOutput = o.Spec.HasGroundTruth ? o.Spec.Expected : null,
                Metadata = new Dictionary<string, object?>
                {
                    ["judge_behavior"] = o.Spec.Judge.ToString(),
                    ["exact_behavior"] = o.Spec.Exact.ToString(),
                },
            },
            ActualOutput: o.Spec.Actual,
            Results: o.Result is null ? Array.Empty<EvalResult>() : new[] { o.Result },
            Error: o.Error)).ToList();

        // q6 has no result at all, so the Suite cannot learn its graders from it. The Suite builder
        // rejects that on purpose; here the case is described by a skipped placeholder from the same
        // composite definition, so the Suite still lists the two graders it was meant to run.
        for (var i = 0; i < cases.Count; i++)
        {
            if (cases[i].Results.Count == 0)
            {
                var composite = BuildComposite(outcomes[i].Spec, judgeRequired);
                var placeholder = new[]
                {
                    EvalResult.Skipped(composite.Components[0].Eval, "not run: the case errored before grading"),
                    EvalResult.Skipped(composite.Components[1].Eval, "not run: the case errored before grading"),
                };
                cases[i] = cases[i] with { Results = placeholder };
            }
        }

        return new EvalPortRun
        {
            SuiteId = "suite_agenteval_errored_graders",
            SuiteName = "Errored graders in AgentEval",
            SuiteDescription =
                "Seven cases graded by a CompositeEval of an exact-answer AtomicCodeEval and an AtomicLlmEval whose " +
                "judge is a canned chat client. The judge replies with no JSON on q2-q4 (AgentEval's \"error\" label), " +
                "throws on q6 (the exception escapes the composite), and is skipped on q7; exact is skipped on q4 and " +
                "inapplicable on q5. Variant: " + variant + ".",
            RunId = $"run_agenteval_errored_graders_{variant}",
            Cases = cases,
            StartedAt = startedAt,
            Metadata = new Dictionary<string, object?>
            {
                ["variant"] = variant,
                ["judge_required"] = judgeRequired,
                ["aggregation"] = "WeightedSum, threshold null (severity-driven verdict), MinimumMeasuredShare 0.5",
                ["llm"] = "none (FakeChatClient with canned replies, no network)",
            },
        };
    }

    /// <summary>A deterministic grader: 1.0 when the response equals the ground truth, else 0.0 with severity <c>high</c>.</summary>
    public sealed class ExactAnswerEval : AtomicCodeEval
    {
        /// <summary>Create the grader.</summary>
        public ExactAnswerEval() : base("exact", "Exact answer", "quality", "1.0.0") { }

        /// <inheritdoc/>
        protected override EvalResult Evaluate(EvalInput input)
        {
            if (input.GroundTruth is null)
                return NotApplicable("no ground truth supplied for this case");
            var ok = string.Equals(input.Response?.Trim(), input.GroundTruth.Trim(), StringComparison.Ordinal);
            return Build(ok ? 1.0 : 0.0, ok, ok ? "none" : "high");
        }
    }

    /// <summary>Wraps an eval and returns <see cref="EvalResult.Skipped"/> in its place, as AgentEval's own guards do.</summary>
    public sealed class SkippingEval : IEval
    {
        private readonly IEval _inner;
        private readonly string _reason;

        /// <summary>Create the wrapper.</summary>
        public SkippingEval(IEval inner, string reason)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            _reason = reason ?? throw new ArgumentNullException(nameof(reason));
        }

        /// <inheritdoc/>
        public string Key => _inner.Key;
        /// <inheritdoc/>
        public string Name => _inner.Name;
        /// <inheritdoc/>
        public string Category => _inner.Category;
        /// <inheritdoc/>
        public string Version => _inner.Version;

        /// <inheritdoc/>
        public Task<EvalResult> EvaluateAsync(EvalInput input, CancellationToken ct = default) =>
            Task.FromResult(EvalResult.Skipped(_inner, _reason));
    }
}
