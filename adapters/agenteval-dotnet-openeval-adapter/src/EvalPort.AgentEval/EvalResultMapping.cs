// SPDX-License-Identifier: Apache-2.0

using System.Text.Json.Nodes;
using AgentEval.Evals;
using AgentEval.Evals.Meta;

namespace EvalPort.AgentEval;

/// <summary>
/// How one AgentEval <see cref="EvalResult"/> (an atomic eval, or a leaf of a composite) becomes one
/// EvalPort <c>GraderResult</c>, and how its <see cref="MeasurementState"/> is written.
/// </summary>
/// <remarks>
/// <para>
/// EvalPort Validation Rule 6: a grader that produced no score is <c>"score": null</c> with
/// <c>"passed": false</c> ("not verified"); it is excluded from aggregation and MUST NOT be read as a
/// scored failure. AgentEval has the same idea one level down, in <see cref="MeasurementState"/>
/// (<c>NotApplicable</c>, <c>NotMeasured</c>) and in the <c>"skipped"</c> / <c>"error"</c> /
/// <c>"inapplicable"</c> labels. The two are joined here through AgentEval's own predicate,
/// <see cref="EvalScoreExtensions.CountsTowardAggregate"/>: a score that AgentEval would not count
/// is written as <c>null</c>, and nothing else is.
/// </para>
/// <para>
/// What a <c>null</c> loses is <em>which</em> kind of non-measurement it was. AgentEval keeps the
/// three states apart on purpose (different owners, different fixes), so the state is written
/// under <c>metadata.agenteval.measurement</c> next to the null, with AgentEval's label and severity.
/// </para>
/// </remarks>
public static class EvalResultMapping
{
    /// <summary>
    /// The EvalPort score for an AgentEval score: the value when AgentEval counts it toward an
    /// aggregate, otherwise <c>null</c> (Rule 6 "not verified").
    /// </summary>
    public static double? ScoreOf(EvalScore score)
    {
        ArgumentNullException.ThrowIfNull(score);
        return score.CountsTowardAggregate() ? EvalPortJson.Clamp01(score.Value) : null;
    }

    /// <summary>
    /// AgentEval's measurement state for a score, as the string its own <c>eval-result.schema.json</c>
    /// uses: <c>"measured"</c>, <c>"notApplicable"</c> or <c>"notMeasured"</c>. Uses
    /// <see cref="EvalScoreExtensions.CensusBucket"/>, so a <c>"skipped"</c> or <c>"error"</c> label
    /// is <c>"notMeasured"</c> even though its <see cref="EvalScore.Measurement"/> is <c>Measured</c>.
    /// </summary>
    public static string MeasurementOf(EvalScore score)
    {
        ArgumentNullException.ThrowIfNull(score);
        return score.CensusBucket() switch
        {
            MeasurementState.Measured => "measured",
            MeasurementState.NotApplicable => "notApplicable",
            MeasurementState.NotMeasured => "notMeasured",
            var other => other.ToString(),
        };
    }

    /// <summary>
    /// The EvalPort grader <c>type</c> for an AgentEval result: <c>agenteval_</c> plus the
    /// provenance type (<c>atomic-llm</c>, <c>atomic-code</c>, <c>atomic-decision</c>,
    /// <c>composite</c>, <c>multi-judge-adjudicated</c>) with dashes as underscores. These are
    /// framework-specific types in EvalPort's sense (validated like <c>custom</c>), not the
    /// well-known <c>llm_judge</c> or <c>code</c>, because an AgentEval result does not carry the
    /// prompt text or source those types require. A result whose provenance type is
    /// <c>"skipped"</c> says nothing about the grader's kind, so it maps to <c>agenteval_eval</c>.
    /// </summary>
    public static string GraderTypeOf(EvalResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        var type = result.Provenance?.Type;
        if (string.IsNullOrWhiteSpace(type) || type == "skipped")
            return "agenteval_eval";
        return "agenteval_" + type.Replace('-', '_');
    }

    /// <summary>
    /// True when the result is a composite: it has sub-results. Only leaves become <c>GraderResult</c>s;
    /// a composite's own score and label are written on the <c>Result</c> (see <see cref="EvalPortDocuments"/>).
    /// </summary>
    public static bool IsComposite(EvalResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        return result.Details?.SubResults is { Count: > 0 };
    }

    /// <summary>Depth-first list of the leaves under a result (the result itself when it has no sub-results).</summary>
    public static IReadOnlyList<EvalResult> Leaves(EvalResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        var list = new List<EvalResult>();
        Collect(result, list, 0);
        return list;

        static void Collect(EvalResult r, List<EvalResult> into, int depth)
        {
            if (depth > EvalTreeLimits.MaxTreeWalkDepth)
                throw new InvalidOperationException($"EvalResult tree deeper than {EvalTreeLimits.MaxTreeWalkDepth}.");
            if (!IsComposite(r))
            {
                into.Add(r);
                return;
            }
            foreach (var sub in r.Details!.SubResults!)
                Collect(sub, into, depth + 1);
        }
    }

    /// <summary>
    /// One EvalPort <c>GraderResult</c> for one leaf. <c>grader_id</c> is the eval's <c>Key</c>.
    /// </summary>
    /// <param name="leaf">An atomic result. A composite is rejected; flatten it with <see cref="Leaves"/> first.</param>
    /// <param name="options">Export options (judge identity self-report).</param>
    public static JsonObject ToGraderResult(EvalResult leaf, EvalPortExportOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(leaf);
        if (IsComposite(leaf))
            throw new ArgumentException("A composite result is not a grader; export its leaves.", nameof(leaf));
        options ??= EvalPortExportOptions.Default;

        var score = leaf.Score;
        var value = ScoreOf(score);
        var passed = value is null ? false : score.Passed;
        if (value is null && score.Passed)
        {
            // Cannot happen with AgentEval's own types (EvalScore refuses Passed=true when the
            // measurement is not Measured, and "skipped"/"error" are built with Passed=false), but
            // the EvalPort rule is stated here so a future AgentEval type cannot violate it silently.
            passed = false;
        }

        var gr = new JsonObject
        {
            ["grader_id"] = leaf.Metric.Key,
            ["type"] = GraderTypeOf(leaf),
            ["score"] = value,
            ["passed"] = passed,
        };

        var reason = ReasonOf(leaf);
        if (reason is not null)
            gr["reason"] = reason;

        gr["metadata"] = GraderMetadata(leaf, options);
        return gr;
    }

    /// <summary>
    /// The human-readable reason for a leaf: AgentEval's <c>Details.Summary</c>, else the
    /// recommendations, else the evidence messages. <c>null</c> when the result carries none.
    /// </summary>
    public static string? ReasonOf(EvalResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        var d = result.Details;
        if (d is null)
            return null;
        if (!string.IsNullOrWhiteSpace(d.Summary))
            return d.Summary;
        if (d.Recommendations is { Count: > 0 })
            return string.Join(" | ", d.Recommendations.Where(r => !string.IsNullOrWhiteSpace(r)));
        if (d.Evidence is { Count: > 0 })
            return string.Join(" | ", d.Evidence.Select(e => $"{e.Source}:{e.Reference}: {e.Message}"));
        return null;
    }

    private static JsonObject GraderMetadata(EvalResult leaf, EvalPortExportOptions options)
    {
        var score = leaf.Score;
        var ae = new JsonObject
        {
            ["name"] = leaf.Metric.Name,
            ["category"] = leaf.Metric.Category,
            ["version"] = leaf.Metric.Version,
            ["label"] = score.Label,
            ["severity"] = score.Severity,
            ["measurement"] = MeasurementOf(score),
            // The raw value is kept even when the EvalPort score is null, so nothing is lost; readers
            // of the typed field see null, readers of this key see what AgentEval wrote (0.0 by convention).
            ["value"] = score.Value,
        };
        if (score.Ordinal is { } ordinal)
            ae["ordinal"] = ordinal;
        if (score.Threshold is { } threshold)
            ae["threshold"] = threshold;
        if (score.Confidence is { } confidence)
            ae["confidence"] = confidence;
        var evaluatedAt = options.Deterministic ? options.FixedTimestamp : leaf.EvaluatedAt;
        ae["evaluated_at"] = EvalPortJson.Timestamp(evaluatedAt);

        if (leaf.Provenance is { } p)
        {
            var prov = new JsonObject { ["type"] = p.Type };
            if (p.JudgeModel is not null) prov["judge_model"] = p.JudgeModel;
            if (p.PromptId is not null) prov["prompt_id"] = p.PromptId;
            // AgentEval's PromptHash is the first 16 hex characters of a SHA-256 over its judge-input
            // framing version, the prompt material and the criteria. It is NOT a SHA-256 of the prompt,
            // so it is not written as the proposed openeval.judge.prompt_sha256 (Discussion #118).
            if (p.PromptHash is not null) prov["prompt_hash"] = p.PromptHash;
            if (p.TokensUsed is { } tokens) prov["tokens_used"] = tokens;
            prov["estimated_cost"] = p.EstimatedCost;
            prov["cache_hit"] = p.CacheHit;
            ae["provenance"] = prov;
        }

        if (leaf.Details is { } d)
        {
            if (d.Dimensions is { Count: > 0 })
                ae["dimensions"] = EvalPortJson.ToObject(d.Dimensions);
            if (d.Evidence is { Count: > 0 })
            {
                var evidence = new JsonArray();
                foreach (var e in d.Evidence)
                    evidence.Add(new JsonObject { ["source"] = e.Source, ["reference"] = e.Reference, ["message"] = e.Message });
                ae["evidence"] = evidence;
            }
            if (d.Recommendations is { Count: > 0 })
                ae["recommendations"] = new JsonArray(d.Recommendations.Select(r => (JsonNode?)JsonValue.Create(r)).ToArray());
        }

        var metadata = new JsonObject { [EvalPortJson.MetadataKey] = ae };

        if (options.EmitProposedJudgeIdentity && leaf.Provenance?.JudgeModel is { } judgeModel)
        {
            // Discussion #118 (PROPOSED, not in the spec): a runner MAY self-report the judge it observed
            // under metadata.openeval.judge. AgentEval records the judge model it was configured with;
            // whether that is the model that answered is not something AgentEval observes, so only
            // `model` is written and no fingerprint is invented.
            metadata["openeval"] = new JsonObject
            {
                ["judge"] = new JsonObject
                {
                    ["model"] = judgeModel,
                    ["observed_at"] = EvalPortJson.Timestamp(evaluatedAt),
                },
            };
        }

        return metadata;
    }
}
