// SPDX-License-Identifier: Apache-2.0

namespace EvalPort.AgentEval;

/// <summary>
/// Options for the documents this package writes. The defaults produce documents that validate
/// against the EvalPort 1.0.0 schemas and SDK validators on <c>main</c>; the two <c>EmitProposed*</c>
/// flags add fields from open RFCs that are <b>not in the spec</b>.
/// </summary>
public sealed record EvalPortExportOptions
{
    /// <summary>The defaults: spec-conformant output, no proposed fields.</summary>
    public static EvalPortExportOptions Default { get; } = new();

    /// <summary>
    /// Add <c>Result.verdict</c> (<c>"passed"</c> / <c>"failed"</c> / <c>"unverified"</c>) as proposed
    /// in EvalPort Discussion #49 (reference PR #83). <b>Not in the spec.</b> A document with this
    /// field is rejected by the schemas on <c>main</c> (<c>additionalProperties: false</c> on
    /// <c>Result</c>); it exists to test the proposal against real AgentEval output.
    /// </summary>
    public bool EmitProposedVerdict { get; init; }

    /// <summary>
    /// Add <c>metadata.openeval.judge.model</c> on a grader whose provenance names a judge model, as
    /// proposed in EvalPort Discussion #118 (reference PR #119). <b>Not in the spec</b>, but it is a
    /// <c>metadata</c> key, so the documents stay valid on <c>main</c>.
    /// </summary>
    public bool EmitProposedJudgeIdentity { get; init; }

    /// <summary>
    /// Fixed timestamps and no <c>completed_at</c>, for checked-in sample output and diffs. Also
    /// overrides every leaf's <c>evaluated_at</c> with <see cref="FixedTimestamp"/>.
    /// </summary>
    public bool Deterministic { get; init; }

    /// <summary>The timestamp written when <see cref="Deterministic"/> is set.</summary>
    public DateTimeOffset FixedTimestamp { get; init; } = new(2026, 10, 4, 0, 0, 0, TimeSpan.Zero);
}
