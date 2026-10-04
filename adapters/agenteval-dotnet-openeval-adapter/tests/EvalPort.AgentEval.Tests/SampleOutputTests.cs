// SPDX-License-Identifier: Apache-2.0

using System.Text.Json.Nodes;
using EvalPort.AgentEval.Samples;

namespace EvalPort.AgentEval.Tests;

/// <summary>
/// The checked-in <c>sample_output/</c> is the deterministic output of the samples program. If the
/// framework, the scenario or the mapping changes, this fails, and the files must be regenerated on purpose:
/// <c>dotnet run --project samples/EvalPort.AgentEval.Samples -- --out-dir sample_output --deterministic</c>
/// and again with <c>--proposed-verdict</c>.
/// </summary>
public class SampleOutputTests
{
    public static IEnumerable<object[]> Variants()
    {
        foreach (var required in new[] { true, false })
            foreach (var verdict in new[] { false, true })
                yield return new object[] { required, verdict };
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public async Task CheckedInSampleOutput_IsCurrent(bool judgeRequired, bool proposedVerdict)
    {
        var options = new EvalPortExportOptions { Deterministic = true, EmitProposedVerdict = proposedVerdict };
        var outcomes = await Scenario.RunAsync(judgeRequired);
        var run = Scenario.ToRun(outcomes, judgeRequired, options.FixedTimestamp);
        var (suite, rs) = EvalPortDocuments.Build(run, options);

        var name = judgeRequired ? "judge_required" : "judge_optional";
        var dir = proposedVerdict
            ? Path.Combine(RepoPaths.SampleOutputDir, "proposed-verdict", name)
            : Path.Combine(RepoPaths.SampleOutputDir, name);

        Assert.Equal(File.ReadAllText(Path.Combine(dir, "suite.json")), EvalPortJson.ToJsonText(suite));
        Assert.Equal(File.ReadAllText(Path.Combine(dir, "results.json")), EvalPortJson.ToJsonText(rs));
    }

    [Fact]
    public void CheckedInSampleOutput_IsSchemaValid_ExceptForTheProposedField()
    {
        foreach (var name in new[] { "judge_required", "judge_optional" })
        {
            var dir = Path.Combine(RepoPaths.SampleOutputDir, name);
            EvalPortSchemas.Instance.AssertValidSuite(JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "suite.json")))!);
            EvalPortSchemas.Instance.AssertValidResultSet(JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "results.json")))!);

            var proposed = JsonNode.Parse(File.ReadAllText(Path.Combine(RepoPaths.SampleOutputDir, "proposed-verdict", name, "results.json")))!;
            var errors = EvalPortSchemas.Instance.Errors(EvalPortSchemas.Instance.ResultSet, proposed);
            Assert.Equal(7, errors.Count);
            Assert.All(errors, e => Assert.Matches(@"^/results/\d+/verdict: ", e));
        }
    }
}
