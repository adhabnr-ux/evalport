// SPDX-License-Identifier: Apache-2.0

using System.Text.Json.Nodes;
using AgentEval.Evals;
using AgentEval.Models;
using EvalPort.AgentEval.Samples;

namespace EvalPort.AgentEval.Tests;

/// <summary><see cref="EvalPortDatasetLoader"/> on this repository's example suites.</summary>
public class DatasetLoaderTests
{
    private static string Example(string name) => Path.Combine(RepoPaths.ExamplesDir, name);

    [Fact]
    public void Loader_IdentifiesItself()
    {
        var loader = new EvalPortDatasetLoader();
        Assert.Equal("evalport", loader.Format);
        Assert.Equal(new[] { ".evalport.json" }, loader.SupportedExtensions);
        Assert.False(loader.IsTrulyStreaming);
    }

    [Fact]
    public async Task BasicSuite_LoadsIdsInputsExpectedOutputsAndGraders()
    {
        var cases = await new EvalPortDatasetLoader().LoadAsync(Example("basic-suite.json"));
        Assert.Equal(2, cases.Count);
        var tc = cases[0];
        Assert.Equal("tc_1", tc.Id);
        Assert.False(string.IsNullOrEmpty(tc.Input));
        Assert.Equal("4", tc.ExpectedOutput);
        Assert.Equal("suite_qa_basic", tc.Metadata[EvalPortDatasetLoader.SuiteIdMetadataKey]);
        var graders = Assert.IsType<List<object?>>(tc.Metadata[EvalPortDatasetLoader.GradersMetadataKey]);
        var grader = Assert.IsType<Dictionary<string, object?>>(graders.Single());
        Assert.Equal("gr_exact", grader["id"]);
        Assert.Equal("exact_match", grader["type"]); // the shared definition was resolved, not just the id
        Assert.Null(tc.Category);
        Assert.Null(tc.PassingScore);
    }

    [Fact]
    public async Task MultiTurnInput_IsJoined_AndTheTurnsAreKept()
    {
        var cases = await new EvalPortDatasetLoader().LoadAsync(Example("multi-turn.json"));
        var tc = cases.Single(c => c.Id == "tc_001");
        var turns = Assert.IsType<List<string>>(tc.Metadata[EvalPortDatasetLoader.InputTurnsMetadataKey]);
        Assert.Equal(3, turns.Count);
        Assert.Equal(string.Join("\n", turns), tc.Input);
        Assert.Equal(3L, tc.Metadata["turns"]);
        Assert.Equal("programming", tc.Metadata["topic"]);
    }

    [Fact]
    public async Task AgentTools_ExpectedToolsAndTagsAndMetadata_Survive()
    {
        var cases = await new EvalPortDatasetLoader().LoadAsync(Example("agent-tools.json"));
        var tc = cases.Single(c => c.Id == "tc_001");
        Assert.Equal(new[] { "web_search" }, tc.ExpectedTools);
        Assert.Equal("tool_selection", tc.Metadata["scenario"]);
        var graders = Assert.IsType<List<object?>>(tc.Metadata[EvalPortDatasetLoader.GradersMetadataKey]);
        Assert.Equal(3, graders.Count);
    }

    [Fact]
    public async Task RagSuite_ContextIsCarried()
    {
        var cases = await new EvalPortDatasetLoader().LoadAsync(Example("rag-eval-suite.json"));
        Assert.Equal(new[] { "K8s orchestrates containers" }, cases.Single().Context);
    }

    [Fact]
    public async Task StreamingOverload_YieldsTheSameCases()
    {
        var loader = new EvalPortDatasetLoader();
        var all = await loader.LoadAsync(Example("agent-tools.json"));
        var streamed = new List<DatasetTestCase>();
        await foreach (var tc in loader.LoadStreamingAsync(Example("agent-tools.json")))
            streamed.Add(tc);
        Assert.Equal(all.Select(c => c.Id), streamed.Select(c => c.Id));
    }

    [Fact]
    public async Task TestCasesFile_IsFollowedRelativeToTheSuite()
    {
        var dir = Directory.CreateTempSubdirectory("evalport-loader");
        try
        {
            File.WriteAllText(Path.Combine(dir.FullName, "suite.evalport.json"),
                """{"version":"1.0.0","id":"suite_large","test_cases_file":"cases.jsonl","graders":[{"id":"gr_exact","type":"exact_match"}]}""");
            File.WriteAllText(Path.Combine(dir.FullName, "cases.jsonl"),
                "{\"id\":\"a\",\"input\":\"1+1?\",\"expected_output\":\"2\",\"graders\":[\"gr_exact\"]}\n\n" +
                "{\"id\":\"b\",\"input\":\"2+2?\",\"expected_output\":\"4\",\"graders\":[\"gr_exact\"]}\n");
            var cases = await new EvalPortDatasetLoader().LoadAsync(Path.Combine(dir.FullName, "suite.evalport.json"));
            Assert.Equal(new[] { "a", "b" }, cases.Select(c => c.Id));
            Assert.Equal("2", cases[0].ExpectedOutput);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public async Task MissingFile_InvalidJson_AndMissingInput_AreClearErrors()
    {
        var loader = new EvalPortDatasetLoader();
        await Assert.ThrowsAsync<FileNotFoundException>(() => loader.LoadAsync(Example("does-not-exist.json")));

        var dir = Directory.CreateTempSubdirectory("evalport-loader-bad");
        try
        {
            var bad = Path.Combine(dir.FullName, "bad.evalport.json");
            File.WriteAllText(bad, "{ not json");
            await Assert.ThrowsAsync<InvalidDataException>(() => loader.LoadAsync(bad));

            var noInput = JsonNode.Parse("""{"id":"s","test_cases":[{"id":"x","graders":["g"]}]}""")!.AsObject();
            Assert.Throws<InvalidDataException>(() => EvalPortDatasetLoader.FromSuite(noInput));

            var noCases = JsonNode.Parse("""{"id":"s"}""")!.AsObject();
            Assert.Throws<InvalidDataException>(() => EvalPortDatasetLoader.FromSuite(noCases));
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public async Task RoundTrip_SuiteToCasesToDocuments_KeepsIdsInputsAndExpectedOutputs()
    {
        // Load an EvalPort suite as AgentEval cases, grade them with the real ExactAnswerEval, write them back.
        var cases = await new EvalPortDatasetLoader().LoadAsync(Example("basic-suite.json"));
        var exact = new Scenario.ExactAnswerEval();
        var runs = new List<EvalPortCaseRun>();
        foreach (var tc in cases)
        {
            var result = await exact.EvaluateAsync(new EvalInput(tc.Input, Response: tc.ExpectedOutput, GroundTruth: tc.ExpectedOutput));
            runs.Add(new EvalPortCaseRun(tc, tc.ExpectedOutput, new[] { result }));
        }
        var (suite, rs) = EvalPortDocuments.Build(
            new EvalPortRun { SuiteId = "suite_qa_basic_roundtrip", RunId = "r1", Cases = runs },
            new EvalPortExportOptions { Deterministic = true });
        EvalPortSchemas.Instance.AssertValidSuite(EvalPortSchemas.Reparse(suite));
        EvalPortSchemas.Instance.AssertValidResultSet(EvalPortSchemas.Reparse(rs));

        var original = JsonNode.Parse(File.ReadAllText(Example("basic-suite.json")))!["test_cases"]!.AsArray();
        var written = suite["test_cases"]!.AsArray();
        Assert.Equal(original.Count, written.Count);
        for (var i = 0; i < original.Count; i++)
        {
            Assert.Equal((string)original[i]!["id"]!, (string)written[i]!["id"]!);
            Assert.Equal((string)original[i]!["input"]!, (string)written[i]!["input"]!);
            Assert.Equal((string)original[i]!["expected_output"]!, (string)written[i]!["expected_output"]!);
            // The original graders are kept as data, not re-emitted as EvalPort graders: the eval that ran is.
            Assert.Equal(new[] { "exact" }, written[i]!["graders"]!.AsArray().Select(g => (string)g!));
            Assert.NotNull(written[i]!["metadata"]!["agenteval"]!["metadata"]![EvalPortDatasetLoader.GradersMetadataKey]);
        }
        Assert.All(rs["results"]!.AsArray(), r => Assert.True(r!["passed"]!.GetValue<bool>()));
    }
}
