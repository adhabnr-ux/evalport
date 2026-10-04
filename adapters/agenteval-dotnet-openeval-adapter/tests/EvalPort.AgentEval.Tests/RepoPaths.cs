// SPDX-License-Identifier: Apache-2.0

using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using Json.Schema;

namespace EvalPort.AgentEval.Tests;

/// <summary>Paths into this repository, baked in at build time by the test project file.</summary>
public static class RepoPaths
{
    private static string Meta(string key) =>
        typeof(RepoPaths).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().Single(a => a.Key == key).Value
        ?? throw new InvalidOperationException($"assembly metadata {key} missing");

    /// <summary><c>schema/</c> at the repository root.</summary>
    public static string SchemaDir => Path.GetFullPath(Meta("EvalPortSchemaDir"));

    /// <summary><c>examples/</c> at the repository root.</summary>
    public static string ExamplesDir => Path.GetFullPath(Meta("EvalPortExamplesDir"));

    /// <summary>This adapter's checked-in <c>sample_output/</c>.</summary>
    public static string SampleOutputDir => Path.GetFullPath(Meta("EvalPortSampleOutputDir"));
}

/// <summary>
/// The repository's four JSON Schemas, loaded once and registered so that the <c>$ref</c>s between them
/// (<c>suite.json</c> → <c>testcase.json</c> → <c>grader.json</c>) resolve.
/// </summary>
public sealed class EvalPortSchemas
{
    /// <summary>The shared instance.</summary>
    public static EvalPortSchemas Instance { get; } = new();

    /// <summary>The Suite schema.</summary>
    public JsonSchema Suite { get; }

    /// <summary>The ResultSet schema.</summary>
    public JsonSchema ResultSet { get; }

    private readonly EvaluationOptions _options;

    private EvalPortSchemas()
    {
        // JsonSchema.Net 7.x: each EvaluationOptions owns a registry; registering the four schemas there by their
        // $id lets suite.json's $ref to testcase.json and testcase.json's $ref to grader.json resolve offline.
        _options = new EvaluationOptions { OutputFormat = OutputFormat.List };
        JsonSchema Load(string name)
        {
            var path = Path.Combine(RepoPaths.SchemaDir, name);
            var schema = JsonSchema.FromText(File.ReadAllText(path));
            var id = schema.GetId() ?? throw new InvalidOperationException($"{name} has no $id");
            _options.SchemaRegistry.Register(id, schema);
            return schema;
        }
        Load("grader.json");
        Load("testcase.json");
        Suite = Load("suite.json");
        ResultSet = Load("resultset.json");
    }

    /// <summary>Evaluate a node and return the list of failing keyword locations (empty when valid).</summary>
    public IReadOnlyList<string> Errors(JsonSchema schema, JsonNode node)
    {
        var result = schema.Evaluate(node, _options);
        if (result.IsValid) return Array.Empty<string>();
        return result.Details
            .Where(d => !d.IsValid && d.HasErrors)
            .SelectMany(d => d.Errors!.Select(e => $"{d.InstanceLocation}: {e.Key}: {e.Value}"))
            .Distinct()
            .ToList();
    }

    /// <summary>Assert that a Suite document is schema-valid.</summary>
    public void AssertValidSuite(JsonNode suite) => Assert.Empty(Errors(Suite, suite));

    /// <summary>Assert that a ResultSet document is schema-valid.</summary>
    public void AssertValidResultSet(JsonNode resultSet) => Assert.Empty(Errors(ResultSet, resultSet));

    /// <summary>Re-parse a node through text, so tests see exactly what a file reader would.</summary>
    public static JsonNode Reparse(JsonNode node) =>
        JsonNode.Parse(node.ToJsonString(EvalPortJson.Options)) ?? throw new JsonException("null document");
}
