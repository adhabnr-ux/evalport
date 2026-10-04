// SPDX-License-Identifier: Apache-2.0

using System.Globalization;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace EvalPort.AgentEval;

/// <summary>
/// Constants and small helpers shared by the exporter, the loader and the document builder.
/// </summary>
public static class EvalPortJson
{
    /// <summary>The EvalPort specification version the documents claim conformance to.</summary>
    public const string SpecVersion = "1.0.0";

    /// <summary>The <c>$schema</c> URL of an EvalPort Suite document.</summary>
    public const string SuiteSchema = "https://evalport.org/schema/suite.json";

    /// <summary>The <c>$schema</c> URL of an EvalPort ResultSet document.</summary>
    public const string ResultSetSchema = "https://evalport.org/schema/resultset.json";

    /// <summary>
    /// The reserved <c>metadata</c> key under which this package writes AgentEval-specific data.
    /// Everything that has no typed EvalPort slot goes here, never into a typed field.
    /// </summary>
    public const string MetadataKey = "agenteval";

    /// <summary>
    /// Serializer options used for the files this package writes: indented, keys in insertion
    /// order, <c>null</c> values written (EvalPort Rule 6 depends on <c>"score": null</c> being present).
    /// </summary>
    public static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>The version of the AgentEval assembly this package was loaded next to, for <c>runner.version</c>.</summary>
    public static string AgentEvalVersion { get; } = ReadAgentEvalVersion();

    /// <summary>The version of this package, for <c>metadata.agenteval.exporter_version</c>.</summary>
    public static string PackageVersion { get; } =
        typeof(EvalPortJson).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? typeof(EvalPortJson).Assembly.GetName().Version?.ToString() ?? "0.0.0";

    /// <summary>RFC 3339 / ISO 8601 UTC timestamp with second precision, e.g. <c>2026-10-03T00:00:00Z</c>.</summary>
    public static string Timestamp(DateTimeOffset value) =>
        value.ToUniversalTime().ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture);

    /// <summary>Clamp a score into EvalPort's <c>[0, 1]</c> range. Non-finite values are an error, never a 0.</summary>
    public static double Clamp01(double value)
    {
        if (!double.IsFinite(value))
            throw new ArgumentOutOfRangeException(nameof(value), value, "A score must be a finite number.");
        return Math.Clamp(value, 0.0, 1.0);
    }

    /// <summary>Round a ratio for <c>summary</c> fields the way the other examples in this repository do (4 decimals).</summary>
    public static double Round4(double value) => Math.Round(value, 4, MidpointRounding.AwayFromZero);

    /// <summary>Serialize a document to the indented JSON text this package writes, with a trailing newline.</summary>
    public static string ToJsonText(JsonNode node) => node.ToJsonString(Options) + "\n";

    /// <summary>Convert an arbitrary CLR value (as found in AgentEval metadata dictionaries) into a JSON node.</summary>
    public static JsonNode? ToNode(object? value) => value switch
    {
        null => null,
        JsonNode node => node.DeepClone(),
        JsonElement element => JsonSerializer.SerializeToNode(element),
        string s => JsonValue.Create(s),
        bool b => JsonValue.Create(b),
        int i => JsonValue.Create(i),
        long l => JsonValue.Create(l),
        double d => double.IsFinite(d) ? JsonValue.Create(d) : JsonValue.Create(d.ToString(CultureInfo.InvariantCulture)),
        float f => float.IsFinite(f) ? JsonValue.Create(f) : JsonValue.Create(f.ToString(CultureInfo.InvariantCulture)),
        decimal m => JsonValue.Create(m),
        DateTimeOffset dto => JsonValue.Create(Timestamp(dto)),
        DateTime dt => JsonValue.Create(Timestamp(new DateTimeOffset(dt.ToUniversalTime()))),
        Enum e => JsonValue.Create(e.ToString()),
        _ => JsonSerializer.SerializeToNode(value, value.GetType(), Options),
    };

    /// <summary>Convert a string-keyed dictionary into a JSON object, dropping nothing (null values become JSON null).</summary>
    public static JsonObject ToObject<TValue>(IEnumerable<KeyValuePair<string, TValue>> pairs)
    {
        var obj = new JsonObject();
        foreach (var (key, value) in pairs)
            obj[key] = ToNode(value);
        return obj;
    }

    private static string ReadAgentEvalVersion()
    {
        var asm = typeof(global::AgentEval.Evals.EvalResult).Assembly;
        var info = asm.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        if (info is null)
            return asm.GetName().Version?.ToString() ?? "unknown";
        // "0.42.0-beta+<commit>" -> "0.42.0-beta": the build metadata is not part of the version.
        var plus = info.IndexOf('+');
        return plus < 0 ? info : info[..plus];
    }
}
