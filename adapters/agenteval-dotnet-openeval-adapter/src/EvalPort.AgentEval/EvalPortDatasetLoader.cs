// SPDX-License-Identifier: Apache-2.0

using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using AgentEval.DataLoaders;
using AgentEval.Models;

namespace EvalPort.AgentEval;

/// <summary>
/// AgentEval <see cref="IDatasetLoader"/> that reads an EvalPort <c>Suite</c> document
/// (<c>suite.json</c>) as a list of <see cref="DatasetTestCase"/>s. Registered under the format name
/// <c>"evalport"</c> and the extension <c>.evalport.json</c>.
/// </summary>
/// <remarks>
/// <para>
/// Field mapping (EvalPort <c>TestCase</c> → <see cref="DatasetTestCase"/>): <c>id</c> → <c>Id</c>;
/// <c>input</c> → <c>Input</c> (a multi-turn array is joined with newlines and kept whole under
/// <c>Metadata["evalport.input_turns"]</c>); <c>expected_output</c> → <c>ExpectedOutput</c>;
/// <c>context</c> and <c>retrieval_context</c> → <c>Context</c>; <c>expected_tools</c> →
/// <c>ExpectedTools</c>; <c>tags</c> → <c>Tags</c>; <c>metadata</c> → <c>Metadata</c>. The case's
/// <c>graders</c> (ids or inline objects), with the suite's shared grader definitions resolved, are kept
/// under <c>Metadata["evalport.graders"]</c>: AgentEval has no slot for a grader definition, and this
/// loader does not turn one into an AgentEval metric. <c>Category</c> and <c>PassingScore</c> are left
/// unset, because the suite does not state them.
/// </para>
/// <para>
/// <c>test_cases_file</c> (a JSONL file of <c>TestCase</c> documents next to the suite) is followed, and
/// is confined to the suite's directory: an absolute path, a <c>..</c> escape or a link that leads outside
/// is rejected (see <see cref="ResolveTestCasesFile"/>). The loader checks the fields it needs and nothing else; it is not a validator. Validate with the
/// EvalPort SDKs.
/// </para>
/// </remarks>
public sealed class EvalPortDatasetLoader : IDatasetLoader
{
    /// <summary>The format name of this loader.</summary>
    public const string Name = "evalport";

    /// <summary>Metadata key under which a case's resolved EvalPort graders are kept.</summary>
    public const string GradersMetadataKey = "evalport.graders";

    /// <summary>Metadata key under which a multi-turn input's turns are kept.</summary>
    public const string InputTurnsMetadataKey = "evalport.input_turns";

    /// <summary>Metadata key for the suite id the case came from.</summary>
    public const string SuiteIdMetadataKey = "evalport.suite_id";

    /// <inheritdoc/>
    public string Format => Name;

    /// <inheritdoc/>
    /// <remarks><c>.json</c> is left to AgentEval's own <c>JsonDatasetLoader</c>; an EvalPort suite is picked by name or by this suffix.</remarks>
    public IReadOnlyList<string> SupportedExtensions { get; } = new[] { ".evalport.json" };

    /// <inheritdoc/>
    public bool IsTrulyStreaming => false;

    /// <inheritdoc/>
    public async Task<IReadOnlyList<DatasetTestCase>> LoadAsync(string path, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"EvalPort suite not found: {path}", path);

        JsonNode? root;
        await using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, useAsync: true))
        {
            try
            {
                root = await JsonNode.ParseAsync(stream, cancellationToken: ct).ConfigureAwait(false);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"Invalid JSON in EvalPort suite {path}: {ex.Message}", ex);
            }
        }
        if (root is not JsonObject suite)
            throw new InvalidDataException($"EvalPort suite {path} is not a JSON object.");

        if (suite["test_cases"] is null && suite["test_cases_file"] is JsonValue fileNode && fileNode.TryGetValue<string>(out var relative))
        {
            var casesPath = ResolveTestCasesFile(path, relative);
            suite["test_cases"] = await ReadJsonLinesAsync(casesPath, ct).ConfigureAwait(false);
        }

        return FromSuite(suite);
    }

    /// <summary>
    /// Resolve a suite's <c>test_cases_file</c> and confine it to the suite's own directory (or a
    /// subdirectory of it). Suites can come from other people, and the lines of that file become model
    /// inputs, so an absolute path, a <c>..</c> escape, or a link whose target lies outside the suite's
    /// directory is rejected with <see cref="InvalidDataException"/>. Spec: the file is "next to the suite".
    /// </summary>
    public static string ResolveTestCasesFile(string suitePath, string testCasesFile)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(suitePath);
        if (string.IsNullOrWhiteSpace(testCasesFile))
            throw new InvalidDataException("test_cases_file is empty.");
        if (Path.IsPathRooted(testCasesFile))
            throw new InvalidDataException($"test_cases_file must be a relative path inside the suite's directory, got an absolute path: {testCasesFile}");

        var suiteDir = Path.GetDirectoryName(Path.GetFullPath(suitePath)) ?? Directory.GetCurrentDirectory();
        var casesPath = Path.GetFullPath(Path.Combine(suiteDir, testCasesFile));
        if (!IsInside(suiteDir, casesPath))
            throw new InvalidDataException($"test_cases_file must stay inside the suite's directory ({suiteDir}); {testCasesFile} resolves to {casesPath}");

        // A link (the file itself, or any directory on the way to it) must not point outside either.
        var realCases = RealPath(casesPath);
        if (!IsInside(RealPath(suiteDir), realCases))
            throw new InvalidDataException($"test_cases_file {testCasesFile} is a link to {realCases}, outside the suite's directory ({suiteDir})");
        return casesPath;
    }

    /// <summary>Resolve every link on the way to <paramref name="fullPath"/>; components that do not exist are kept as written.</summary>
    private static string RealPath(string fullPath)
    {
        var root = Path.GetPathRoot(fullPath) ?? string.Empty;
        var current = root;
        foreach (var part in fullPath[root.Length..].Split(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            FileSystemInfo info = Directory.Exists(current) ? new DirectoryInfo(current) : new FileInfo(current);
            if (info.Exists && info.LinkTarget is not null)
                current = info.ResolveLinkTarget(returnFinalTarget: true)?.FullName ?? current;
        }
        return current;
    }

    private static bool IsInside(string directory, string fullPath)
    {
        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory)) + Path.DirectorySeparatorChar;
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        return fullPath.StartsWith(root, comparison);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<DatasetTestCase> LoadStreamingAsync(string path, [EnumeratorCancellation] CancellationToken ct = default)
    {
        foreach (var testCase in await LoadAsync(path, ct).ConfigureAwait(false))
        {
            ct.ThrowIfCancellationRequested();
            yield return testCase;
        }
    }

    /// <summary>Convert an in-memory EvalPort <c>Suite</c> object.</summary>
    public static IReadOnlyList<DatasetTestCase> FromSuite(JsonObject suite)
    {
        ArgumentNullException.ThrowIfNull(suite);
        if (suite["test_cases"] is not JsonArray cases)
            throw new InvalidDataException("EvalPort suite has no test_cases array (and no test_cases_file was resolved).");

        var suiteId = suite["id"]?.GetValue<string>();
        var shared = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
        if (suite["graders"] is JsonArray graders)
        {
            foreach (var g in graders.OfType<JsonObject>())
            {
                if (g["id"] is JsonValue idNode && idNode.TryGetValue<string>(out var gid))
                    shared[gid] = g;
            }
        }

        var list = new List<DatasetTestCase>(cases.Count);
        var index = 0;
        foreach (var node in cases)
        {
            if (node is not JsonObject tc)
                throw new InvalidDataException($"test_cases[{index}] is not an object.");
            list.Add(ToDatasetTestCase(tc, shared, suiteId, index));
            index++;
        }
        return list;
    }

    /// <summary>Convert one EvalPort <c>TestCase</c>.</summary>
    /// <param name="testCase">The test case object.</param>
    /// <param name="sharedGraders">The suite's shared grader definitions, by id, for resolving string references.</param>
    /// <param name="suiteId">The suite id, kept in metadata.</param>
    /// <param name="index">Position in the suite, for error messages.</param>
    public static DatasetTestCase ToDatasetTestCase(JsonObject testCase, IReadOnlyDictionary<string, JsonObject> sharedGraders, string? suiteId, int index)
    {
        ArgumentNullException.ThrowIfNull(testCase);
        ArgumentNullException.ThrowIfNull(sharedGraders);

        var id = testCase["id"]?.GetValue<string>();
        if (string.IsNullOrEmpty(id))
            throw new InvalidDataException($"test_cases[{index}] has no id.");

        var metadata = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (testCase["metadata"] is JsonObject md)
        {
            foreach (var (key, value) in md)
                metadata[key] = ToClr(value);
        }

        string input;
        switch (testCase["input"])
        {
            case JsonValue v when v.TryGetValue<string>(out var s):
                input = s;
                break;
            case JsonArray turns:
                var strings = turns.Select(t => t?.GetValue<string>() ?? "").ToList();
                input = string.Join("\n", strings);
                metadata[InputTurnsMetadataKey] = strings;
                break;
            default:
                throw new InvalidDataException($"test_cases[{index}] ('{id}') has no string or array input.");
        }

        var context = new List<string>();
        AddStrings(testCase["context"], context);
        AddStrings(testCase["retrieval_context"], context);

        var expectedTools = new List<string>();
        AddStrings(testCase["expected_tools"], expectedTools);

        var tags = new List<string>();
        AddStrings(testCase["tags"], tags);

        if (testCase["graders"] is JsonArray graderRefs)
        {
            var resolved = new List<object?>();
            foreach (var g in graderRefs)
            {
                if (g is JsonValue gv && gv.TryGetValue<string>(out var gid))
                    resolved.Add(sharedGraders.TryGetValue(gid, out var def) ? ToClr(def) : new Dictionary<string, object?> { ["id"] = gid });
                else if (g is JsonObject inline)
                    resolved.Add(ToClr(inline));
            }
            metadata[GradersMetadataKey] = resolved;
        }
        if (suiteId is not null)
            metadata[SuiteIdMetadataKey] = suiteId;
        foreach (var passthrough in new[] { "provider", "params", "timeout_ms", "weight", "tools_called" })
        {
            if (testCase[passthrough] is { } extra)
                metadata["evalport." + passthrough] = ToClr(extra);
        }

        return new DatasetTestCase
        {
            Id = id,
            Input = input,
            ExpectedOutput = testCase["expected_output"]?.GetValue<string>(),
            Context = context.Count > 0 ? context : null,
            ExpectedTools = expectedTools.Count > 0 ? expectedTools : null,
            Tags = tags.Count > 0 ? tags : null,
            Metadata = metadata,
        };
    }

    private static async Task<JsonArray> ReadJsonLinesAsync(string path, CancellationToken ct)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"test_cases_file not found: {path}", path);
        var array = new JsonArray();
        var lineNumber = 0;
        foreach (var line in await File.ReadAllLinesAsync(path, ct).ConfigureAwait(false))
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                array.Add(JsonNode.Parse(line));
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"Invalid JSON at line {lineNumber} in {path}: {ex.Message}", ex);
            }
        }
        return array;
    }

    private static void AddStrings(JsonNode? node, List<string> into)
    {
        if (node is not JsonArray array) return;
        foreach (var item in array)
        {
            if (item is JsonValue v && v.TryGetValue<string>(out var s))
                into.Add(s);
        }
    }

    /// <summary>JSON node → plain CLR values (string, bool, long, double, list, dictionary), the shapes AgentEval metadata holds.</summary>
    internal static object? ToClr(JsonNode? node) => node switch
    {
        null => null,
        JsonObject o => o.ToDictionary(kv => kv.Key, kv => ToClr(kv.Value), StringComparer.Ordinal),
        JsonArray a => a.Select(ToClr).ToList(),
        JsonValue v when v.TryGetValue<string>(out var s) => s,
        JsonValue v when v.TryGetValue<bool>(out var b) => b,
        JsonValue v when v.TryGetValue<long>(out var l) => l,
        JsonValue v when v.TryGetValue<double>(out var d) => d,
        JsonValue v => v.ToJsonString(),
        _ => node.ToJsonString(),
    };
}
