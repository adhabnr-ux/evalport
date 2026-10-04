// SPDX-License-Identifier: Apache-2.0

using System.Globalization;
using System.Text;
using AgentEval.Evals;
using EvalPort.AgentEval;
using EvalPort.AgentEval.Samples;

// Usage:
//   dotnet run --project samples/EvalPort.AgentEval.Samples -- [--out-dir DIR] [--variant required|optional|all]
//                                                               [--proposed-verdict] [--deterministic] [--quiet]
//
// Writes DIR/<variant>/suite.json and results.json (and, with --proposed-verdict, DIR/proposed-verdict/<variant>/...)
// and prints one comparison line per case. The checked-in sample_output/ is `--out-dir sample_output --deterministic`
// followed by `--out-dir sample_output --deterministic --proposed-verdict`.

var outDir = new DirectoryInfo("sample_output");
var variant = "all";
var proposedVerdict = false;
var deterministic = false;
var quiet = false;
for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--out-dir": outDir = new DirectoryInfo(args[++i]); break;
        case "--variant": variant = args[++i]; break;
        case "--proposed-verdict": proposedVerdict = true; break;
        case "--deterministic": deterministic = true; break;
        case "--quiet": quiet = true; break;
        default:
            Console.Error.WriteLine($"unknown argument: {args[i]}");
            return 2;
    }
}

var variants = variant switch
{
    "required" => new[] { true },
    "optional" => new[] { false },
    "all" => new[] { true, false },
    _ => throw new ArgumentException("--variant must be required, optional or all"),
};

var options = new EvalPortExportOptions { Deterministic = deterministic, EmitProposedVerdict = proposedVerdict };
var started = deterministic ? options.FixedTimestamp : DateTimeOffset.UtcNow;

foreach (var judgeRequired in variants)
{
    var outcomes = await Scenario.RunAsync(judgeRequired);
    var run = Scenario.ToRun(outcomes, judgeRequired, started);
    var (suite, resultSet) = EvalPortDocuments.Build(run, options);

    var name = judgeRequired ? "judge_required" : "judge_optional";
    var dir = proposedVerdict
        ? new DirectoryInfo(Path.Combine(outDir.FullName, "proposed-verdict", name))
        : new DirectoryInfo(Path.Combine(outDir.FullName, name));
    dir.Create();
    var utf8 = new UTF8Encoding(false);
    File.WriteAllText(Path.Combine(dir.FullName, "suite.json"), EvalPortJson.ToJsonText(suite), utf8);
    File.WriteAllText(Path.Combine(dir.FullName, "results.json"), EvalPortJson.ToJsonText(resultSet), utf8);

    if (quiet) continue;
    Console.WriteLine($"== AgentEval {EvalPortJson.AgentEvalVersion}   variant={name}   proposed_verdict={proposedVerdict}");
    Console.WriteLine(ComparisonTable(outcomes, resultSet));
    var s = resultSet["summary"]!;
    Console.WriteLine($"summary: total={s["total"]} passed={s["passed"]} failed={s["failed"]} skipped={s["skipped"]} pass_rate={s["pass_rate"]}");
    Console.WriteLine($"wrote {Path.Combine(dir.FullName, "suite.json")} and results.json");
    Console.WriteLine();
}
return 0;

static string ComparisonTable(IReadOnlyList<Scenario.CaseOutcome> outcomes, System.Text.Json.Nodes.JsonObject resultSet)
{
    var header = $"{"case",-5} {"exact",-14} {"judge",-14} {"composite label",-16} {"Result.passed",-13} {"EvalPort graders (exact, judge)",-32} {"verdict (#49)",-14}";
    var sb = new StringBuilder().AppendLine(header).AppendLine(new string('-', header.Length));
    var results = resultSet["results"]!.AsArray();
    for (var i = 0; i < outcomes.Count; i++)
    {
        var o = outcomes[i];
        var r = results[i]!;
        string Leaf(string key)
        {
            if (o.Result is null) return "(no result)";
            var leaf = EvalResultMapping.Leaves(o.Result).FirstOrDefault(l => l.Metric.Key == key);
            return leaf is null ? "-" : $"{leaf.Score.Label}/{EvalResultMapping.MeasurementOf(leaf.Score)}";
        }
        var graders = string.Join(", ", r["grader_results"]!.AsArray().Select(g =>
            g!["score"] is null ? "null" : g["score"]!.GetValue<double>().ToString("0.##", CultureInfo.InvariantCulture)));
        var label = o.Result?.Score.Label ?? $"threw {o.Error?.GetType().Name}";
        var verdict = r["verdict"]?.GetValue<string>() ?? "-";
        sb.AppendLine($"{o.Spec.Id,-5} {Leaf("exact"),-14} {Leaf("judge"),-14} {label,-16} {r["passed"]!.GetValue<bool>().ToString().ToLowerInvariant(),-13} {graders,-32} {verdict,-14}");
    }
    return sb.ToString().TrimEnd();
}
