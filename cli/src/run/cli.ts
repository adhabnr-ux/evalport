import type { DryRunReport } from "./cost";
import type { ProviderName, RunOptions } from "./types";
import { runEval } from "./runner";
import type { ResultSet } from "../../../sdk/typescript/src/types";

export class CliArgError extends Error {}
/** Thrown for `--help`/`-h` specifically, so runCommand can print to stdout
 * and exit 0 instead of treating it like a usage error. */
export class HelpRequested extends Error {}

export const RUN_HELP = `Usage: evalport run <suite.json> --provider <openai|anthropic> [options]

Required:
  --provider <name>          "openai" or "anthropic" (or any OpenAI-compatible
                              endpoint via --api-base with --provider openai)

Model / provider:
  --model <name>              Falls back to config.provider.model in the suite if omitted
  --api-base <url>            Falls back to the provider's default endpoint
  --api-key-env <VAR>         Env var holding the API key (default: OPENAI_API_KEY / ANTHROPIC_API_KEY)
  --temperature <n>
  --max-tokens <n>

Execution:
  --parallel <n>               Concurrent test cases in flight (default: 1)
  --limit <n>                  Only run the first <n> test cases
  --output <path>               Write the ResultSet JSON here (also written incrementally as cases complete)
  --dry-run                     Estimate cost and exit — makes zero API calls
  --max-attempts <n>            Retry attempts for retryable (429/5xx) provider errors (default: 3)
  --backoff-ms <n>               Base backoff between retries, doubles each attempt (default: 1000)
  --run-id <id>                  Override the generated run_id

Semantic similarity embeddings (always OpenAI-compatible, independent of --provider):
  --embedding-api-base <url>
  --embedding-api-key-env <VAR>
  --embedding-model <name>       Default: text-embedding-3-small

Examples:
  evalport run suite.json --provider openai --model gpt-4o-mini --dry-run
  evalport run suite.json --provider anthropic --model claude-3-5-sonnet-20241022 --output results.json
  evalport run suite.json --provider openai --model gpt-4o-mini --api-base http://localhost:11434/v1 --api-key-env LOCAL_KEY
`;

function parseIntArg(flag: string, v: string): number {
  const n = Number(v);
  if (!Number.isFinite(n) || !Number.isInteger(n)) throw new CliArgError(`${flag} expects an integer, got "${v}"`);
  return n;
}

function parseFloatArg(flag: string, v: string): number {
  const n = Number(v);
  if (!Number.isFinite(n)) throw new CliArgError(`${flag} expects a number, got "${v}"`);
  return n;
}

function requireProvider(v: string): ProviderName {
  if (v !== "openai" && v !== "anthropic") throw new CliArgError(`Invalid --provider "${v}" — must be "openai" or "anthropic".`);
  return v;
}

/** Pure argument parser — no I/O, no process access beyond reading the
 * array it's given — so it's trivial to unit test every flag and every
 * error path without spawning a subprocess. */
export function parseRunArgs(argv: string[]): RunOptions {
  const opts: Partial<RunOptions> = { parallel: 1, dryRun: false, maxAttempts: 3, backoffMs: 1000 };
  const positional: string[] = [];

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = (): string => {
      i++;
      if (i >= argv.length) throw new CliArgError(`Missing value for ${a}`);
      return argv[i];
    };
    switch (a) {
      case "--provider":
        opts.provider = requireProvider(next());
        break;
      case "--model":
        opts.model = next();
        break;
      case "--api-base":
        opts.apiBase = next();
        break;
      case "--api-key-env":
        opts.apiKeyEnv = next();
        break;
      case "--temperature":
        opts.temperature = parseFloatArg(a, next());
        break;
      case "--max-tokens":
        opts.maxTokens = parseIntArg(a, next());
        break;
      case "--parallel":
        opts.parallel = parseIntArg(a, next());
        break;
      case "--limit":
        opts.limit = parseIntArg(a, next());
        break;
      case "--output":
        opts.output = next();
        break;
      case "--dry-run":
        opts.dryRun = true;
        break;
      case "--max-attempts":
        opts.maxAttempts = parseIntArg(a, next());
        break;
      case "--backoff-ms":
        opts.backoffMs = parseIntArg(a, next());
        break;
      case "--embedding-api-base":
        opts.embeddingApiBase = next();
        break;
      case "--embedding-api-key-env":
        opts.embeddingApiKeyEnv = next();
        break;
      case "--embedding-model":
        opts.embeddingModel = next();
        break;
      case "--run-id":
        opts.runId = next();
        break;
      case "--help":
      case "-h":
        throw new HelpRequested(RUN_HELP);
      default:
        if (a.startsWith("--")) throw new CliArgError(`Unknown flag: ${a}\n\n${RUN_HELP}`);
        positional.push(a);
    }
  }

  if (positional.length === 0) throw new CliArgError(`Missing suite file.\n\n${RUN_HELP}`);
  opts.suitePath = positional[0];
  if (!opts.provider) throw new CliArgError(`--provider is required (openai|anthropic).\n\n${RUN_HELP}`);
  if (opts.parallel !== undefined && opts.parallel < 1) throw new CliArgError(`--parallel must be >= 1, got ${opts.parallel}`);
  if (opts.limit !== undefined && opts.limit < 0) throw new CliArgError(`--limit must be >= 0, got ${opts.limit}`);

  return opts as RunOptions;
}

export function formatDryRunReport(report: DryRunReport): string {
  const lines: string[] = [];
  lines.push(`Dry run: ${report.testCaseCount} test case(s), model "${report.model}"${report.modelPricingKnown ? "" : " (pricing unknown — fallback rate used)"}`);
  if (report.judgeModelsUsed.length > 0) lines.push(`Also calls: ${report.judgeModelsUsed.join(", ")}`);
  lines.push("");
  lines.push("Test case            Graders                   Est. input tok   Est. output tok   Est. cost");
  for (const item of report.lineItems) {
    lines.push(
      `${item.testCaseId.padEnd(20)}  ${item.graderIds.join(",").padEnd(24)}  ${String(item.estimatedInputTokens).padStart(14)}   ${String(item.estimatedOutputTokens).padStart(15)}   $${item.estimatedCostUsd.toFixed(4)}`,
    );
  }
  lines.push("");
  lines.push(`Total estimated tokens: ${report.totalEstimatedInputTokens} in / ${report.totalEstimatedOutputTokens} out`);
  lines.push(`TOTAL ESTIMATED COST: $${report.totalEstimatedCostUsd.toFixed(4)}`);
  if (report.warnings.length > 0) {
    lines.push("");
    for (const w of report.warnings) lines.push(`⚠ ${w}`);
  }
  return lines.join("\n");
}

export function formatRunSummary(resultSet: ResultSet): string {
  const s = resultSet.summary;
  if (!s) return `Run ${resultSet.run_id} complete — ${resultSet.results.length} result(s), no summary computed.`;
  const lines: string[] = [];
  lines.push(`Run ${resultSet.run_id} complete.`);
  lines.push(`  Total:  ${s.total ?? resultSet.results.length}`);
  lines.push(`  Passed: ${s.passed ?? 0}`);
  lines.push(`  Failed: ${s.failed ?? 0}`);
  if (s.skipped) lines.push(`  Skipped grader results: ${s.skipped}`);
  if (typeof s.pass_rate === "number") lines.push(`  Pass rate: ${(s.pass_rate * 100).toFixed(1)}%`);
  if (typeof s.avg_score === "number") lines.push(`  Avg score: ${s.avg_score.toFixed(3)}`);
  const cost = (resultSet.metadata as Record<string, unknown> | undefined)?.openeval as { cost?: { total_estimated_cost_usd?: number } } | undefined;
  if (cost?.cost?.total_estimated_cost_usd !== undefined) lines.push(`  Estimated cost: $${cost.cost.total_estimated_cost_usd.toFixed(4)}`);
  return lines.join("\n");
}

/** The full `evalport run` command: parse args, run (or dry-run), print a
 * report, return a process exit code. Kept separate from index.ts's
 * dispatcher so it's directly unit-testable without spawning a subprocess. */
export async function runCommand(argv: string[]): Promise<number> {
  let opts: RunOptions;
  try {
    opts = parseRunArgs(argv);
  } catch (e) {
    if (e instanceof HelpRequested) {
      console.log(e.message);
      return 0;
    }
    console.error((e as Error).message);
    return 1;
  }

  try {
    const { resultSet, dryRun } = await runEval(opts);
    if (dryRun) {
      console.log(formatDryRunReport(dryRun));
      console.log("\nDry run only — no API calls were made and nothing was spent. Get explicit budget approval before re-running without --dry-run.");
      return 0;
    }
    if (!resultSet) {
      console.error("Internal error: runEval() returned neither a result set nor a dry-run report.");
      return 1;
    }
    console.log(formatRunSummary(resultSet));
    if (opts.output) console.log(`\nFull results written to ${opts.output}`);
    const failed = resultSet.summary?.failed ?? resultSet.results.filter((r) => !r.passed).length;
    return failed > 0 ? 1 : 0;
  } catch (e) {
    console.error("Error: " + (e as Error).message);
    return 1;
  }
}
