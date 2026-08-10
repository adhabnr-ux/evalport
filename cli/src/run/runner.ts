import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname, resolve as resolvePath } from "path";
import type { EvalSuite, GraderResult, ProviderConfig, Result, ResultSet, TestCase } from "../../../sdk/typescript/src/types";
import { OPENEVAL_VERSION } from "../../../sdk/typescript/src/types";
import { validateResultSet, validateSuite } from "../../../sdk/typescript/src/validate";
import { computeSummary } from "../../../sdk/typescript/src/convert";
import type { GraderClients } from "./graders/index";
import { gradeOne, resolveGraders } from "./graders/index";
import { estimateCostUsd, estimateTokens, pricingFor } from "./cost";
import type { DryRunLineItem, DryRunReport } from "./cost";
import {
  DEFAULT_API_BASE,
  DEFAULT_API_KEY_ENV,
  MissingApiKeyError,
  ProviderHttpError,
  createProviderClient,
  resolveProviderConfig,
} from "./providers";
import type { ProviderClient, ResolvedProviderConfig, RunOptions } from "./types";

// The orchestration layer for `evalport run`. Everything here is built to be
// testable without a live API key: the provider client factory is injectable
// (see RunDeps below), so unit and integration tests supply a mock
// ProviderClient and never touch the network.

/** Identifies this runner in ResultSet.runner — bump alongside cli/package.json's version. */
const RUNNER_NAME = "evalport-cli";
const RUNNER_VERSION = "1.0.0";

const DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small";
/** Used only to size --dry-run's completion-token guess when no --max-tokens
 * is given — real runs use whatever the provider actually returns. */
const ASSUMED_COMPLETION_TOKENS_FOR_DRY_RUN = 256;
/** Rough guess for a judge model's structured {"score":...} reply. */
const ASSUMED_JUDGE_OUTPUT_TOKENS = 64;

export class TimeoutError extends Error {
  constructor(public readonly ms: number) {
    super(`Timed out after ${ms}ms`);
    this.name = "TimeoutError";
  }
}

export interface LoadedSuite {
  suite: EvalSuite;
  testCases: TestCase[];
}

/** Loads and validates a suite file, resolving `test_cases_file` (JSONL,
 * per SPEC.md's "large suites" section) relative to the suite file if the
 * suite doesn't inline its test cases. Refuses to return an invalid suite —
 * `evalport run` never executes against a suite that fails validateSuite(). */
export function loadSuite(suitePath: string): LoadedSuite {
  if (!existsSync(suitePath)) throw new Error(`Suite file not found: ${suitePath}`);

  let suite: EvalSuite;
  try {
    suite = JSON.parse(readFileSync(suitePath, "utf-8"));
  } catch (e) {
    throw new Error(`Suite file is not valid JSON: ${(e as Error).message}`);
  }

  const validation = validateSuite(suite);
  if (!validation.valid) {
    const details = validation.errors.map((e) => `  ${e.path}: ${e.message} [${e.code}]`).join("\n");
    throw new Error(`Suite failed validation — refusing to run an invalid suite:\n${details}`);
  }

  let testCases: TestCase[];
  if (Array.isArray(suite.test_cases) && suite.test_cases.length > 0) {
    testCases = suite.test_cases;
  } else if (typeof suite.test_cases_file === "string") {
    testCases = loadTestCasesFile(resolvePath(dirname(suitePath), suite.test_cases_file));
  } else {
    // Unreachable in practice — validateSuite() already requires one of these — but stay defensive.
    throw new Error("Suite has neither test_cases nor test_cases_file");
  }

  return { suite, testCases };
}

function loadTestCasesFile(path: string): TestCase[] {
  if (!existsSync(path)) throw new Error(`test_cases_file not found: ${path}`);
  const lines = readFileSync(path, "utf-8")
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  return lines.map((line, i) => {
    try {
      return JSON.parse(line) as TestCase;
    } catch (e) {
      throw new Error(`Invalid JSON on line ${i + 1} of ${path}: ${(e as Error).message}`);
    }
  });
}

function inputText(tc: TestCase): string {
  return Array.isArray(tc.input) ? tc.input.join("\n") : tc.input;
}

/** Merges a test-case-level provider override onto the suite/CLI-resolved
 * base config. Per SPEC.md, TestCase.provider can override model, api_base,
 * api_key_env, temperature, and max_tokens for that one test case; it cannot
 * switch vendor (openai vs anthropic) — that stays fixed for the whole run. */
function effectiveProviderConfig(base: ResolvedProviderConfig, override?: ProviderConfig): ResolvedProviderConfig {
  if (!override) return base;
  return {
    provider: base.provider,
    model: override.model ?? base.model,
    apiBase: override.api_base ?? base.apiBase,
    apiKeyEnv: override.api_key_env ?? base.apiKeyEnv,
    temperature: override.temperature ?? base.temperature,
    maxTokens: override.max_tokens ?? base.maxTokens,
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function withTimeout<T>(promise: Promise<T>, ms: number | undefined): Promise<T> {
  if (!ms || ms <= 0) return promise;
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new TimeoutError(ms)), ms);
    promise.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      },
    );
  });
}

/** Retries only ProviderHttpError with retryable:true (429s and 5xxs) —
 * everything else (bad auth, malformed request, missing key, timeout) fails
 * fast rather than repeating the same doomed call `maxAttempts` times. */
async function withRetry<T>(fn: () => Promise<T>, maxAttempts: number, backoffMs: number): Promise<T> {
  let attempt = 0;
  for (;;) {
    try {
      return await fn();
    } catch (e) {
      attempt++;
      const retryable = e instanceof ProviderHttpError && e.retryable;
      if (!retryable || attempt >= maxAttempts) throw e;
      await sleep(backoffMs * 2 ** (attempt - 1));
    }
  }
}

/** Simple bounded-concurrency worker pool — no external deps, no dropped
 * results, preserves result ordering by index regardless of completion order. */
async function runWithConcurrency<T, R>(items: T[], limit: number, worker: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  const concurrency = Math.max(1, Math.min(limit, items.length || 1));
  let nextIndex = 0;
  const runners = Array.from({ length: concurrency }, async () => {
    for (;;) {
      const i = nextIndex++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  });
  await Promise.all(runners);
  return results;
}

export interface RunDeps {
  /** Overridable for tests — default is the real HTTP client factory. */
  providerClientFactory?: (provider: ResolvedProviderConfig["provider"], apiBase: string, apiKeyEnv: string) => ProviderClient;
}

export interface RunResult {
  resultSet?: ResultSet;
  dryRun?: DryRunReport;
}

/** Builds a --dry-run cost estimate without making a single API call.
 * Heuristic by design (chars/4 token approximation, an assumed completion
 * length) — see cost.ts for the reasoning. Always run this and get explicit
 * user approval before spending real money, per the project's budget-gate rule. */
export function buildDryRunReport(suite: EvalSuite, testCases: TestCase[], resolvedProvider: ResolvedProviderConfig, embeddingModel: string): DryRunReport {
  const lineItems: DryRunLineItem[] = [];
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalCostUsd = 0;
  const judgeModels = new Set<string>();
  const warnings: string[] = [];
  const assumedCompletionTokens = resolvedProvider.maxTokens ?? ASSUMED_COMPLETION_TOKENS_FOR_DRY_RUN;

  for (const tc of testCases) {
    const text = inputText(tc);
    const model = tc.provider?.model ?? resolvedProvider.model;
    let tcInputTokens = estimateTokens(text);
    let tcOutputTokens = assumedCompletionTokens;
    let tcCostUsd = estimateCostUsd(model, tcInputTokens, tcOutputTokens);

    const graders = resolveGraders(suite, tc);
    const graderIds = graders.map((g) => g.id);
    for (const g of graders) {
      if (g.type === "llm_judge" || g.type === "model graded") {
        const judgeModel = typeof g.params?.model === "string" && g.params.model ? g.params.model : model;
        judgeModels.add(judgeModel);
        const promptTemplate = typeof g.params?.prompt === "string" ? g.params.prompt : "";
        const judgeInputTokens = estimateTokens(promptTemplate) + estimateTokens(text) + assumedCompletionTokens;
        tcInputTokens += judgeInputTokens;
        tcOutputTokens += ASSUMED_JUDGE_OUTPUT_TOKENS;
        tcCostUsd += estimateCostUsd(judgeModel, judgeInputTokens, ASSUMED_JUDGE_OUTPUT_TOKENS);
      } else if (g.type === "semantic_similarity") {
        const embModel = typeof g.params?.model === "string" && g.params.model ? g.params.model : embeddingModel;
        judgeModels.add(embModel);
        const embTokens = estimateTokens(text) * 2; // actual output + expected_output, both embedded
        tcInputTokens += embTokens;
        tcCostUsd += estimateCostUsd(embModel, embTokens, 0);
      }
    }

    lineItems.push({
      testCaseId: tc.id,
      graderIds,
      estimatedInputTokens: tcInputTokens,
      estimatedOutputTokens: tcOutputTokens,
      estimatedCostUsd: tcCostUsd,
    });
    totalInputTokens += tcInputTokens;
    totalOutputTokens += tcOutputTokens;
    totalCostUsd += tcCostUsd;
  }

  const { isKnown: modelPricingKnown } = pricingFor(resolvedProvider.model);
  if (!modelPricingKnown) {
    warnings.push(`Pricing for model "${resolvedProvider.model}" isn't in the known-pricing table — used a conservative fallback rate. Treat this as an upper bound, not an exact quote.`);
  }
  for (const jm of judgeModels) {
    if (!pricingFor(jm).isKnown) warnings.push(`Pricing for "${jm}" isn't in the known-pricing table — used a conservative fallback rate.`);
  }
  warnings.push("This is a heuristic estimate (chars/4 token approximation, an assumed completion length) computed before any real call. Actual spend will differ — use it to sanity-check budget, not as an invoice.");

  return {
    model: resolvedProvider.model,
    modelPricingKnown,
    testCaseCount: testCases.length,
    lineItems,
    totalEstimatedInputTokens: totalInputTokens,
    totalEstimatedOutputTokens: totalOutputTokens,
    totalEstimatedCostUsd: totalCostUsd,
    judgeModelsUsed: [...judgeModels],
    warnings,
  };
}

function ensureDirFor(filePath: string): void {
  const dir = dirname(filePath);
  if (dir && dir !== "." && !existsSync(dir)) mkdirSync(dir, { recursive: true });
}

interface RunTestCaseArgs {
  suite: EvalSuite;
  tc: TestCase;
  baseProviderConfig: ResolvedProviderConfig;
  clientFor: (cfg: ResolvedProviderConfig) => ProviderClient;
  graderClientsFor: (cfg: ResolvedProviderConfig) => GraderClients;
  suiteDefaultTimeoutMs?: number;
  maxAttempts: number;
  backoffMs: number;
}

async function runTestCase(args: RunTestCaseArgs): Promise<Result> {
  const { suite, tc, baseProviderConfig, clientFor, graderClientsFor, suiteDefaultTimeoutMs, maxAttempts, backoffMs } = args;
  const start = Date.now();
  const providerConfig = effectiveProviderConfig(baseProviderConfig, tc.provider);
  const client = clientFor(providerConfig);
  const text = inputText(tc);
  const timeoutMs = tc.timeout_ms ?? suiteDefaultTimeoutMs;

  let actualOutput: string | undefined;
  let inputTokens = 0;
  let outputTokens = 0;
  let error: Result["error"] | undefined;

  try {
    const completion = await withRetry(
      () => withTimeout(client.complete({ model: providerConfig.model, prompt: text, temperature: providerConfig.temperature, maxTokens: providerConfig.maxTokens }), timeoutMs),
      maxAttempts,
      backoffMs,
    );
    actualOutput = completion.text;
    inputTokens = completion.inputTokens;
    outputTokens = completion.outputTokens;
  } catch (e) {
    if (e instanceof TimeoutError) {
      error = { type: "timeout", message: e.message, retryable: false };
    } else if (e instanceof MissingApiKeyError) {
      error = { type: "runner_error", message: e.message, retryable: false };
    } else if (e instanceof ProviderHttpError) {
      error = { type: "provider_error", message: e.message, code: e.status, retryable: e.retryable };
    } else {
      error = { type: "runner_error", message: (e as Error).message, retryable: false };
    }
  }

  const durationMs = Date.now() - start;

  if (error) {
    return { test_case_id: tc.id, grader_results: [], passed: false, duration_ms: durationMs, error };
  }

  const graders = resolveGraders(suite, tc);
  const graderClients = graderClientsFor(providerConfig);
  const graderOutcomes = await Promise.all(
    graders.map(async (g) => {
      try {
        return await gradeOne(g, { actualOutput: actualOutput!, input: text, expectedOutput: tc.expected_output, context: tc.context }, graderClients);
      } catch (e) {
        // GRADER_ERROR per SPEC.md's error table: never let a grader bug crash the run.
        return { graderId: g.id, type: g.type, score: null, passed: false, reason: `grader threw: ${(e as Error).message}`, metadata: { error: (e as Error).message } };
      }
    }),
  );

  const graderResults: GraderResult[] = graderOutcomes.map((o) => ({
    grader_id: o.graderId,
    type: o.type,
    score: o.score,
    passed: o.passed,
    reason: o.reason,
    metadata: o.metadata,
  }));

  const passed = graderResults.length > 0 && graderResults.every((g) => g.passed);
  const graderCostUsd = graderOutcomes.reduce((sum, o) => sum + (o.costUsd ?? 0), 0);
  const completionCostUsd = estimateCostUsd(providerConfig.model, inputTokens, outputTokens);

  return {
    test_case_id: tc.id,
    actual_output: actualOutput,
    grader_results: graderResults,
    passed,
    duration_ms: durationMs,
    metadata: {
      openeval: {
        cost: {
          input_tokens: inputTokens,
          output_tokens: outputTokens,
          estimated_cost_usd: Number((completionCostUsd + graderCostUsd).toFixed(6)),
        },
      },
    },
  };
}

/** Runs (or dry-run-estimates) an eval suite end to end. This is the single
 * entry point `evalport run` calls — everything provider-related is
 * injectable via `deps.providerClientFactory` so tests never need a live key. */
export async function runEval(opts: RunOptions, deps: RunDeps = {}): Promise<RunResult> {
  const { suite, testCases: allTestCases } = loadSuite(opts.suitePath);
  const testCases = typeof opts.limit === "number" ? allTestCases.slice(0, Math.max(0, opts.limit)) : allTestCases;

  if (testCases.length === 0) {
    throw new Error("No test cases to run (the suite is empty, or --limit is 0).");
  }

  const suiteProviderCfg = suite.config?.provider;
  const model = opts.model ?? suiteProviderCfg?.model;
  if (!model) throw new Error("No model specified — pass --model, or set config.provider.model in the suite.");

  const baseProviderConfig = resolveProviderConfig({
    provider: opts.provider,
    model,
    apiBase: opts.apiBase ?? suiteProviderCfg?.api_base,
    apiKeyEnv: opts.apiKeyEnv ?? suiteProviderCfg?.api_key_env,
    temperature: opts.temperature ?? suiteProviderCfg?.temperature,
    maxTokens: opts.maxTokens ?? suiteProviderCfg?.max_tokens,
  });

  const embeddingModel = opts.embeddingModel ?? DEFAULT_EMBEDDING_MODEL;

  if (opts.dryRun) {
    return { dryRun: buildDryRunReport(suite, testCases, baseProviderConfig, embeddingModel) };
  }

  const factory = deps.providerClientFactory ?? createProviderClient;
  const clientCache = new Map<string, ProviderClient>();
  const clientFor = (cfg: ResolvedProviderConfig): ProviderClient => {
    const key = `${cfg.provider}::${cfg.apiBase}::${cfg.apiKeyEnv}`;
    let client = clientCache.get(key);
    if (!client) {
      client = factory(cfg.provider, cfg.apiBase, cfg.apiKeyEnv);
      clientCache.set(key, client);
    }
    return client;
  };

  const embeddingApiBase = opts.embeddingApiBase ?? DEFAULT_API_BASE.openai;
  const embeddingApiKeyEnv = opts.embeddingApiKeyEnv ?? DEFAULT_API_KEY_ENV.openai;
  const embeddingClient = factory("openai", embeddingApiBase, embeddingApiKeyEnv);
  const graderClientsFor = (cfg: ResolvedProviderConfig): GraderClients => ({
    judgeClient: clientFor(cfg),
    embeddingClient,
    defaultEmbeddingModel: embeddingModel,
  });

  const runId = opts.runId ?? `run_${suite.id}_${startTimestamp()}`;
  const startedAt = new Date().toISOString();
  const results: (Result | undefined)[] = new Array(testCases.length);
  const parallel = Math.max(1, opts.parallel || 1);
  const maxAttempts = Math.max(1, opts.maxAttempts || 1);
  const backoffMs = Math.max(0, opts.backoffMs ?? 0);
  const suiteDefaultTimeoutMs = suite.config?.defaults?.timeout_ms;

  const writePartial = (done: number) => {
    if (!opts.output) return;
    const partial: Partial<ResultSet> & { results: Result[] } = {
      version: OPENEVAL_VERSION,
      suite_id: suite.id,
      suite_version: suite.version,
      run_id: runId,
      started_at: startedAt,
      provider: { model: baseProviderConfig.model, api_base: baseProviderConfig.apiBase },
      runner: { name: RUNNER_NAME, version: RUNNER_VERSION },
      results: results.filter((r): r is Result => r !== undefined),
      metadata: { openeval: { partial: done < testCases.length } },
    };
    ensureDirFor(opts.output);
    writeFileSync(opts.output, JSON.stringify(partial, null, 2));
  };

  let completed = 0;
  await runWithConcurrency(testCases, parallel, async (tc, i) => {
    const result = await runTestCase({ suite, tc, baseProviderConfig, clientFor, graderClientsFor, suiteDefaultTimeoutMs, maxAttempts, backoffMs });
    results[i] = result;
    completed++;
    writePartial(completed);
    return result;
  });

  const finalResults = results.filter((r): r is Result => r !== undefined);
  const summary = computeSummary(finalResults);
  const totalCostUsd = finalResults.reduce((sum, r) => {
    const cost = (r.metadata as Record<string, unknown> | undefined)?.openeval as { cost?: { estimated_cost_usd?: number } } | undefined;
    return sum + (cost?.cost?.estimated_cost_usd ?? 0);
  }, 0);

  const resultSet: ResultSet = {
    version: OPENEVAL_VERSION,
    suite_id: suite.id,
    suite_version: suite.version,
    run_id: runId,
    started_at: startedAt,
    completed_at: new Date().toISOString(),
    provider: { model: baseProviderConfig.model, api_base: baseProviderConfig.apiBase },
    runner: { name: RUNNER_NAME, version: RUNNER_VERSION },
    results: finalResults,
    summary,
    metadata: { openeval: { cost: { total_estimated_cost_usd: Number(totalCostUsd.toFixed(6)) } } },
  };

  const selfValidation = validateResultSet(resultSet);
  if (!selfValidation.valid) {
    const details = selfValidation.errors.map((e) => `  ${e.path}: ${e.message} [${e.code}]`).join("\n");
    throw new Error(`Internal error: evalport run produced a ResultSet that fails its own spec validation — this is a runner bug, please file an issue:\n${details}`);
  }

  if (opts.output) {
    ensureDirFor(opts.output);
    writeFileSync(opts.output, JSON.stringify(resultSet, null, 2));
  }

  return { resultSet };
}

/** Millisecond timestamp used only to make a default run_id unique.
 * Kept as its own function so tests can see it's the only Date.now() call
 * in the hot path (everything else uses new Date().toISOString() for
 * human-readable timestamps in the output document). */
function startTimestamp(): number {
  return Date.now();
}
