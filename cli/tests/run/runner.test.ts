import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { buildDryRunReport, loadSuite, runEval } from "../../src/run/runner";
import { resolveProviderConfig } from "../../src/run/providers";
import type { CompletionResult, EmbeddingResult, ProviderClient, RunOptions } from "../../src/run/types";
import { validateResultSet } from "../../../sdk/typescript/src/validate";

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "evalport-runner-test-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function writeJson(name: string, obj: unknown): string {
  const p = join(dir, name);
  writeFileSync(p, JSON.stringify(obj, null, 2));
  return p;
}

const BASE_SUITE = {
  version: "1.0.0",
  id: "test_suite",
  graders: [
    { id: "gr_exact", type: "exact_match" },
    { id: "gr_judge", type: "llm_judge", params: { model: "gpt-4o-mini", prompt: "Score {output} vs {expected}.", threshold: 0.5 } },
  ],
  test_cases: [
    { id: "tc_exact", input: "2+2?", expected_output: "4", graders: ["gr_exact"] },
    { id: "tc_judge", input: "explain gravity", expected_output: "attraction between masses", graders: ["gr_judge"] },
  ],
  config: { provider: { model: "gpt-4o-mini", temperature: 0 } },
};

function baseOpts(overrides: Partial<RunOptions> = {}): RunOptions {
  return {
    suitePath: "",
    provider: "openai",
    model: "gpt-4o-mini",
    parallel: 1,
    dryRun: false,
    maxAttempts: 3,
    backoffMs: 1,
    ...overrides,
  } as RunOptions;
}

function client(overrides: Partial<ProviderClient> = {}): ProviderClient {
  return {
    complete: vi.fn(async (): Promise<CompletionResult> => ({ text: "4", inputTokens: 1, outputTokens: 1 })),
    embed: vi.fn(async (): Promise<EmbeddingResult> => ({ vector: [1, 0], inputTokens: 1 })),
    ...overrides,
  };
}

describe("loadSuite", () => {
  test("loads inline test_cases from a valid suite", () => {
    const p = writeJson("suite.json", BASE_SUITE);
    const { suite, testCases } = loadSuite(p);
    expect(suite.id).toBe("test_suite");
    expect(testCases).toHaveLength(2);
  });

  test("rejects a suite that fails validateSuite(), with details in the message", () => {
    const p = writeJson("bad.json", { version: "not-semver", id: "x" });
    expect(() => loadSuite(p)).toThrow(/failed validation/);
  });

  test("throws a clear error for a missing file", () => {
    expect(() => loadSuite(join(dir, "nope.json"))).toThrow(/not found/);
  });

  test("throws a clear error for malformed JSON", () => {
    const p = join(dir, "broken.json");
    writeFileSync(p, "{not json");
    expect(() => loadSuite(p)).toThrow(/not valid JSON/);
  });

  test("resolves test_cases_file (JSONL) relative to the suite file", () => {
    writeFileSync(
      join(dir, "cases.jsonl"),
      ['{"id":"tc_001","input":"a","expected_output":"a","graders":["gr_exact"]}', '{"id":"tc_002","input":"b","expected_output":"b","graders":["gr_exact"]}'].join("\n"),
    );
    const p = writeJson("suite_with_file.json", { version: "1.0.0", id: "s", graders: [{ id: "gr_exact", type: "exact_match" }], test_cases_file: "cases.jsonl" });
    const { testCases } = loadSuite(p);
    expect(testCases.map((t) => t.id)).toEqual(["tc_001", "tc_002"]);
  });

  test("throws with a line number for malformed JSONL", () => {
    writeFileSync(join(dir, "cases.jsonl"), '{"id":"tc_001","input":"a","graders":["gr_exact"]}\nnot json\n');
    const p = writeJson("suite2.json", { version: "1.0.0", id: "s", graders: [{ id: "gr_exact", type: "exact_match" }], test_cases_file: "cases.jsonl" });
    expect(() => loadSuite(p)).toThrow(/line 2/);
  });
});

describe("buildDryRunReport", () => {
  test("produces a line item per test case with cost estimates, and never calls a provider", () => {
    const { suite, testCases } = loadSuite(writeJson("suite.json", BASE_SUITE));
    const cfg = resolveProviderConfig({ provider: "openai", model: "gpt-4o-mini" });
    const report = buildDryRunReport(suite, testCases, cfg, "text-embedding-3-small");
    expect(report.testCaseCount).toBe(2);
    expect(report.lineItems).toHaveLength(2);
    expect(report.totalEstimatedCostUsd).toBeGreaterThan(0);
    expect(report.judgeModelsUsed).toContain("gpt-4o-mini");
  });

  test("warns when a model's pricing is unknown", () => {
    const { suite, testCases } = loadSuite(writeJson("suite.json", BASE_SUITE));
    const cfg = resolveProviderConfig({ provider: "openai", model: "totally-unpriced-model" });
    const report = buildDryRunReport(suite, testCases, cfg, "text-embedding-3-small");
    expect(report.modelPricingKnown).toBe(false);
    expect(report.warnings.some((w) => w.includes("totally-unpriced-model"))).toBe(true);
  });
});

describe("runEval — dry run", () => {
  test("returns a report and never constructs a provider client", async () => {
    const p = writeJson("suite.json", BASE_SUITE);
    const factory = vi.fn(() => {
      throw new Error("must not be called during --dry-run");
    });
    const { dryRun, resultSet } = await runEval(baseOpts({ suitePath: p, dryRun: true }), { providerClientFactory: factory });
    expect(dryRun).toBeDefined();
    expect(resultSet).toBeUndefined();
    expect(factory).not.toHaveBeenCalled();
  });
});

describe("runEval — real run", () => {
  test("produces a spec-valid, self-validated ResultSet with correct grading", async () => {
    const p = writeJson("suite.json", BASE_SUITE);
    const c = client({
      complete: vi.fn(async ({ prompt }: { prompt: string }) => {
        if (prompt.includes("2+2")) return { text: "4", inputTokens: 5, outputTokens: 1 };
        return { text: '{"score": 0.9}', inputTokens: 20, outputTokens: 5 };
      }),
    });
    const { resultSet } = await runEval(baseOpts({ suitePath: p }), { providerClientFactory: () => c });
    expect(resultSet).toBeDefined();
    expect(validateResultSet(resultSet).valid).toBe(true);
    const byId = Object.fromEntries(resultSet!.results.map((r) => [r.test_case_id, r]));
    expect(byId.tc_exact.passed).toBe(true);
    expect(byId.tc_judge.passed).toBe(true); // score 0.9 >= threshold 0.5
    expect(resultSet!.summary?.total).toBe(2);
    expect(resultSet!.summary?.passed).toBe(2);
  });

  test("marks a test case failed when the grader doesn't pass, without erroring the run", async () => {
    const p = writeJson("suite.json", BASE_SUITE);
    const c = client({ complete: vi.fn(async () => ({ text: "wrong answer", inputTokens: 1, outputTokens: 1 })) });
    const { resultSet } = await runEval(baseOpts({ suitePath: p, limit: 1 }), { providerClientFactory: () => c });
    expect(resultSet!.results[0].passed).toBe(false);
  });

  test("--limit truncates the test cases actually run", async () => {
    const p = writeJson("suite.json", BASE_SUITE);
    const c = client();
    const { resultSet } = await runEval(baseOpts({ suitePath: p, limit: 1 }), { providerClientFactory: () => c });
    expect(resultSet!.results).toHaveLength(1);
    expect(resultSet!.results[0].test_case_id).toBe("tc_exact");
  });

  test("limit:0 throws rather than silently running everything or nothing", async () => {
    const p = writeJson("suite.json", BASE_SUITE);
    await expect(runEval(baseOpts({ suitePath: p, limit: 0 }), { providerClientFactory: () => client() })).rejects.toThrow(/No test cases to run/);
  });

  test("no model from CLI or suite config throws a clear error", async () => {
    const suiteNoModel = { ...BASE_SUITE, config: undefined };
    const p = writeJson("suite.json", suiteNoModel);
    await expect(runEval(baseOpts({ suitePath: p, model: undefined }), { providerClientFactory: () => client() })).rejects.toThrow(/No model specified/);
  });

  test("falls back to the suite's config.provider.model when --model is omitted", async () => {
    const p = writeJson("suite.json", BASE_SUITE);
    const complete = vi.fn(async (_args: { model: string }) => ({ text: "4", inputTokens: 1, outputTokens: 1 }));
    await runEval(baseOpts({ suitePath: p, model: undefined, limit: 1 }), { providerClientFactory: () => client({ complete }) });
    expect(complete.mock.calls[0][0].model).toBe("gpt-4o-mini");
  });

  test("preserves result ordering under concurrency (--parallel > 1)", async () => {
    const manyCases = {
      ...BASE_SUITE,
      test_cases: Array.from({ length: 8 }, (_, i) => ({ id: `tc_${i}`, input: `${i}`, expected_output: `${i}`, graders: ["gr_exact"] })),
    };
    const p = writeJson("suite.json", manyCases);
    const c = client({
      complete: vi.fn(async ({ prompt }: { prompt: string }) => {
        await new Promise((r) => setTimeout(r, Math.random() * 5));
        return { text: prompt, inputTokens: 1, outputTokens: 1 };
      }),
    });
    const { resultSet } = await runEval(baseOpts({ suitePath: p, parallel: 4 }), { providerClientFactory: () => c });
    expect(resultSet!.results.map((r) => r.test_case_id)).toEqual(manyCases.test_cases.map((t) => t.id));
  });

  test("retries a retryable provider error and succeeds on a later attempt", async () => {
    let attempts = 0;
    const c = client({
      complete: vi.fn(async () => {
        attempts++;
        if (attempts < 3) {
          const { ProviderHttpError } = await import("../../src/run/providers");
          throw new ProviderHttpError(429, "rate limited", true);
        }
        return { text: "4", inputTokens: 1, outputTokens: 1 };
      }),
    });
    const p = writeJson("suite.json", BASE_SUITE);
    const { resultSet } = await runEval(baseOpts({ suitePath: p, limit: 1, maxAttempts: 5, backoffMs: 1 }), { providerClientFactory: () => c });
    expect(attempts).toBe(3);
    expect(resultSet!.results[0].passed).toBe(true);
  });

  test("gives up after maxAttempts and records a provider_error result", async () => {
    const { ProviderHttpError } = await import("../../src/run/providers");
    const c = client({
      complete: vi.fn(async () => {
        throw new ProviderHttpError(503, "down", true);
      }),
    });
    const p = writeJson("suite.json", BASE_SUITE);
    const { resultSet } = await runEval(baseOpts({ suitePath: p, limit: 1, maxAttempts: 2, backoffMs: 1 }), { providerClientFactory: () => c });
    expect(c.complete).toHaveBeenCalledTimes(2);
    expect(resultSet!.results[0].passed).toBe(false);
    expect(resultSet!.results[0].error?.type).toBe("provider_error");
    expect(resultSet!.results[0].error?.retryable).toBe(true);
  });

  test("does not retry a non-retryable provider error", async () => {
    const { ProviderHttpError } = await import("../../src/run/providers");
    const complete = vi.fn(async () => {
      throw new ProviderHttpError(401, "bad key", false);
    });
    const c = client({ complete });
    const p = writeJson("suite.json", BASE_SUITE);
    const { resultSet } = await runEval(baseOpts({ suitePath: p, limit: 1, maxAttempts: 5, backoffMs: 1 }), { providerClientFactory: () => c });
    expect(complete).toHaveBeenCalledTimes(1);
    expect(resultSet!.results[0].error?.type).toBe("provider_error");
    expect(resultSet!.results[0].error?.retryable).toBe(false);
  });

  test("a missing API key surfaces as a runner_error result, not a crash", async () => {
    const { MissingApiKeyError } = await import("../../src/run/providers");
    const complete = vi.fn(async () => {
      throw new MissingApiKeyError("OPENAI_API_KEY", "openai");
    });
    const c = client({ complete });
    const p = writeJson("suite.json", BASE_SUITE);
    const { resultSet } = await runEval(baseOpts({ suitePath: p, limit: 1 }), { providerClientFactory: () => c });
    expect(resultSet!.results[0].error?.type).toBe("runner_error");
    expect(resultSet!.results[0].error?.retryable).toBe(false);
  });

  test("a test case that exceeds timeout_ms is recorded as a timeout error", async () => {
    const suite = { ...BASE_SUITE, test_cases: [{ id: "tc_slow", input: "x", expected_output: "x", graders: ["gr_exact"], timeout_ms: 15 }] };
    const p = writeJson("suite.json", suite);
    const c = client({
      complete: vi.fn(async () => {
        await new Promise((r) => setTimeout(r, 200));
        return { text: "x", inputTokens: 1, outputTokens: 1 };
      }),
    });
    const { resultSet } = await runEval(baseOpts({ suitePath: p, maxAttempts: 1 }), { providerClientFactory: () => c });
    expect(resultSet!.results[0].error?.type).toBe("timeout");
  });

  test("an unsupported grader type is skipped cleanly and the test case still fails per spec", async () => {
    const suite = {
      version: "1.0.0",
      id: "s",
      test_cases: [{ id: "tc1", input: "x", expected_output: "x", graders: [{ id: "gr_custom", type: "custom", params: { handler: "nope" } }] }],
      config: { provider: { model: "gpt-4o-mini" } },
    };
    const p = writeJson("suite.json", suite);
    const { resultSet } = await runEval(baseOpts({ suitePath: p }), { providerClientFactory: () => client() });
    expect(resultSet!.results[0].grader_results[0].metadata?.skip_reason).toBe("unsupported_grader_type");
    expect(resultSet!.results[0].passed).toBe(false);
  });

  test("test-case-level provider.model override changes the model used for that call", async () => {
    const suite = { ...BASE_SUITE, test_cases: [{ id: "tc1", input: "x", expected_output: "x", graders: ["gr_exact"], provider: { model: "gpt-4o" } }] };
    const p = writeJson("suite.json", suite);
    const complete = vi.fn(async (_args: { model: string }) => ({ text: "x", inputTokens: 1, outputTokens: 1 }));
    await runEval(baseOpts({ suitePath: p }), { providerClientFactory: () => client({ complete }) });
    expect(complete.mock.calls[0][0].model).toBe("gpt-4o");
  });

  test("embedding calls always go through an OpenAI-compatible client, even when --provider is anthropic", async () => {
    const suite = {
      version: "1.0.0",
      id: "s",
      graders: [{ id: "gr_sim", type: "semantic_similarity", params: { threshold: 0.1 } }],
      test_cases: [{ id: "tc1", input: "x", expected_output: "y", graders: ["gr_sim"] }],
      config: { provider: { model: "claude-3-5-sonnet-20241022" } },
    };
    const p = writeJson("suite.json", suite);
    const factory = vi.fn((provider: string) => client());
    await runEval(baseOpts({ suitePath: p, provider: "anthropic", model: "claude-3-5-sonnet-20241022" }), { providerClientFactory: factory });
    const providersUsed = factory.mock.calls.map((c) => c[0]);
    expect(providersUsed).toContain("openai");
    expect(providersUsed).toContain("anthropic");
  });

  test("writes the ResultSet to --output, matching the returned object", async () => {
    const p = writeJson("suite.json", BASE_SUITE);
    const outPath = join(dir, "out.json");
    const { resultSet } = await runEval(baseOpts({ suitePath: p, output: outPath }), { providerClientFactory: () => client() });
    expect(existsSync(outPath)).toBe(true);
    const onDisk = JSON.parse(readFileSync(outPath, "utf-8"));
    expect(onDisk.run_id).toBe(resultSet!.run_id);
    expect(onDisk.results).toHaveLength(2);
    expect(validateResultSet(onDisk).valid).toBe(true);
  });

  test("an invalid-JSON output fails a json_schema grader cleanly (no crash)", async () => {
    const suite = {
      version: "1.0.0",
      id: "s",
      graders: [{ id: "gr_bad_schema", type: "json_schema", params: { schema: { type: "object" } } }],
      test_cases: [{ id: "tc1", input: "x", expected_output: "x", graders: ["gr_bad_schema"] }],
      config: { provider: { model: "gpt-4o-mini" } },
    };
    const p = writeJson("suite.json", suite);
    const c = client({ complete: vi.fn(async () => ({ text: "not json at all", inputTokens: 1, outputTokens: 1 })) });
    const { resultSet } = await runEval(baseOpts({ suitePath: p }), { providerClientFactory: () => c });
    expect(resultSet!.results[0].passed).toBe(false);
    expect(resultSet!.results[0].grader_results[0].reason).toMatch(/not valid JSON/);
  });

  test("a grader implementation that throws is caught and recorded as a GRADER_ERROR result, not a crashed run", async () => {
    const suite = {
      version: "1.0.0",
      id: "s",
      graders: [{ id: "gr_sim", type: "semantic_similarity", params: { threshold: 0.5 } }],
      test_cases: [{ id: "tc1", input: "x", expected_output: "y", graders: ["gr_sim"] }],
      config: { provider: { model: "gpt-4o-mini" } },
    };
    const p = writeJson("suite.json", suite);
    const c = client({
      embed: vi.fn(async () => {
        throw new Error("embedding backend exploded unexpectedly");
      }),
    });
    const { resultSet } = await runEval(baseOpts({ suitePath: p }), { providerClientFactory: () => c });
    expect(resultSet!.results[0].passed).toBe(false);
    const gr = resultSet!.results[0].grader_results[0];
    expect(gr.score).toBeNull();
    expect(gr.reason).toMatch(/grader threw/);
    expect(gr.metadata?.error).toMatch(/embedding backend exploded/);
  });
});
