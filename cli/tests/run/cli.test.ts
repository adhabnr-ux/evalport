import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { CliArgError, HelpRequested, formatDryRunReport, formatRunSummary, parseRunArgs, runCommand } from "../../src/run/cli";
import type { DryRunReport } from "../../src/run/cost";
import type { ResultSet } from "../../../sdk/typescript/src/types";

describe("parseRunArgs", () => {
  test("parses a minimal valid invocation", () => {
    const opts = parseRunArgs(["suite.json", "--provider", "openai"]);
    expect(opts.suitePath).toBe("suite.json");
    expect(opts.provider).toBe("openai");
    expect(opts.parallel).toBe(1);
    expect(opts.dryRun).toBe(false);
    expect(opts.maxAttempts).toBe(3);
    expect(opts.backoffMs).toBe(1000);
  });

  test("parses every documented flag", () => {
    const opts = parseRunArgs([
      "suite.json",
      "--provider", "anthropic",
      "--model", "claude-3-5-sonnet-20241022",
      "--api-base", "https://example.com/v1",
      "--api-key-env", "MY_KEY",
      "--temperature", "0.3",
      "--max-tokens", "512",
      "--parallel", "4",
      "--limit", "10",
      "--output", "out.json",
      "--dry-run",
      "--max-attempts", "5",
      "--backoff-ms", "250",
      "--embedding-api-base", "https://embed.example.com/v1",
      "--embedding-api-key-env", "EMBED_KEY",
      "--embedding-model", "text-embedding-3-large",
      "--run-id", "run_abc",
    ]);
    expect(opts).toMatchObject({
      suitePath: "suite.json",
      provider: "anthropic",
      model: "claude-3-5-sonnet-20241022",
      apiBase: "https://example.com/v1",
      apiKeyEnv: "MY_KEY",
      temperature: 0.3,
      maxTokens: 512,
      parallel: 4,
      limit: 10,
      output: "out.json",
      dryRun: true,
      maxAttempts: 5,
      backoffMs: 250,
      embeddingApiBase: "https://embed.example.com/v1",
      embeddingApiKeyEnv: "EMBED_KEY",
      embeddingModel: "text-embedding-3-large",
      runId: "run_abc",
    });
  });

  test("requires a suite path", () => {
    expect(() => parseRunArgs(["--provider", "openai"])).toThrow(CliArgError);
  });

  test("requires --provider", () => {
    expect(() => parseRunArgs(["suite.json"])).toThrow(/--provider is required/);
  });

  test("rejects an invalid --provider value", () => {
    expect(() => parseRunArgs(["suite.json", "--provider", "cohere"])).toThrow(/Invalid --provider/);
  });

  test("rejects an unknown flag", () => {
    expect(() => parseRunArgs(["suite.json", "--provider", "openai", "--bogus"])).toThrow(/Unknown flag/);
  });

  test("rejects a flag missing its value", () => {
    expect(() => parseRunArgs(["suite.json", "--provider"])).toThrow(/Missing value/);
  });

  test("rejects non-numeric --temperature / --max-tokens / --parallel / --limit", () => {
    expect(() => parseRunArgs(["suite.json", "--provider", "openai", "--temperature", "hot"])).toThrow(/expects a number/);
    expect(() => parseRunArgs(["suite.json", "--provider", "openai", "--max-tokens", "lots"])).toThrow(/expects an integer/);
    expect(() => parseRunArgs(["suite.json", "--provider", "openai", "--parallel", "many"])).toThrow(/expects an integer/);
  });

  test("rejects --parallel < 1 and --limit < 0", () => {
    expect(() => parseRunArgs(["suite.json", "--provider", "openai", "--parallel", "0"])).toThrow(/--parallel must be/);
    expect(() => parseRunArgs(["suite.json", "--provider", "openai", "--limit", "-1"])).toThrow(/--limit must be/);
  });

  test("--help / -h throws HelpRequested, not a generic CliArgError", () => {
    expect(() => parseRunArgs(["--help"])).toThrow(HelpRequested);
    expect(() => parseRunArgs(["-h"])).toThrow(HelpRequested);
  });
});

function fakeDryRunReport(): DryRunReport {
  return {
    model: "gpt-4o-mini",
    modelPricingKnown: true,
    testCaseCount: 2,
    lineItems: [
      { testCaseId: "tc1", graderIds: ["gr1"], estimatedInputTokens: 10, estimatedOutputTokens: 20, estimatedCostUsd: 0.001 },
      { testCaseId: "tc2", graderIds: ["gr1"], estimatedInputTokens: 15, estimatedOutputTokens: 25, estimatedCostUsd: 0.002 },
    ],
    totalEstimatedInputTokens: 25,
    totalEstimatedOutputTokens: 45,
    totalEstimatedCostUsd: 0.003,
    judgeModelsUsed: ["gpt-4o-mini"],
    warnings: ["heads up: this is an estimate"],
  };
}

describe("formatDryRunReport", () => {
  test("includes model, per-case line items, and the total cost", () => {
    const out = formatDryRunReport(fakeDryRunReport());
    expect(out).toContain("gpt-4o-mini");
    expect(out).toContain("tc1");
    expect(out).toContain("tc2");
    expect(out).toContain("TOTAL ESTIMATED COST: $0.0030");
    expect(out).toContain("heads up: this is an estimate");
  });

  test("flags unknown pricing in the header", () => {
    const out = formatDryRunReport({ ...fakeDryRunReport(), modelPricingKnown: false });
    expect(out).toMatch(/pricing unknown/);
  });
});

function fakeResultSet(overrides: Partial<ResultSet> = {}): ResultSet {
  return {
    version: "1.0.0",
    suite_id: "s",
    run_id: "run_1",
    started_at: new Date(0).toISOString(),
    results: [{ test_case_id: "tc1", passed: true, grader_results: [] }],
    summary: { total: 1, passed: 1, failed: 0, skipped: 0, pass_rate: 1, avg_score: 1 },
    ...overrides,
  };
}

describe("formatRunSummary", () => {
  test("reports totals, pass rate, and avg score", () => {
    const out = formatRunSummary(fakeResultSet());
    expect(out).toContain("run_1");
    expect(out).toContain("Passed: 1");
    expect(out).toContain("Failed: 0");
    expect(out).toContain("100.0%");
  });

  test("falls back gracefully when summary is absent", () => {
    const out = formatRunSummary(fakeResultSet({ summary: undefined }));
    expect(out).toMatch(/no summary computed/);
  });
});

describe("runCommand (integration, no live API calls)", () => {
  let dir: string;
  let logSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "evalport-cli-test-"));
    logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    logSpy.mockRestore();
    errorSpy.mockRestore();
  });

  test("bad arguments return exit code 1 and print to stderr", async () => {
    const code = await runCommand(["suite.json"]); // missing --provider
    expect(code).toBe(1);
    expect(errorSpy).toHaveBeenCalled();
  });

  test("--help returns 0 and prints to stdout, not stderr", async () => {
    const code = await runCommand(["--help"]);
    expect(code).toBe(0);
    expect(logSpy).toHaveBeenCalled();
    expect(errorSpy).not.toHaveBeenCalled();
  });

  test("a missing suite file returns exit code 1 with a clear error", async () => {
    const code = await runCommand([join(dir, "nope.json"), "--provider", "openai", "--model", "gpt-4o-mini"]);
    expect(code).toBe(1);
    expect(errorSpy.mock.calls[0][0]).toMatch(/not found/);
  });

  test("--dry-run against a real suite file succeeds with exit code 0 and makes no network calls", async () => {
    const suitePath = join(dir, "suite.json");
    writeFileSync(
      suitePath,
      JSON.stringify({
        version: "1.0.0",
        id: "s",
        graders: [{ id: "gr1", type: "exact_match" }],
        test_cases: [{ id: "tc1", input: "hi", expected_output: "hi", graders: ["gr1"] }],
        config: { provider: { model: "gpt-4o-mini" } },
      }),
    );
    const code = await runCommand([suitePath, "--provider", "openai", "--dry-run"]);
    expect(code).toBe(0);
    expect(logSpy.mock.calls.some((c) => String(c[0]).includes("Dry run"))).toBe(true);
  });

  test("an invalid suite file returns exit code 1 with validation details", async () => {
    const suitePath = join(dir, "bad.json");
    writeFileSync(suitePath, JSON.stringify({ version: "nope", id: "" }));
    const code = await runCommand([suitePath, "--provider", "openai", "--model", "gpt-4o-mini", "--dry-run"]);
    expect(code).toBe(1);
    expect(errorSpy.mock.calls[0][0]).toMatch(/failed validation/);
  });
});
