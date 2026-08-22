import { test, expect, describe } from "vitest";
import { computeSummary, createResultSet, fromPromptfoo } from "../src/convert";
import { OPENEVAL_VERSION } from "../src/types";
import { validateSuite, validateResultSet } from "../src/validate";

// Tests for src/convert.ts: fromPromptfoo(), computeSummary(), createResultSet().
//
// None of these three functions had any test coverage on the TypeScript side
// before this file -- src/convert.ts existed, was imported by the CLI, and had
// zero direct tests, despite sdk/python/tests/test_convert.py covering the
// equivalent Python functions in detail. Added alongside the 1.0.0-rc.4
// OPENEVAL_VERSION bump specifically to close that gap: this file's first test
// (below) is the same version-drift regression guard test_convert.py already
// had on the Python side, asserted against the *live* OPENEVAL_VERSION import,
// not a hardcoded literal -- which is exactly the class of bug that let
// OPENEVAL_VERSION itself silently go stale for a full spec revision on the
// Python side before it was caught in 1.0.0-rc.3.
//
// Assertions here validate against the real validateSuite()/validateResultSet()
// (src/validate.ts), the same hand-rolled validator every other TypeScript SDK
// test uses -- not shape assertions against a mocked or simplified checker.
//
// The promptfoo `tests`/`assert`/`providers` shape mirrors Promptfoo's real,
// documented YAML/JS test-case format
// (https://www.promptfoo.dev/docs/configuration/test-cases/), matching the
// fixtures test_convert.py already uses for the Python side of the same
// converter logic.

describe("fromPromptfoo", () => {
  test("produces a spec-valid suite stamped with the live OPENEVAL_VERSION", () => {
    const pf = {
      tests: [
        {
          vars: { query: "What is the capital of France?" },
          assert: [{ type: "equals", value: "Paris" }],
        },
        {
          vars: { query: "Name a fruit." },
          assert: [{ type: "contains", value: "apple" }],
        },
      ],
      providers: [{ id: "openai:gpt-4o-mini", model: "gpt-4o-mini" }],
    };
    const suite = fromPromptfoo(pf);
    const result = validateSuite(suite);
    expect(result.valid, JSON.stringify(result.errors)).toBe(true);
    // Asserted against the live constant, not a hardcoded literal -- see this
    // file's module comment for why that distinction matters here specifically.
    expect(suite.version).toBe(OPENEVAL_VERSION);
    expect(suite.test_cases).toHaveLength(2);
  });

  test("maps 'equals' assert to an exact_match grader", () => {
    const pf = { tests: [{ vars: { query: "2+2?" }, assert: [{ type: "equals", value: "4" }] }] };
    const suite = fromPromptfoo(pf);
    expect(suite.graders![0].type).toBe("exact_match");
    expect(suite.test_cases![0].graders).toEqual([suite.graders![0].id]);
  });

  test("maps 'contains' assert to a contains grader with a substring param", () => {
    const pf = { tests: [{ vars: { query: "list a color" }, assert: [{ type: "contains", value: "blue" }] }] };
    const suite = fromPromptfoo(pf);
    const grader = suite.graders![0];
    expect(grader.type).toBe("contains");
    expect(grader.params?.substring).toBe("blue");
    expect(validateSuite(suite).valid).toBe(true);
  });

  test("maps an unrecognized assert type to a custom grader with a promptfoo: handler", () => {
    const pf = {
      tests: [
        { vars: { query: "translate hello" }, assert: [{ type: "llm-rubric", value: "is a translation" }] },
      ],
    };
    const suite = fromPromptfoo(pf);
    const grader = suite.graders![0];
    expect(grader.type).toBe("custom");
    expect(grader.params?.handler).toBe("promptfoo:llm-rubric");
    expect(validateSuite(suite).valid).toBe(true);
  });

  test("preserves expected_output from vars.expected", () => {
    const pf = {
      tests: [
        { vars: { query: "2+2?", expected: "4" }, assert: [{ type: "equals", value: "4" }] },
      ],
    };
    const suite = fromPromptfoo(pf);
    expect(suite.test_cases![0].expected_output).toBe("4");
  });

  test("a test case with no asserts falls back to a shared gr_default exact_match grader", () => {
    // graders.minItems: 1 on TestCase means a promptfoo test with no `assert`
    // list still has to satisfy EvalPort's schema -- must not produce an
    // invalid TestCase with an empty graders array.
    const pf = { tests: [{ vars: { query: "just checking it runs" } }] };
    const suite = fromPromptfoo(pf);
    expect(suite.test_cases![0].graders).toEqual(["gr_default"]);
    expect(suite.graders).toEqual([{ id: "gr_default", type: "exact_match" }]);
    expect(validateSuite(suite).valid).toBe(true);
  });

  test("falls back to vars.prompt as input when vars.query is absent", () => {
    const pf = {
      tests: [{ vars: { prompt: "Summarize this document." }, assert: [{ type: "equals", value: "ok" }] }],
    };
    const suite = fromPromptfoo(pf);
    expect(suite.test_cases![0].input).toBe("Summarize this document.");
  });

  test("extracts the first provider's model into suite.config.provider", () => {
    const pf = {
      tests: [{ vars: { query: "hi" }, assert: [{ type: "equals", value: "hello" }] }],
      providers: [{ id: "anthropic:claude-3-5-sonnet", model: "claude-3-5-sonnet-20241022" }],
    };
    const suite = fromPromptfoo(pf);
    expect(suite.config?.provider?.model).toBe("claude-3-5-sonnet-20241022");
  });

  test("handles multiple asserts on one test case as multiple graders", () => {
    const pf = {
      tests: [
        {
          vars: { query: "describe a dog" },
          assert: [
            { type: "contains", value: "animal" },
            { type: "equals", value: "A dog is a domesticated animal." },
          ],
        },
      ],
    };
    const suite = fromPromptfoo(pf);
    expect(suite.graders).toHaveLength(2);
    expect(suite.test_cases![0].graders).toHaveLength(2);
    expect(validateSuite(suite).valid).toBe(true);
  });

  test("an empty tests list still gets the default grader and an empty test_cases array", () => {
    const suite = fromPromptfoo({ tests: [] });
    expect(suite.graders).toEqual([{ id: "gr_default", type: "exact_match" }]);
    expect(suite.test_cases).toEqual([]);
  });
});

describe("computeSummary", () => {
  test("counts pass/fail and averages scores across all grader_results", () => {
    const results = [
      {
        test_case_id: "tc1",
        passed: true,
        grader_results: [
          { grader_id: "g1", type: "exact_match", score: 1.0, passed: true },
          { grader_id: "g2", type: "exact_match", score: 0.8, passed: true },
        ],
      },
      {
        test_case_id: "tc2",
        passed: false,
        grader_results: [{ grader_id: "g1", type: "exact_match", score: 0.0, passed: false }],
      },
    ];
    const summary = computeSummary(results as any);
    expect(summary.total).toBe(2);
    expect(summary.passed).toBe(1);
    expect(summary.failed).toBe(1);
    expect(summary.pass_rate).toBe(0.5);
    expect(summary.avg_score).toBeCloseTo((1.0 + 0.8 + 0.0) / 3, 9);
  });

  test("ignores null scores when averaging", () => {
    const results = [
      {
        test_case_id: "tc1",
        passed: true,
        grader_results: [
          { grader_id: "g1", type: "custom", score: null, passed: false },
          { grader_id: "g2", type: "exact_match", score: 1.0, passed: true },
        ],
      },
    ];
    const summary = computeSummary(results as any);
    expect(summary.avg_score).toBe(1.0);
  });

  test("an empty results array does not divide by zero", () => {
    const summary = computeSummary([]);
    expect(summary.total).toBe(0);
    expect(summary.pass_rate).toBe(0);
    expect(summary.avg_score).toBe(0);
  });

  test("aggregates per-grader pass/fail/avg_score in by_grader", () => {
    const results = [
      {
        test_case_id: "tc1",
        passed: true,
        grader_results: [{ grader_id: "g1", type: "exact_match", score: 1.0, passed: true }],
      },
      {
        test_case_id: "tc2",
        passed: false,
        grader_results: [{ grader_id: "g1", type: "exact_match", score: 0.0, passed: false }],
      },
    ];
    const summary = computeSummary(results as any);
    expect(summary.by_grader?.g1.passed).toBe(1);
    expect(summary.by_grader?.g1.failed).toBe(1);
    expect(summary.by_grader?.g1.avg_score).toBe(0.5);
  });
});

describe("createResultSet", () => {
  test("produces a spec-valid ResultSet stamped with the live OPENEVAL_VERSION", () => {
    const pf = { tests: [{ vars: { query: "2+2?" }, assert: [{ type: "equals", value: "4" }] }] };
    const suite = fromPromptfoo(pf);
    const tc = suite.test_cases![0];
    const graderId = tc.graders[0] as string;
    const results = [
      {
        test_case_id: tc.id,
        passed: true,
        grader_results: [{ grader_id: graderId, type: "exact_match", score: 1.0, passed: true }],
      },
    ];
    const rs = createResultSet(suite, results as any, "run-1");
    const validation = validateResultSet(rs);
    expect(validation.valid, JSON.stringify(validation.errors)).toBe(true);
    expect(rs.version).toBe(OPENEVAL_VERSION);
    expect(rs.suite_id).toBe(suite.id);
    expect(rs.run_id).toBe("run-1");
    expect(rs.summary?.total).toBe(1);
    expect(rs.summary?.passed).toBe(1);
  });

  test("carries the provider config through from suite.config.provider", () => {
    const suite = {
      version: "1.0.0-rc.1",
      id: "s1",
      test_cases: [{ id: "tc1", input: "hi", graders: ["g1"] }],
      graders: [{ id: "g1", type: "exact_match" }],
      config: { provider: { model: "gpt-4o-mini" } },
    } as any;
    const results = [
      {
        test_case_id: "tc1",
        passed: true,
        grader_results: [{ grader_id: "g1", type: "exact_match", score: 1.0, passed: true }],
      },
    ];
    const rs = createResultSet(suite, results, "run-2");
    expect(rs.provider).toEqual({ model: "gpt-4o-mini" });
    expect(validateResultSet(rs).valid).toBe(true);
  });

  test("end-to-end: promptfoo import -> suite -> simulated grading -> ResultSet, both validate", () => {
    const pf = {
      tests: [
        { vars: { query: "2+2?" }, assert: [{ type: "equals", value: "4" }] },
        { vars: { query: "list a fruit" }, assert: [{ type: "contains", value: "apple" }] },
      ],
      providers: [{ id: "openai:gpt-4o-mini", model: "gpt-4o-mini" }],
    };
    const suite = fromPromptfoo(pf);
    expect(validateSuite(suite).valid).toBe(true);

    const results = suite.test_cases!.map((tc, i) => {
      const graderId = tc.graders[0] as string;
      const grader = suite.graders!.find((g) => g.id === graderId)!;
      const passed = i === 0;
      return {
        test_case_id: tc.id,
        passed,
        grader_results: [
          { grader_id: graderId, type: grader.type, score: passed ? 1.0 : 0.0, passed },
        ],
      };
    });
    const rs = createResultSet(suite, results as any, "run-e2e");
    const validation = validateResultSet(rs);
    expect(validation.valid, JSON.stringify(validation.errors)).toBe(true);
    expect(rs.summary?.passed).toBe(1);
    expect(rs.summary?.failed).toBe(1);
    expect(rs.summary?.pass_rate).toBe(0.5);
  });
});
