// Unknown fields: strict validation by default, { allowUnknown: true } to opt out.
// Issue #107 / Discussion #108 (PROPOSED -- reference implementation, DO NOT
// MERGE). Unit tests for UNKNOWN_FIELD; the agreement with the raw JSON Schemas
// is checked in schema-consistency-unknown-fields.test.ts.
import { test, expect, describe } from "vitest";
import { validateSuite, validateGrader, validateTestCase, validateResultSet, validateDocument } from "../src/validate";

// ---------------------------------------------------------------------------
// Unknown fields (issue #107 / Discussion #108, PROPOSED -- DO NOT MERGE).
// Strict by default: every object the schemas close with
// additionalProperties: false rejects an undefined key with UNKNOWN_FIELD at
// that key's path. metadata / provider.extra / params / summary.by_grader
// entries stay open. { allowUnknown: true } turns only this check off.
// Mirrors sdk/python/tests/test_unknown_fields.py test-for-test.
// ---------------------------------------------------------------------------

function ufRs(): any {
  return { version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: [{ test_case_id: "tc1", passed: true,
      grader_results: [{ grader_id: "g1", type: "exact_match", score: 1.0, passed: true }] }] };
}
function ufSuite(): any {
  return { version: "1.0.0", id: "s", graders: [{ id: "g1", type: "exact_match" }],
    test_cases: [{ id: "tc1", input: "hi", graders: ["g1"] }] };
}
const unknownOf = (r: { errors: { path: string; code: string }[] }) =>
  r.errors.filter((e) => e.code === "UNKNOWN_FIELD").map((e) => [e.path, e.code]);

const RESULTSET_UNKNOWN_CASES: [string, (d: any) => void, string][] = [
  ["resultset root", (d) => { d.verdict = "pass"; }, "$.verdict"],
  ["provider", (d) => { d.provider = { model: "m", seed: 1 }; }, "$.provider.seed"],
  ["runner", (d) => { d.runner = { name: "x", commit: "abc" }; }, "$.runner.commit"],
  ["summary", (d) => { d.summary = { total: 1, threshold: 0.8 }; }, "$.summary.threshold"],
  ["group", (d) => { d.group = { group_id: "g", gruop_id: "typo" }; }, "$.group.gruop_id"],
  ["result", (d) => { d.results[0].extra_key = 1; }, "$.results[0].extra_key"],
  ["result.error", (d) => { d.results[0].passed = false; d.results[0].error = { type: "timeout", stack: "..." }; }, "$.results[0].error.stack"],
  ["grader_result", (d) => { d.results[0].grader_results[0].threshold = 0.5; }, "$.results[0].grader_results[0].threshold"],
];

describe("unknown fields: resultset", () => {
  for (const [name, mutate, path] of RESULTSET_UNKNOWN_CASES) {
    test(`${name}: rejected by default with UNKNOWN_FIELD at ${path}`, () => {
      const doc = ufRs(); mutate(doc);
      const r = validateResultSet(doc);
      expect(r.valid).toBe(false);
      expect(unknownOf(r)).toEqual([[path, "UNKNOWN_FIELD"]]);
      expect(r.errors.map((e) => e.code)).toEqual(["UNKNOWN_FIELD"]);
    });
    test(`${name}: accepted with allowUnknown`, () => {
      const doc = ufRs(); mutate(doc);
      expect(validateResultSet(doc, { allowUnknown: true }).valid).toBe(true);
    });
  }
});

const SUITE_UNKNOWN_CASES: [string, (d: any) => void, string][] = [
  ["suite root (suite-level threshold, gauntlet#76)", (d) => { d.threshold = 0.8; }, "$.threshold"],
  ["config", (d) => { d.config = { parallel: 2, concurrency: 2 }; }, "$.config.concurrency"],
  ["config.provider", (d) => { d.config = { provider: { model: "m", seed: 1 } }; }, "$.config.provider.seed"],
  ["config.defaults", (d) => { d.config = { defaults: { threshold: 0.5 } }; }, "$.config.defaults.threshold"],
  ["config.retry", (d) => { d.config = { retry: { max_attempts: 2, jitter: true } }; }, "$.config.retry.jitter"],
  // Nested paths keep validateSuite's existing "$.test_cases[i]." + "$.<field>" prefixing.
  ["test case", (d) => { d.test_cases[0].extra_tc_key = 1; }, "$.test_cases[0].$.extra_tc_key"],
  ["test case provider", (d) => { d.test_cases[0].provider = { model: "m", seed: 1 }; }, "$.test_cases[0].$.provider.seed"],
  ["shared grader (config instead of params, llama-cookbook#1072)", (d) => { d.graders[0].config = { x: 1 }; }, "$.graders[0].$.config"],
  ["inline grader in a test case", (d) => { d.test_cases[0].graders = ["g1", { id: "g2", type: "exact_match", threshold: 1 }]; }, "$.test_cases[0].$.graders[1].$.threshold"],
];

describe("unknown fields: suite", () => {
  for (const [name, mutate, path] of SUITE_UNKNOWN_CASES) {
    test(`${name}: rejected by default with UNKNOWN_FIELD at ${path}`, () => {
      const doc = ufSuite(); mutate(doc);
      const r = validateSuite(doc);
      expect(r.valid).toBe(false);
      expect(unknownOf(r)).toEqual([[path, "UNKNOWN_FIELD"]]);
      expect(r.errors.map((e) => e.code)).toEqual(["UNKNOWN_FIELD"]);
    });
    test(`${name}: accepted with allowUnknown`, () => {
      const doc = ufSuite(); mutate(doc);
      expect(validateSuite(doc, { allowUnknown: true }).valid).toBe(true);
    });
  }
});

test("unknown fields: test case and grader rejected standalone", () => {
  let r = validateTestCase({ id: "tc1", input: "hi", graders: ["g1"], extra_tc_key: 1, provider: { seed: 1 } } as any);
  expect(unknownOf(r)).toEqual([["$.extra_tc_key", "UNKNOWN_FIELD"], ["$.provider.seed", "UNKNOWN_FIELD"]]);
  r = validateGrader({ id: "judge", type: "llm_judge", params: { model: "m", prompt: "{output}" }, config: { temperature: 0 } } as any);
  expect(r.errors.map((e) => [e.path, e.code])).toEqual([["$.config", "UNKNOWN_FIELD"]]);
  expect(validateTestCase({ id: "tc1", input: "hi", graders: ["g1"], extra_tc_key: 1 } as any, { allowUnknown: true }).valid).toBe(true);
  expect(validateGrader({ id: "g", type: "exact_match", config: {} } as any, { allowUnknown: true }).valid).toBe(true);
});

test("unknown fields: issue #107 reproduction rejected strict, accepted lenient", () => {
  const rs = ufRs();
  rs.verdict = "pass";
  rs.results[0].extra_result_key = 1;
  rs.results[0].grader_results[0].extra_gr_key = 1;
  expect(unknownOf(validateResultSet(rs))).toEqual([
    ["$.verdict", "UNKNOWN_FIELD"], ["$.results[0].extra_result_key", "UNKNOWN_FIELD"],
    ["$.results[0].grader_results[0].extra_gr_key", "UNKNOWN_FIELD"],
  ]);
  expect(validateResultSet(rs, { allowUnknown: true }).valid).toBe(true);
  const suite = ufSuite();
  suite.threshold = 0.8;
  suite.test_cases[0].extra_tc_key = 1;
  expect(unknownOf(validateSuite(suite))).toEqual([["$.threshold", "UNKNOWN_FIELD"], ["$.test_cases[0].$.extra_tc_key", "UNKNOWN_FIELD"]]);
  expect(validateSuite(suite, { allowUnknown: true }).valid).toBe(true);
});

test("unknown fields: metadata and the other open objects stay open in strict mode", () => {
  const rs = ufRs();
  rs.metadata = { anything: { nested: [1, 2] }, "com.example.x": true };
  rs.provider = { model: "m", extra: { seed: 1, top_p: 0.9 } };
  rs.summary = { total: 1, by_grader: { g1: { passed: 1, failed: 0, p95_ms: 12 } } };
  rs.results[0].metadata = { trace_id: "abc", verdict: "pass" };
  rs.results[0].grader_results[0].metadata = { threshold: 0.5, openeval: { raw_score: 4 } };
  const r1 = validateResultSet(rs);
  expect(r1.valid, JSON.stringify(r1.errors)).toBe(true);
  const suite = ufSuite();
  suite.metadata = { threshold: 0.8 };
  suite.config = { provider: { model: "m", extra: { seed: 1 } } };
  suite.graders[0].params = { ignore_case: true, custom_knob: 1 };
  Object.assign(suite.test_cases[0], { metadata: { k: 1 }, params: { top_p: 1 }, provider: { extra: { a: 1 } } });
  const r2 = validateSuite(suite);
  expect(r2.valid, JSON.stringify(r2.errors)).toBe(true);
});

test("unknown fields: allowUnknown only disables the UNKNOWN_FIELD check", () => {
  const rs = ufRs();
  rs.verdict = "pass";
  rs.results[0].actual_output = ["a", "b"]; // still a TYPE_ERROR
  expect(validateResultSet(rs, { allowUnknown: true }).errors.map((e) => [e.path, e.code])).toEqual([["$.results[0].actual_output", "TYPE_ERROR"]]);
  expect(validateResultSet(rs).errors.map((e) => e.code).sort()).toEqual(["TYPE_ERROR", "UNKNOWN_FIELD"]);
});

test("unknown fields: reported once per key and does not mask other errors", () => {
  const rs = ufRs();
  Object.assign(rs.results[0], { a: 1, b: 2 });
  delete rs.results[0].passed;
  const r = validateResultSet(rs);
  expect(r.errors.map((e) => [e.path, e.code])).toContainEqual(["$.results[0].passed", "REQUIRED"]);
  expect(unknownOf(r)).toEqual([["$.results[0].a", "UNKNOWN_FIELD"], ["$.results[0].b", "UNKNOWN_FIELD"]]);
});

test("unknown fields: validateDocument passes allowUnknown through", () => {
  const rs = ufRs(); rs.verdict = "pass";
  const suite = ufSuite(); suite.threshold = 1;
  const cases: [unknown, "resultset" | "suite" | "testcase" | "grader"][] = [
    [rs, "resultset"], [suite, "suite"],
    [{ id: "tc1", input: "hi", graders: ["g1"], x: 1 }, "testcase"],
    [{ id: "g1", type: "exact_match", x: 1 }, "grader"],
  ];
  for (const [doc, t] of cases) {
    expect(validateDocument(doc, t).valid).toBe(false);
    expect(validateDocument(doc, t, { allowUnknown: true }).valid).toBe(true);
  }
});

test("unknown fields: default is strict; a key whose value is undefined is not a JSON property", () => {
  const rs = ufRs(); rs.verdict = "pass";
  expect(validateResultSet(rs).valid).toBe(false);
  expect(validateResultSet(rs, {}).valid).toBe(false);
  expect(validateResultSet(rs, { allowUnknown: false }).valid).toBe(false);
  const rs2 = ufRs(); rs2.verdict = undefined; // JSON.stringify drops it
  expect(validateResultSet(rs2).valid).toBe(true);
});
