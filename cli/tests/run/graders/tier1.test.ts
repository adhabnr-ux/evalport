import { describe, expect, test } from "vitest";
import { TIER1_TYPES, gradeContains, gradeExactMatch, gradeJsonPath, gradeJsonSchema, gradeRegex, gradeTier1 } from "../../../src/run/graders/tier1";
import type { Grader } from "../../../../sdk/typescript/src/types";

function grader(type: Grader["type"], params: Record<string, unknown> = {}, id = "g1"): Grader {
  return { id, type, params };
}

describe("gradeExactMatch", () => {
  test("matches exactly, trims by default", () => {
    const r = gradeExactMatch(grader("exact_match"), "  4  ", "4");
    expect(r.passed).toBe(true);
    expect(r.score).toBe(1);
  });

  test("case-sensitive by default", () => {
    const r = gradeExactMatch(grader("exact_match"), "Paris", "paris");
    expect(r.passed).toBe(false);
  });

  test("ignore_case:true matches case-insensitively", () => {
    const r = gradeExactMatch(grader("exact_match", { ignore_case: true }), "Paris", "paris");
    expect(r.passed).toBe(true);
  });

  test("trim_whitespace:false preserves whitespace sensitivity", () => {
    const r = gradeExactMatch(grader("exact_match", { trim_whitespace: false }), "4 ", "4");
    expect(r.passed).toBe(false);
  });

  test("handles missing expected_output as empty string", () => {
    const r = gradeExactMatch(grader("exact_match"), "", undefined);
    expect(r.passed).toBe(true);
  });
});

describe("gradeContains", () => {
  test("substring found / not found", () => {
    expect(gradeContains(grader("contains", { substring: "cat" }), "the cat sat").passed).toBe(true);
    expect(gradeContains(grader("contains", { substring: "dog" }), "the cat sat").passed).toBe(false);
  });

  test("ignore_case option", () => {
    expect(gradeContains(grader("contains", { substring: "CAT", ignore_case: true }), "the cat sat").passed).toBe(true);
    expect(gradeContains(grader("contains", { substring: "CAT" }), "the cat sat").passed).toBe(false);
  });
});

describe("gradeRegex", () => {
  test("matches a pattern", () => {
    expect(gradeRegex(grader("regex", { pattern: "^\\d+$" }), "12345").passed).toBe(true);
    expect(gradeRegex(grader("regex", { pattern: "^\\d+$" }), "12a45").passed).toBe(false);
  });

  test("supports flags", () => {
    expect(gradeRegex(grader("regex", { pattern: "hello", flags: "i" }), "HELLO world").passed).toBe(true);
  });

  test("invalid regex fails gracefully instead of throwing", () => {
    const r = gradeRegex(grader("regex", { pattern: "(unclosed" }), "anything");
    expect(r.passed).toBe(false);
    expect(r.reason).toMatch(/invalid regex/);
  });
});

describe("gradeJsonSchema", () => {
  test("valid JSON matching schema passes", () => {
    const r = gradeJsonSchema(grader("json_schema", { schema: { type: "object", required: ["ok"], properties: { ok: { type: "boolean" } } } }), '{"ok": true}');
    expect(r.passed).toBe(true);
  });

  test("invalid JSON fails with a clear reason", () => {
    const r = gradeJsonSchema(grader("json_schema", { schema: {} }), "{not json");
    expect(r.passed).toBe(false);
    expect(r.reason).toMatch(/not valid JSON/);
  });

  test("valid JSON that violates the schema fails with schema_errors metadata", () => {
    const r = gradeJsonSchema(grader("json_schema", { schema: { type: "object", required: ["ok"] } }), "{}");
    expect(r.passed).toBe(false);
    expect(r.metadata?.schema_errors).toBeDefined();
  });
});

describe("gradeJsonPath", () => {
  test("eq operator (default)", () => {
    const r = gradeJsonPath(grader("json_path", { path: "$.answer", expected: 42 }), '{"answer": 42}');
    expect(r.passed).toBe(true);
  });

  test("numeric operators: gt, lt, gte, lte, ne", () => {
    const body = '{"n": 5}';
    expect(gradeJsonPath(grader("json_path", { path: "$.n", expected: 3, operator: "gt" }), body).passed).toBe(true);
    expect(gradeJsonPath(grader("json_path", { path: "$.n", expected: 10, operator: "lt" }), body).passed).toBe(true);
    expect(gradeJsonPath(grader("json_path", { path: "$.n", expected: 5, operator: "gte" }), body).passed).toBe(true);
    expect(gradeJsonPath(grader("json_path", { path: "$.n", expected: 5, operator: "lte" }), body).passed).toBe(true);
    expect(gradeJsonPath(grader("json_path", { path: "$.n", expected: 6, operator: "ne" }), body).passed).toBe(true);
  });

  test("contains operator on stringified value", () => {
    const r = gradeJsonPath(grader("json_path", { path: "$.msg", expected: "wor", operator: "contains" }), '{"msg": "hello world"}');
    expect(r.passed).toBe(true);
  });

  test("loose equality allows numeric/string coercion", () => {
    const r = gradeJsonPath(grader("json_path", { path: "$.n", expected: "5" }), '{"n": 5}');
    expect(r.passed).toBe(true);
  });

  test("path matching nothing fails cleanly", () => {
    const r = gradeJsonPath(grader("json_path", { path: "$.missing", expected: 1 }), "{}");
    expect(r.passed).toBe(false);
    expect(r.reason).toMatch(/matched nothing/);
  });

  test("malformed JSON output fails cleanly", () => {
    const r = gradeJsonPath(grader("json_path", { path: "$.x", expected: 1 }), "not json");
    expect(r.passed).toBe(false);
    expect(r.reason).toMatch(/not valid JSON/);
  });

  test("invalid JSONPath syntax fails cleanly, not by throwing", () => {
    const r = gradeJsonPath(grader("json_path", { path: "no-dollar", expected: 1 }), "{}");
    expect(r.passed).toBe(false);
    expect(r.reason).toMatch(/invalid JSONPath/);
  });
});

describe("TIER1_TYPES / gradeTier1 dispatcher", () => {
  test("contains exactly the five local-only grader types", () => {
    expect([...TIER1_TYPES].sort()).toEqual(["contains", "exact_match", "json_path", "json_schema", "regex"]);
  });

  test("dispatches to the right implementation for each type", () => {
    expect(gradeTier1(grader("exact_match"), "a", "a").passed).toBe(true);
    expect(gradeTier1(grader("contains", { substring: "a" }), "abc", undefined).passed).toBe(true);
    expect(gradeTier1(grader("regex", { pattern: "a" }), "abc", undefined).passed).toBe(true);
    expect(gradeTier1(grader("json_schema", { schema: {} }), "{}", undefined).passed).toBe(true);
    expect(gradeTier1(grader("json_path", { path: "$.a", expected: 1 }), '{"a":1}', undefined).passed).toBe(true);
  });

  test("throws for a non-tier-1 type — callers must route via TIER1_TYPES first", () => {
    expect(() => gradeTier1(grader("llm_judge"), "x", undefined)).toThrow(/non-tier-1/);
  });
});
