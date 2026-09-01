import { test, expect, describe } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import Ajv2020 from "ajv/dist/2020";
import { validateGrader, validateSuite, validateResultSet, SEMVER_RE } from "../src/validate";

// Cross-validates the raw JSON Schema files (spec/schemas/*.json -- the source of
// truth that any JSON-Schema-based tool, not just this SDK, would validate against)
// against this SDK's hand-rolled TypeScript validator.
//
// These two validation paths are maintained independently (the hand-rolled
// validator exists for zero-dependency, fast, structured-error validation; the
// JSON Schema files exist as the portable, tool-agnostic spec artifact). History
// has already shown they drift: this SDK once accepted "1.0.0-rc.1" while the
// JSON Schema's `version` pattern silently rejected it, and the JSON Schema's
// grader `allOf` blocks declared per-type `params` requirements that were never
// actually enforced (a `then` block that says "if params is present, it must
// have `substring`" says nothing about whether `params` itself must be present).
//
// This test suite is the regression guard against that class of drift: every
// case here is checked against BOTH validation paths and must agree.

const SCHEMA_DIR = join(__dirname, "..", "..", "..", "spec", "schemas");

function loadSchema(name: string): any {
  return JSON.parse(readFileSync(join(SCHEMA_DIR, `${name}.json`), "utf8"));
}

const testcaseSchema = loadSchema("testcase");
const graderSchema = loadSchema("grader");
const suiteSchema = loadSchema("suite");
const resultsetSchema = loadSchema("resultset");

const ajv = new Ajv2020({ allErrors: true, strict: false });
// suite.json and testcase.json both $ref grader.json/testcase.json by their $id
// URL, so all four schemas must be registered together for $ref resolution to
// work fully offline (no network fetch of https://evalport.org/schema/*.json).
ajv.addSchema(testcaseSchema, testcaseSchema.$id);
ajv.addSchema(graderSchema, graderSchema.$id);
ajv.addSchema(suiteSchema, suiteSchema.$id);
ajv.addSchema(resultsetSchema, resultsetSchema.$id);

const graderValidate = ajv.getSchema(graderSchema.$id)!;
const suiteValidate = ajv.getSchema(suiteSchema.$id)!;
const resultsetValidate = ajv.getSchema(resultsetSchema.$id)!;

describe("schema files are well-formed Draft 2020-12 schemas", () => {
  for (const [name, schema] of [
    ["testcase", testcaseSchema],
    ["grader", graderSchema],
    ["suite", suiteSchema],
    ["resultset", resultsetSchema],
  ] as const) {
    test(`${name}.json compiles as a valid schema`, () => {
      expect(() => ajv.compile(schema)).not.toThrow();
    });
  }
});

describe("grader: JSON Schema and hand-rolled validator agree", () => {
  const cases: [string, unknown, boolean][] = [
    ["well-known type, valid params", { id: "g1", type: "exact_match" }, true],
    ["well-known type (contains), missing required param", { id: "g2", type: "contains", params: {} }, false],
    ["well-known type (contains), no params object at all", { id: "g3", type: "contains" }, false],
    ["custom, no params at all", { id: "g4", type: "custom" }, false],
    ["custom, with handler", { id: "g5", type: "custom", params: { handler: "my.module:fn" } }, true],
    ["non-standard type, no params at all", { id: "g6", type: "trulens_feedback" }, false],
    ["non-standard type, empty params (no handler)", { id: "g7", type: "trulens_feedback", params: {} }, false],
    [
      "non-standard type, with handler",
      { id: "g8", type: "trulens_feedback", params: { handler: "trulens.feedback:run" } },
      true,
    ],
    ["empty-string type", { id: "g9", type: "" }, false],
  ];

  for (const [name, doc, expected] of cases) {
    test(name, () => {
      const jsOk = graderValidate(doc) as boolean;
      const handOk = validateGrader(doc).valid;
      expect(jsOk, `JSON Schema acceptance for "${name}"`).toBe(expected);
      expect(handOk, `hand-rolled validator acceptance for "${name}"`).toBe(expected);
    });
  }
});

describe("suite/resultset: semver 2.0.0 version pattern agrees with SEMVER_RE", () => {
  const versions: [string, string, boolean][] = [
    ["plain release", "1.0.0", true],
    ["legacy -draft suffix", "1.0.0-draft", true],
    ["numeric prerelease", "1.0.0-rc.1", true],
    ["alpha prerelease", "1.1.0-beta.2", true],
    ["build metadata", "1.0.0+build.5", true],
    ["prerelease + build metadata", "1.0.0-rc.1+build.5", true],
    ["garbage string", "garbage", false],
    ["missing patch component", "1.0", false],
    ["trailing dash, no prerelease identifier", "1.0.0-", false],
  ];

  const suitePattern = new RegExp(suiteSchema.properties.version.pattern);
  const resultsetPattern = new RegExp(resultsetSchema.properties.version.pattern);

  for (const [name, version, expected] of versions) {
    test(`suite.json pattern: ${name}`, () => {
      expect(suitePattern.test(version)).toBe(expected);
      expect(SEMVER_RE.test(version)).toBe(expected);
    });
    test(`resultset.json pattern: ${name}`, () => {
      expect(resultsetPattern.test(version)).toBe(expected);
      expect(SEMVER_RE.test(version)).toBe(expected);
    });
  }

  function minimalSuite(version: string) {
    return {
      version,
      id: "s1",
      graders: [{ id: "g1", type: "exact_match" }],
      test_cases: [{ id: "tc1", input: "hi", graders: ["g1"] }],
    };
  }

  function minimalResultSet(version: string) {
    return {
      version,
      suite_id: "s1",
      run_id: "run1",
      started_at: "2026-08-16T00:00:00Z",
      results: [
        {
          test_case_id: "tc1",
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 0.9, passed: true }],
          passed: true,
        },
      ],
    };
  }

  for (const [name, version, expected] of versions) {
    test(`suite end-to-end: ${name}`, () => {
      const doc = minimalSuite(version);
      expect(suiteValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(validateSuite(doc).valid, "hand-rolled").toBe(expected);
    });
    test(`resultset end-to-end: ${name}`, () => {
      const doc = minimalResultSet(version);
      expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(validateResultSet(doc).valid, "hand-rolled").toBe(expected);
    });
  }
});

describe("resultset: [0,1] score range enforcement agrees", () => {
  const cases: [string, number | null | boolean, boolean][] = [
    ["in-range score", 0.5, true],
    ["lower bound", 0.0, true],
    ["upper bound", 1.0, true],
    ["null (skipped/pending grader)", null, true],
    ["above range", 1.5, false],
    ["below range", -0.1, false],
    ["boolean true is not a valid score", true, false],
    ["boolean false is not a valid score", false, false],
  ];

  for (const [name, score, expected] of cases) {
    test(name, () => {
      const doc = {
        version: "1.0.0",
        suite_id: "s1",
        run_id: "run1",
        started_at: "2026-08-16T00:00:00Z",
        results: [
          {
            test_case_id: "tc1",
            grader_results: [{ grader_id: "g1", type: "human", score, passed: false }],
            passed: false,
          },
        ],
      };
      const jsOk = resultsetValidate(doc) as boolean;
      const handOk = validateResultSet(doc).valid;
      expect(jsOk, "JSON Schema").toBe(expected);
      expect(handOk, "hand-rolled").toBe(expected);
    });
  }
});

// Mirrors sdk/python/tests/test_schema_consistency.py's per-result `completed_at`
// tests (added for resumable/partial runs, Discussion #10). The hand-rolled
// validator never enforced additionalProperties, so it already silently accepted
// an unknown `completed_at` key -- but the JSON Schema's `additionalProperties:
// false` on the result item would have REJECTED it until the schema declared the
// field. Before that schema change, this exact fixture would have failed jsOk
// while still passing handOk.
describe("resultset: per-result completed_at (Discussion #10) agrees", () => {
  test("present validates in both paths", () => {
    const doc = {
      version: "1.0.0",
      suite_id: "s1",
      run_id: "run1",
      started_at: "2026-08-16T00:00:00Z",
      results: [
        {
          test_case_id: "tc1",
          completed_at: "2026-08-16T00:00:05Z",
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 1.0, passed: true }],
          passed: true,
        },
      ],
    };
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
  });

  test("absent still validates in both paths (optional field)", () => {
    const doc = {
      version: "1.0.0",
      suite_id: "s1",
      run_id: "run1",
      started_at: "2026-08-16T00:00:00Z",
      results: [
        {
          test_case_id: "tc1",
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 0.9, passed: true }],
          passed: true,
        },
      ],
    };
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
  });

  test("multi-attempt ResultSet with ResultSet-level isolation validates in both paths", () => {
    // Discussion #22 / issue #20: multiple Results per test_case_id,
    // distinguished by ascending attempt, plus a single ResultSet-level
    // isolation. Mirrors spec/conformance/fixtures/multi_attempt_resultset_valid.json.
    const doc = {
      version: "1.0.0",
      suite_id: "s1",
      run_id: "run1",
      started_at: "2026-08-16T00:00:00Z",
      isolation: "fresh",
      results: [
        {
          test_case_id: "tc1",
          attempt: 1,
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 1.0, passed: true }],
          passed: true,
        },
        {
          test_case_id: "tc1",
          attempt: 2,
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 0.0, passed: false }],
          passed: false,
        },
      ],
    };
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
  });

  test("duplicate (test_case_id, run_id, attempt) rejected by the hand-rolled validator", () => {
    // Cross-item uniqueness like (test_case_id, run_id, attempt) can't be
    // expressed by JSON Schema's per-item `minimum: 1` on attempt -- it's a
    // hand-rolled-validator-only rule, by design, the same way DUPLICATE_ID
    // for suite test case ids is. So this is checked only against the
    // hand-rolled path, not asserted to also fail the raw JSON Schema.
    const doc = {
      version: "1.0.0",
      suite_id: "s1",
      run_id: "run1",
      started_at: "2026-08-16T00:00:00Z",
      results: [
        {
          test_case_id: "tc1",
          attempt: 1,
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 1.0, passed: true }],
          passed: true,
        },
        {
          test_case_id: "tc1",
          attempt: 1,
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 0.0, passed: false }],
          passed: false,
        },
      ],
    };
    const r = validateResultSet(doc);
    expect(r.valid).toBe(false);
    expect(r.errors.some((e) => e.code === "DUPLICATE_ATTEMPT")).toBe(true);
  });

  test("attempt/isolation absent still validates in both paths (backward compatibility)", () => {
    const doc = {
      version: "1.0.0",
      suite_id: "s1",
      run_id: "run1",
      started_at: "2026-08-16T00:00:00Z",
      results: [
        {
          test_case_id: "tc1",
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 0.9, passed: true }],
          passed: true,
        },
      ],
    };
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
  });

  test("metadata.openeval.partial marker needs no schema change and validates in both paths", () => {
    const doc = {
      version: "1.0.0",
      suite_id: "s1",
      run_id: "run1",
      started_at: "2026-08-16T00:00:00Z",
      metadata: { openeval: { partial: true } },
      results: [
        {
          test_case_id: "tc1",
          grader_results: [{ grader_id: "g1", type: "exact_match", score: 0.9, passed: true }],
          passed: true,
        },
      ],
    };
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
  });
});
