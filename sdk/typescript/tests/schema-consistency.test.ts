import { test, expect, describe } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import Ajv2020 from "ajv/dist/2020";
import addFormats from "ajv-formats";
import {
  validateGrader,
  validateSuite,
  validateResultSet,
  validateTestCase,
  SEMVER_RE,
  isRfc3339DateTime,
} from "../src/validate";

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
// resultset.json declares `format: "date-time"` on started_at/completed_at. Ajv
// ignores unknown formats, so without ajv-formats that keyword is a no-op and the
// date-time agreement cases below would be vacuous on the JSON Schema side.
addFormats(ajv);
// suite.json and testcase.json both $ref grader.json/testcase.json by their $id
// URL, so all four schemas must be registered together for $ref resolution to
// work fully offline (no network fetch of https://evalport.org/schema/*.json).
ajv.addSchema(testcaseSchema, testcaseSchema.$id);
ajv.addSchema(graderSchema, graderSchema.$id);
ajv.addSchema(suiteSchema, suiteSchema.$id);
ajv.addSchema(resultsetSchema, resultsetSchema.$id);

const testcaseValidate = ajv.getSchema(testcaseSchema.$id)!;
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

test("testcase empty-string input is rejected by both paths", () => {
  const doc = { id: "tc1", input: "", graders: ["g1"] };

  expect(testcaseValidate(doc) as boolean, "JSON Schema").toBe(false);

  const result = validateTestCase(doc);
  expect(result.valid, "hand-rolled").toBe(false);
  expect(
    result.errors.some(
      e => e.path === "$.input" && e.code === "MIN_LENGTH"
    )
  ).toBe(true);
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

// ---------------------------------------------------------------------------
// ResultSet: `group` (Discussion #45, proposed -- grouped/sibling ResultSets).
// additionalProperties: false on the ResultSet object means the raw JSON Schema
// would have rejected a `group` key before this schema change landed here,
// exactly the same class of drift the per-result completed_at tests above
// document -- both sides (schema + hand-rolled validator) must be updated
// together, which is what this section checks. Mirrors
// sdk/python/tests/test_schema_consistency.py's group section rule-for-rule.
// ---------------------------------------------------------------------------

describe("resultset: group (Discussion #45, proposed) agrees", () => {
  function minimalResultSetDoc(overrides: Record<string, unknown> = {}) {
    return {
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
      ...overrides,
    };
  }

  test("with group_id only valid in both paths", () => {
    const doc = minimalResultSetDoc({ group: { group_id: "mutation-sweep-2026-09-01" } });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
  });

  test("with all fields valid in both paths", () => {
    const doc = minimalResultSetDoc({
      group: {
        group_id: "mutation-sweep-2026-09-01",
        role: "mutant",
        label: "mutant_017 (relational-operator-swap in billing.py:42)",
        sequence: 17,
      },
    });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
  });

  test("absent still valid in both paths (backward compatibility)", () => {
    const doc = minimalResultSetDoc();
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
    expect("group" in doc).toBe(false);
  });

  test("missing group_id rejected by both paths", () => {
    const doc = minimalResultSetDoc({ group: { role: "mutant" } }); // group_id is REQUIRED when group is present
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(false);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(false);
  });

  test("unknown subfield rejected by JSON Schema", () => {
    // additionalProperties: false on the group object itself -- a typo'd
    // sub-field (e.g. "gruop_id") must be caught structurally by the JSON
    // Schema even though the hand-rolled validator (like every other optional
    // object in this file) doesn't police unknown keys.
    const doc = minimalResultSetDoc({ group: { group_id: "g1", not_a_real_field: "oops" } });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(false);
  });

  test("sequence must be a non-negative integer in both paths", () => {
    const doc = minimalResultSetDoc({ group: { group_id: "g1", sequence: -1 } });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(false);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(false);
  });

  test("wrong type (string instead of object) rejected by both paths", () => {
    const doc = minimalResultSetDoc({ group: "mutation-sweep-2026-09-01" }); // must be an object, not a string
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(false);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(false);
  });

  // -------------------------------------------------------------------------
  // group.parent_group_id (nested/hierarchical groups -- sweep-of-sweeps).
  // Same drift class as the rest of this file: additionalProperties: false on
  // the group object means the raw JSON Schema would reject parent_group_id
  // entirely until the schema itself was updated alongside the hand-rolled
  // validator. Mirrors sdk/python/tests/test_schema_consistency.py's
  // parent_group_id section rule-for-rule.
  // -------------------------------------------------------------------------

  test("parent_group_id valid nested sweep agrees in both paths", () => {
    const doc = minimalResultSetDoc({
      group: {
        group_id: "child-sweep-lr-1e-4",
        parent_group_id: "parent-sweep-lr-batchsize-grid-2026-09-08",
        role: "candidate",
        sequence: 3,
      },
    });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
  });

  test("parent_group_id absent still valid in both paths (backward compatibility)", () => {
    const doc = minimalResultSetDoc({ group: { group_id: "mutation-sweep-2026-09-01" } });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(true);
    expect("parent_group_id" in (doc.group as Record<string, unknown>)).toBe(false);
  });

  test("parent_group_id empty string rejected by both paths", () => {
    const doc = minimalResultSetDoc({ group: { group_id: "g1", parent_group_id: "" } });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(false);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(false);
  });

  test("parent_group_id wrong type rejected by both paths", () => {
    const doc = minimalResultSetDoc({ group: { group_id: "g1", parent_group_id: 42 } });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(false);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(false);
  });

  test("parent_group_id equal to group_id: JSON Schema allows it structurally, hand-rolled validator rejects it", () => {
    // The self-parent rule (a group cannot be its own parent) is a
    // cross-field constraint JSON Schema's `properties`/`required` vocabulary
    // cannot express without a $data reference (not part of this project's
    // supported draft usage elsewhere in the schema) -- so, same as
    // uniqueness rules like DUPLICATE_ATTEMPT elsewhere in this suite, this
    // is intentionally enforced only by the hand-rolled validator. Documented
    // here (rather than silently skipped) so a future schema change that
    // *does* add a $data-based check is a deliberate decision, not a
    // rediscovery.
    const doc = minimalResultSetDoc({ group: { group_id: "sweep-42", parent_group_id: "sweep-42" } });
    expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(true);
    expect(validateResultSet(doc).valid, "hand-rolled").toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Validator fidelity: optional-field types, RFC 3339 date-times, and the Rule 6
// null-score/passed rules. Before this section the hand-rolled validator
// accepted every "reject" document below that the raw JSON Schema rejects --
// e.g. Result.actual_output as a list (found by ChelseaKR in
// ChelseaKR/gauntlet#76), a date-only started_at, or a string temperature.
// Unknown-field (additionalProperties) handling is deliberately NOT covered
// here -- that is a separate spec question. Mirrors
// sdk/python/tests/test_schema_consistency.py case-for-case.
// ---------------------------------------------------------------------------

function fidelityRs(): any {
  return {
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
}

function fidelitySuite(): any {
  return {
    version: "1.0.0",
    id: "s1",
    graders: [{ id: "g1", type: "exact_match" }],
    test_cases: [{ id: "tc1", input: "hi", graders: ["g1"] }],
  };
}

test("date-time format is actually asserted by the JSON Schema path", () => {
  const doc = fidelityRs();
  doc.started_at = "2026-08-16";
  expect(resultsetValidate(doc) as boolean).toBe(false);
});

const r0 = (d: any) => d.results[0];
const gr0 = (d: any) => d.results[0].grader_results[0];

const RESULTSET_FIDELITY_CASES: [string, (d: any) => void, boolean][] = [
  ["suite_version number", (d) => { d.suite_version = 1; }, false],
  ["completed_at date-only", (d) => { d.completed_at = "2026-08-16"; }, false],
  ["completed_at number", (d) => { d.completed_at = 1700000000; }, false],
  ["provider string", (d) => { d.provider = "gpt-4o"; }, false],
  ["provider null", (d) => { d.provider = null; }, false],
  ["provider.model number", (d) => { d.provider = { model: 4 }; }, false],
  ["provider.api_base list", (d) => { d.provider = { api_base: ["x"] }; }, false],
  ["provider.temperature string", (d) => { d.provider = { temperature: "0.1" }; }, false],
  ["provider.temperature bool", (d) => { d.provider = { temperature: true }; }, false],
  ["provider.max_tokens fractional", (d) => { d.provider = { max_tokens: 1.5 }; }, false],
  ["provider.max_tokens bool", (d) => { d.provider = { max_tokens: true }; }, false],
  ["provider.extra list", (d) => { d.provider = { extra: [] }; }, false],
  ["runner list", (d) => { d.runner = []; }, false],
  ["runner.name number", (d) => { d.runner = { name: 1 }; }, false],
  ["runner.version number", (d) => { d.runner = { version: 1.0 }; }, false],
  ["summary list", (d) => { d.summary = []; }, false],
  ["summary.total negative", (d) => { d.summary = { total: -1 }; }, false],
  ["summary.total bool", (d) => { d.summary = { total: true }; }, false],
  ["summary.passed fractional", (d) => { d.summary = { passed: 0.5 }; }, false],
  ["summary.pass_rate above 1", (d) => { d.summary = { pass_rate: 1.5 }; }, false],
  ["summary.avg_score string", (d) => { d.summary = { avg_score: "0.5" }; }, false],
  ["summary.duration_ms negative", (d) => { d.summary = { duration_ms: -5 }; }, false],
  ["summary.by_grader list", (d) => { d.summary = { by_grader: [] }; }, false],
  ["summary.by_grader entry number", (d) => { d.summary = { by_grader: { g1: 3 } }; }, false],
  ["summary.by_grader.passed fractional", (d) => { d.summary = { by_grader: { g1: { passed: 1.5 } } }; }, false],
  ["summary.by_grader.avg_score string", (d) => { d.summary = { by_grader: { g1: { avg_score: "x" } } }; }, false],
  ["metadata list", (d) => { d.metadata = []; }, false],
  ["isolation null", (d) => { d.isolation = null; }, false],
  ["group null", (d) => { d.group = null; }, false],
  ["actual_output list (ChelseaKR/gauntlet#76)", (d) => { r0(d).actual_output = ["a", "b"]; }, false],
  ["actual_output null", (d) => { r0(d).actual_output = null; }, false],
  ["test_case_id empty", (d) => { r0(d).test_case_id = ""; }, false],
  ["duration_ms fractional", (d) => { r0(d).duration_ms = 1.5; }, false],
  ["duration_ms negative", (d) => { r0(d).duration_ms = -1; }, false],
  ["duration_ms bool", (d) => { r0(d).duration_ms = true; }, false],
  ["result completed_at offset-less", (d) => { r0(d).completed_at = "2026-08-16T00:00:05"; }, false],
  ["error string", (d) => { r0(d).error = "boom"; }, false],
  ["error.type not in enum", (d) => { r0(d).error = { type: "crash" }; }, false],
  ["error.message number", (d) => { r0(d).error = { message: 1 }; }, false],
  ["error.code fractional", (d) => { r0(d).error = { code: 1.5 }; }, false],
  ["error.code bool", (d) => { r0(d).error = { code: true }; }, false],
  ["error.retryable string", (d) => { r0(d).error = { retryable: "yes" }; }, false],
  ["result metadata string", (d) => { r0(d).metadata = "trace"; }, false],
  ["attempt null", (d) => { r0(d).attempt = null; }, false],
  ["grader_id empty", (d) => { gr0(d).grader_id = ""; }, false],
  ["score missing", (d) => { delete gr0(d).score; gr0(d).passed = false; r0(d).passed = false; }, false],
  ["reason number", (d) => { gr0(d).reason = 1; }, false],
  ["grader metadata list", (d) => { gr0(d).metadata = []; }, false],
  // Valid documents must stay valid in both paths.
  ["every optional field well-typed", (d) => {
    Object.assign(d, {
      $schema: "https://evalport.org/schema/resultset.json",
      suite_version: "1.0.0",
      completed_at: "2026-08-16T00:01:45.123+05:30",
      provider: { model: "gpt-4o", api_base: "https://api.example.com", temperature: 0, max_tokens: 256, extra: { seed: 1 } },
      runner: { name: "evalport-cli", version: "1.3.1" },
      summary: { total: 1, passed: 1, failed: 0, skipped: 0, pass_rate: 1, avg_score: 0.9, duration_ms: 1200,
        by_grader: { g1: { passed: 1, failed: 0, avg_score: 0.9 } } },
      metadata: { "openeval.partial": false },
    });
    Object.assign(r0(d), {
      actual_output: "Paris", duration_ms: 1200, completed_at: "2026-08-16t00:01:44z",
      error: { type: "provider_error", message: "rate limited", code: 429, retryable: true },
      metadata: { trace_id: "abc" },
    });
    Object.assign(gr0(d), { reason: "close enough", metadata: { k: "v" } });
  }, true],
  ["error.code string", (d) => { r0(d).error = { type: "timeout", code: "ETIMEDOUT" }; }, true],
  ["integral float duration_ms", (d) => { r0(d).duration_ms = 1200.0; }, true],
  ["negative temperature allowed (no schema minimum)", (d) => { d.provider = { temperature: -1 }; }, true],
  // PROPOSED (Discussion #49, alt B): a partly-scored row must declare its
  // aggregation (PARTIAL_RESULT_UNDECLARED); both validators enforce it.
  ["mixed null and scored graders, passed, undeclared", (d) => {
    r0(d).grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
  }, false],
  ["mixed null and scored graders, passed, declared on the Result (dotted)", (d) => {
    r0(d).grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
    r0(d).metadata = { "openeval.aggregation": { strategy: "all" } };
  }, true],
  ["mixed null and scored graders, passed, declared on the ResultSet (nested)", (d) => {
    r0(d).grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
    d.metadata = { openeval: { aggregation: { strategy: "strict" } } };
  }, true],
  ["mixed null and scored graders, passed false, undeclared", (d) => {
    r0(d).passed = false;
    r0(d).grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
  }, false],
  ["declared aggregation with unknown strategy", (d) => {
    r0(d).grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
    d.metadata = { openeval: { aggregation: { strategy: "fail-closed" } } };
  }, false],
  ["declared weighted aggregation without threshold", (d) => {
    r0(d).grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
    r0(d).metadata = { "openeval.aggregation": { strategy: "weighted" } };
  }, false],
  ["declared weighted aggregation with threshold", (d) => {
    r0(d).grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
    r0(d).metadata = { "openeval.aggregation": { strategy: "weighted", threshold: 0.5 } };
  }, true],
  ["declared aggregation threshold out of range", (d) => {
    d.metadata = { openeval: { aggregation: { strategy: "all", threshold: 2 } } };
  }, false],
  ["declared aggregation not an object", (d) => {
    d.metadata = { "openeval.aggregation": "all" };
  }, false],
  ["aggregation_status partial on a mixed, declared row", (d) => {
    r0(d).grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
    d.metadata = { openeval: { aggregation: { strategy: "all" } } };
    r0(d).metadata = { openeval: { aggregation_status: "partial" } };
  }, true],
  ["aggregation_status partial on a fully scored row", (d) => {
    r0(d).metadata = { "openeval.aggregation_status": "partial" };
  }, false],
  ["aggregation_status unscored on a scored row", (d) => {
    r0(d).metadata = { openeval: { aggregation_status: "unscored" } };
  }, false],
  ["aggregation_status unknown value", (d) => {
    r0(d).metadata = { openeval: { aggregation_status: "pending" } };
  }, false],
  ["aggregation_status unscored on an all-null row", (d) => {
    r0(d).passed = false;
    r0(d).grader_results = [{ grader_id: "g1", type: "human", score: null, passed: false }];
    r0(d).metadata = { openeval: { aggregation_status: "unscored" } };
  }, true],
  ["all null-scored, passed false", (d) => {
    r0(d).passed = false;
    r0(d).grader_results = [{ grader_id: "g1", type: "human", score: null, passed: false }];
  }, true],
];

describe("resultset fidelity: JSON Schema and hand-rolled validator agree", () => {
  for (const [name, mutate, expected] of RESULTSET_FIDELITY_CASES) {
    test(name, () => {
      const doc = fidelityRs();
      mutate(doc);
      const jsOk = resultsetValidate(doc) as boolean;
      const hand = validateResultSet(doc);
      expect(jsOk, "JSON Schema").toBe(expected);
      expect(hand.valid, `hand-rolled: ${JSON.stringify(hand.errors)}`).toBe(expected);
    });
  }
});

// RFC 3339 date-time values on which the hand-rolled check, ajv-formats and
// jsonschema's FORMAT_CHECKER (the Python twin of this test) all agree.
// Deliberately excluded because the two JSON Schema format implementations
// disagree with RFC 3339 / each other, not with this SDK:
//   "2016-12-31T23:59:60Z" (leap second; RFC 3339 grammar allows it, rfc3339-validator doesn't)
//   "2026-01-15T10:30:00Z\n" (rfc3339-validator's `$` matches before a trailing newline)
//   "2026-01-15 10:30:00Z", "...+0530", "...+05" (ajv-formats accepts; RFC 3339 does not)
const DATE_TIME_CASES: [string, boolean][] = [
  ["2026-01-15T10:30:00Z", true],
  ["2026-01-15T10:30:00+05:30", true],
  ["2026-01-15T10:30:00-08:00", true],
  ["2026-01-15t10:30:00z", true],
  ["2026-01-15T10:30:00.123456Z", true],
  ["2024-02-29T00:00:00Z", true],
  ["2026-01-15T10:30:00.5+00:00", true],
  ["2026-01-15", false],
  ["2026-01-15T10:30:00", false],
  ["2026-01-15T10:30:05", false],
  ["2026-01-15T10:30Z", false],
  ["2025-02-29T00:00:00Z", false],
  ["2026-02-30T10:30:00Z", false],
  ["2026-04-31T00:00:00Z", false],
  ["2026-00-10T00:00:00Z", false],
  ["2026-13-01T10:30:00Z", false],
  ["2026-01-00T00:00:00Z", false],
  ["2026-01-15T24:00:00Z", false],
  ["2026-01-15T10:60:00Z", false],
  ["2026-01-15T10:30:61Z", false],
  ["2026-01-15T10:30:00.Z", false],
  ["20260115T103000Z", false],
  ["1700000000", false],
  ["now", false],
  ["", false],
];

describe("date-time: JSON Schema and hand-rolled validator agree", () => {
  for (const where of ["started_at", "completed_at", "result.completed_at"]) {
    for (const [value, expected] of DATE_TIME_CASES) {
      test(`${where}=${JSON.stringify(value)}`, () => {
        const doc = fidelityRs();
        if (where === "result.completed_at") doc.results[0].completed_at = value;
        else doc[where] = value;
        expect(isRfc3339DateTime(value)).toBe(expected);
        expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(expected);
        expect(validateResultSet(doc).valid, "hand-rolled").toBe(expected);
      });
    }
  }
});

test("null score with passed true: JSON Schema allows, hand-rolled rejects (Rule 6)", () => {
  // SPEC.md Validation Rule 6: "Skipped or not-yet-executed graders [...] MUST
  // be represented with `score: null` and `passed: false`." resultset.json
  // types score and passed independently and does not encode this cross-field
  // rule, so -- like SELF_PARENT and DUPLICATE_ATTEMPT -- it is enforced only
  // by the hand-rolled validators (NULL_SCORE_PASSED).
  // PROPOSED (Discussion #49, alt B): the row is also mixed, so without a
  // declaration both paths reject it; the run-level declaration isolates Rule 6.
  const doc = fidelityRs();
  doc.results[0].grader_results.push({ grader_id: "g2", type: "human", score: null, passed: true });
  expect(resultsetValidate(doc) as boolean).toBe(false);
  expect(validateResultSet(doc).errors.map(e => e.code)).toEqual(["NULL_SCORE_PASSED", "PARTIAL_RESULT_UNDECLARED"]);
  doc.metadata = { openeval: { aggregation: { strategy: "all" } } };
  expect(resultsetValidate(doc) as boolean).toBe(true);
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.map(e => e.code)).toEqual(["NULL_SCORE_PASSED"]);
});

// PROPOSED (Discussion #49, alt B): exhaustive agreement between the JSON Schema
// and the hand-rolled validator over grader shapes x passed x Result metadata x
// ResultSet metadata. Mirrors sdk/python/tests/test_schema_consistency.py.
describe("alt B (PROPOSED) truth table: JSON Schema and hand-rolled validator agree", () => {
  const G = (score: number | null, passed = true) => ({ grader_id: "g", type: "custom", score, passed });
  const SHAPES: Record<string, any[]> = {
    mixed: [G(1.0), G(null, false)],
    single: [G(1.0)],
    allnull: [G(null, false)],
    empty: [],
  };
  const DECLS: Record<string, any> = {
    none: undefined,
    dot: { "openeval.aggregation": { strategy: "all" } },
    nested: { openeval: { aggregation: { strategy: "strict" } } },
    bad_strategy: { openeval: { aggregation: { strategy: "fail-closed" } } },
    weighted_no_thr: { "openeval.aggregation": { strategy: "weighted" } },
    weighted_ok: { "openeval.aggregation": { strategy: "weighted", threshold: 0.5 } },
    thr_oob: { openeval: { aggregation: { strategy: "all", threshold: 2 } } },
    status_partial: { openeval: { aggregation_status: "partial" } },
    status_partial_dot: { "openeval.aggregation_status": "partial" },
    status_unscored: { openeval: { aggregation_status: "unscored" } },
    status_bad: { "openeval.aggregation_status": "pending" },
    not_obj: { "openeval.aggregation": "all" },
    both: { openeval: { aggregation: { strategy: "producer" }, aggregation_status: "partial" } },
  };
  let count = 0;
  const disagreements: string[] = [];
  for (const [shape, graders] of Object.entries(SHAPES)) {
    for (const passed of [true, false]) {
      if (shape === "allnull" && passed) continue; // UNSCORED_RESULT_PASSED, hand-rolled only (tested above)
      for (const [rk, rm] of Object.entries(DECLS)) {
        for (const [kk, km] of Object.entries(DECLS)) {
          const d: any = {
            version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-15T10:30:00Z",
            results: [{ test_case_id: "t", passed, grader_results: structuredClone(graders) }],
          };
          if (rm !== undefined) d.results[0].metadata = structuredClone(rm);
          if (km !== undefined) d.metadata = structuredClone(km);
          count++;
          const a = resultsetValidate(d) as boolean;
          const b = validateResultSet(d).valid;
          if (a !== b) disagreements.push(`${shape} passed=${passed} result=${rk} run=${kk}: schema=${a} hand=${b}`);
        }
      }
    }
  }
  test(`every combination agrees (${count} cases)`, () => {
    expect(count).toBe(7 * 13 * 13);
    expect(disagreements).toEqual([]);
  });
});

test("all null-scored Result with passed true: JSON Schema allows, hand-rolled rejects", () => {
  // SPEC.md Aggregation Extension: "A test case whose graders are *all*
  // null-scored has no basis for a pass/fail verdict; runners MUST report such
  // a case's `passed` as `false`". Hand-rolled only (UNSCORED_RESULT_PASSED).
  const doc = fidelityRs();
  doc.results[0].grader_results = [{ grader_id: "g1", type: "human", score: null, passed: false }];
  doc.results[0].passed = true;
  expect(resultsetValidate(doc) as boolean).toBe(true);
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.map(e => e.code)).toEqual(["UNSCORED_RESULT_PASSED"]);
});

const TESTCASE_FIDELITY_CASES: [string, Record<string, unknown>, boolean][] = [
  ["expected_output number", { expected_output: 1 }, false],
  ["context string", { context: "doc" }, false],
  ["context non-string item", { context: ["ok", 2] }, false],
  ["retrieval_context null item", { retrieval_context: [null] }, false],
  ["tools_called string", { tools_called: "search" }, false],
  ["expected_tools object item", { expected_tools: [{ name: "search" }] }, false],
  ["metadata list", { metadata: [] }, false],
  ["tags string", { tags: "smoke" }, false],
  ["provider string", { provider: "gpt-4o" }, false],
  ["provider.max_tokens 0", { provider: { max_tokens: 0 } }, false],
  ["provider.api_key_env number", { provider: { api_key_env: 1 } }, false],
  ["params list", { params: [] }, false],
  ["timeout_ms 0", { timeout_ms: 0 }, false],
  ["weight negative", { weight: -1 }, false],
  ["weight bool", { weight: true }, false],
  ["every optional field well-typed", {
    expected_output: "Paris", context: ["a"], retrieval_context: ["b"], tools_called: ["search"],
    expected_tools: ["search"], metadata: { k: 1 }, tags: ["smoke"],
    provider: { model: "gpt-4o", api_base: "x", api_key_env: "OPENAI_API_KEY", temperature: 0.2, max_tokens: 1, extra: {} },
    params: { top_p: 1 }, timeout_ms: 1, weight: 0,
  }, true],
];

describe("testcase fidelity: JSON Schema and hand-rolled validator agree", () => {
  for (const [name, extra, expected] of TESTCASE_FIDELITY_CASES) {
    test(name, () => {
      const doc = { id: "tc1", input: "hi", graders: ["g1"], ...extra };
      const hand = validateTestCase(doc);
      expect(testcaseValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(hand.valid, `hand-rolled: ${JSON.stringify(hand.errors)}`).toBe(expected);
    });
  }
});

const GRADER_FIDELITY_CASES: [string, Record<string, unknown>, boolean][] = [
  ["params list", { id: "g", type: "exact_match", params: [] }, false],
  ["weight negative", { id: "g", type: "exact_match", weight: -0.5 }, false],
  ["weight string", { id: "g", type: "exact_match", weight: "1" }, false],
  ["description number", { id: "g", type: "exact_match", description: 1 }, false],
  ["contains.ignore_case string", { id: "g", type: "contains", params: { substring: "a", ignore_case: "yes" } }, false],
  ["regex.flags number", { id: "g", type: "regex", params: { pattern: "a", flags: 1 } }, false],
  ["semantic_similarity.threshold bool", { id: "g", type: "semantic_similarity", params: { threshold: true } }, false],
  ["semantic_similarity.model number", { id: "g", type: "semantic_similarity", params: { threshold: 0.8, model: 1 } }, false],
  ["llm_judge.temperature above 2", { id: "g", type: "llm_judge", params: { model: "m", prompt: "{output}", temperature: 3 } }, false],
  ["llm_judge.schema string", { id: "g", type: "llm_judge", params: { model: "m", prompt: "{output}", schema: "x" } }, false],
  ["json_schema.strict string", { id: "g", type: "json_schema", params: { schema: {}, strict: "true" } }, false],
  ["json_path.operator not in enum", { id: "g", type: "json_path", params: { path: "$.a", expected: "1", operator: "like" } }, false],
  ["code.timeout_ms below 100", { id: "g", type: "code", params: { language: "python", source: "x", timeout_ms: 50 } }, false],
  ["contains, all optional params well-typed", { id: "g", type: "contains", params: { substring: "a", ignore_case: true }, weight: 2, description: "d" }, true],
  ["llm_judge, all optional params well-typed", { id: "g", type: "llm_judge", params: { model: "m", prompt: "{output}", provider: "p", temperature: 2, schema: {} } }, true],
  ["json_path, operator in enum", { id: "g", type: "json_path", params: { path: "$.a", expected: "1", operator: "gte" } }, true],
  ["code, timeout_ms at minimum", { id: "g", type: "code", params: { language: "python", source: "x", timeout_ms: 100 } }, true],
];

describe("grader fidelity: JSON Schema and hand-rolled validator agree", () => {
  for (const [name, doc, expected] of GRADER_FIDELITY_CASES) {
    test(name, () => {
      const hand = validateGrader(doc);
      expect(graderValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(hand.valid, `hand-rolled: ${JSON.stringify(hand.errors)}`).toBe(expected);
    });
  }
});

const SUITE_FIDELITY_CASES: [string, Record<string, unknown>, boolean][] = [
  ["name number", { name: 1 }, false],
  ["description list", { description: ["x"] }, false],
  ["graders object", { graders: { g1: {} } }, false],
  ["metadata string", { metadata: "x" }, false],
  ["tags non-string item", { tags: [1] }, false],
  ["config list", { config: [] }, false],
  ["config.parallel 0", { config: { parallel: 0 } }, false],
  ["config.provider.max_tokens 0", { config: { provider: { max_tokens: 0 } } }, false],
  ["config.defaults.timeout_ms 0", { config: { defaults: { timeout_ms: 0 } } }, false],
  ["config.defaults.weight negative", { config: { defaults: { weight: -1 } } }, false],
  ["config.retry.max_attempts 0", { config: { retry: { max_attempts: 0 } } }, false],
  ["config.retry.backoff_ms below 100", { config: { retry: { backoff_ms: 10 } } }, false],
  ["nested test case tags string", { test_cases: [{ id: "tc1", input: "hi", graders: ["g1"], tags: "smoke" }] }, false],
  ["every optional field well-typed", {
    $schema: "https://evalport.org/schema/suite.json", name: "n", description: "d", metadata: { k: 1 }, tags: ["a"],
    config: { provider: { model: "m", max_tokens: 10 }, defaults: { timeout_ms: 1000, weight: 1 },
      parallel: 4, retry: { max_attempts: 3, backoff_ms: 100 } },
  }, true],
];

describe("suite fidelity: JSON Schema and hand-rolled validator agree", () => {
  for (const [name, extra, expected] of SUITE_FIDELITY_CASES) {
    test(name, () => {
      const doc = { ...fidelitySuite(), ...extra };
      const hand = validateSuite(doc);
      expect(suiteValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(hand.valid, `hand-rolled: ${JSON.stringify(hand.errors)}`).toBe(expected);
    });
  }
});
