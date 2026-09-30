// Unknown fields: the strict default agrees with the raw JSON Schemas.
// Issue #107 / Discussion #108 (PROPOSED -- reference implementation, DO NOT
// MERGE). Extends schema-consistency.test.ts (same machinery: spec/schemas/*.json
// compiled with Ajv 2020 + ajv-formats) to documents with unknown keys, and pins
// ALLOWED_KEYS to the schemas' own property sets so they cannot drift.
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
  ALLOWED_KEYS,
} from "../src/validate";

const SCHEMA_DIR = join(__dirname, "..", "..", "..", "spec", "schemas");

function loadSchema(name: string): any {
  return JSON.parse(readFileSync(join(SCHEMA_DIR, `${name}.json`), "utf8"));
}

const testcaseSchema = loadSchema("testcase");
const graderSchema = loadSchema("grader");
const suiteSchema = loadSchema("suite");
const resultsetSchema = loadSchema("resultset");

const ajv = new Ajv2020({ allErrors: true, strict: false });
addFormats(ajv);
ajv.addSchema(testcaseSchema, testcaseSchema.$id);
ajv.addSchema(graderSchema, graderSchema.$id);
ajv.addSchema(suiteSchema, suiteSchema.$id);
ajv.addSchema(resultsetSchema, resultsetSchema.$id);

const testcaseValidate = ajv.getSchema(testcaseSchema.$id)!;
const graderValidate = ajv.getSchema(graderSchema.$id)!;
const suiteValidate = ajv.getSchema(suiteSchema.$id)!;
const resultsetValidate = ajv.getSchema(resultsetSchema.$id)!;

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

const r0 = (d: any) => d.results[0];
const gr0 = (d: any) => d.results[0].grader_results[0];

// ---------------------------------------------------------------------------
// Unknown fields (issue #107 / Discussion #108, PROPOSED -- DO NOT MERGE).
// Before this section, every document below with an undefined key was ACCEPTED
// by the hand-rolled validator and REJECTED by the raw JSON Schema
// (additionalProperties: false) -- the disagreement issue #107 reproduces. The
// strict default must now agree with the raw schema on every one of them, and
// the hardcoded ALLOWED_KEYS sets must equal the schemas' own property sets so
// the two cannot drift. Mirrors
// sdk/python/tests/test_schema_consistency_unknown_fields.py.
// ---------------------------------------------------------------------------

// JSON Pointer (within each schema file) of every object ALLOWED_KEYS mirrors.
const ALLOWED_KEYS_SCHEMA_LOCATIONS: Record<string, [string, string]> = {
  "testcase": ["testcase", ""],
  "testcase.provider": ["testcase", "/properties/provider"],
  "grader": ["grader", ""],
  "suite": ["suite", ""],
  "suite.config": ["suite", "/properties/config"],
  "suite.config.provider": ["suite", "/properties/config/properties/provider"],
  "suite.config.defaults": ["suite", "/properties/config/properties/defaults"],
  "suite.config.retry": ["suite", "/properties/config/properties/retry"],
  "resultset": ["resultset", ""],
  "resultset.provider": ["resultset", "/properties/provider"],
  "resultset.runner": ["resultset", "/properties/runner"],
  "resultset.group": ["resultset", "/properties/group"],
  "resultset.summary": ["resultset", "/properties/summary"],
  "resultset.result": ["resultset", "/properties/results/items"],
  "resultset.result.error": ["resultset", "/properties/results/items/properties/error"],
  "resultset.grader_result": ["resultset", "/properties/results/items/properties/grader_results/items"],
};
const SCHEMAS_BY_NAME: Record<string, any> = { testcase: testcaseSchema, grader: graderSchema, suite: suiteSchema, resultset: resultsetSchema };

function resolvePointer(schema: any, pointer: string): any {
  return pointer.split("/").filter(Boolean).reduce((node, part) => node[part], schema);
}

function closedObjectPointers(node: any, pointer = ""): string[] {
  const found: string[] = [];
  if (Array.isArray(node)) {
    node.forEach((v, i) => found.push(...closedObjectPointers(v, `${pointer}/${i}`)));
  } else if (node !== null && typeof node === "object") {
    if (node.additionalProperties === false) found.push(pointer);
    for (const [k, v] of Object.entries(node)) found.push(...closedObjectPointers(v, `${pointer}/${k}`));
  }
  return found;
}

describe("unknown fields: ALLOWED_KEYS equals each closed schema object's properties", () => {
  for (const key of Object.keys(ALLOWED_KEYS_SCHEMA_LOCATIONS).sort()) {
    test(key, () => {
      const [schemaName, pointer] = ALLOWED_KEYS_SCHEMA_LOCATIONS[key];
      const node = resolvePointer(SCHEMAS_BY_NAME[schemaName], pointer);
      expect(node.additionalProperties, `${schemaName}.json${pointer || "/"} is closed`).toBe(false);
      expect([...ALLOWED_KEYS[key]].sort()).toEqual(Object.keys(node.properties).sort());
    });
  }

  test("every closed schema object has an ALLOWED_KEYS entry (and vice versa)", () => {
    const closed = Object.entries(SCHEMAS_BY_NAME)
      .flatMap(([name, schema]) => closedObjectPointers(schema).map((p) => `${name}#${p}`))
      .sort();
    const mirrored = Object.values(ALLOWED_KEYS_SCHEMA_LOCATIONS).map(([n, p]) => `${n}#${p}`).sort();
    expect(closed).toEqual(mirrored);
    expect(Object.keys(ALLOWED_KEYS).sort()).toEqual(Object.keys(ALLOWED_KEYS_SCHEMA_LOCATIONS).sort());
  });
});

function withRs(mutate: (d: any) => void): any { const d = fidelityRs(); mutate(d); return d; }
function withSuite(mutate: (d: any) => void): any { const d = fidelitySuite(); mutate(d); return d; }

// [name, document, expected strict validity]. Every unknown key sits directly on
// a closed object; every "open" key sits under metadata / provider.extra /
// params / a summary.by_grader entry.
const UNKNOWN_FIELD_RESULTSET_CASES: [string, any, boolean][] = [
  ["root verdict (issue #107)", withRs((d) => { d.verdict = "pass"; }), false],
  ["provider", withRs((d) => { d.provider = { model: "m", seed: 1 }; }), false],
  ["runner", withRs((d) => { d.runner = { name: "x", commit: "abc" }; }), false],
  ["summary", withRs((d) => { d.summary = { total: 1, threshold: 0.8 }; }), false],
  ["group", withRs((d) => { d.group = { group_id: "g", gruop_id: "typo" }; }), false],
  ["result (issue #107)", withRs((d) => { r0(d).extra_result_key = 1; }), false],
  ["result.error", withRs((d) => { r0(d).passed = false; r0(d).error = { type: "timeout", stack: "..." }; }), false],
  ["grader_result (issue #107)", withRs((d) => { gr0(d).extra_gr_key = 1; }), false],
  ["every metadata open", withRs((d) => {
    d.metadata = { verdict: "pass", nested: { a: [1] } };
    r0(d).metadata = { extra_result_key: 1 };
    gr0(d).metadata = { extra_gr_key: 1 };
  }), true],
  ["provider.extra open", withRs((d) => { d.provider = { model: "m", extra: { seed: 1 } }; }), true],
  ["summary.by_grader entry open", withRs((d) => { d.summary = { by_grader: { g1: { passed: 1, p95_ms: 3 } } }; }), true],
];

const UNKNOWN_FIELD_SUITE_CASES: [string, any, boolean][] = [
  ["root threshold (issue #107)", withSuite((d) => { d.threshold = 0.8; }), false],
  ["config", withSuite((d) => { d.config = { concurrency: 2 }; }), false],
  ["config.provider", withSuite((d) => { d.config = { provider: { seed: 1 } }; }), false],
  ["config.defaults", withSuite((d) => { d.config = { defaults: { threshold: 0.5 } }; }), false],
  ["config.retry", withSuite((d) => { d.config = { retry: { jitter: true } }; }), false],
  ["test case (issue #107)", withSuite((d) => { d.test_cases[0].extra_tc_key = 1; }), false],
  ["test case provider", withSuite((d) => { d.test_cases[0].provider = { seed: 1 }; }), false],
  ["shared grader config (llama-cookbook#1072)", withSuite((d) => { d.graders[0].config = { x: 1 }; }), false],
  ["inline grader", withSuite((d) => { d.test_cases[0].graders = ["g1", { id: "g2", type: "exact_match", threshold: 1 }]; }), false],
  ["metadata / params / provider.extra open", withSuite((d) => {
    d.metadata = { threshold: 0.8 };
    d.config = { provider: { extra: { seed: 1 } } };
    d.graders[0].params = { ignore_case: true, custom_knob: 1 };
    Object.assign(d.test_cases[0], { metadata: { k: 1 }, params: { top_p: 1 }, provider: { extra: { a: 1 } } });
  }), true],
];

describe("unknown fields: strict default agrees with the raw JSON Schema (resultset)", () => {
  for (const [name, doc, expected] of UNKNOWN_FIELD_RESULTSET_CASES) {
    test(name, () => {
      const hand = validateResultSet(doc);
      expect(resultsetValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(hand.valid, `strict hand-rolled: ${JSON.stringify(hand.errors)}`).toBe(expected);
      if (!expected) expect([...new Set(hand.errors.map((e) => e.code))]).toEqual(["UNKNOWN_FIELD"]);
      // Leniency (consumption mode) accepts every one of these documents.
      expect(validateResultSet(doc, { allowUnknown: true }).valid).toBe(true);
    });
  }
});

describe("unknown fields: strict default agrees with the raw JSON Schema (suite)", () => {
  for (const [name, doc, expected] of UNKNOWN_FIELD_SUITE_CASES) {
    test(name, () => {
      const hand = validateSuite(doc);
      expect(suiteValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(hand.valid, `strict hand-rolled: ${JSON.stringify(hand.errors)}`).toBe(expected);
      if (!expected) expect([...new Set(hand.errors.map((e) => e.code))]).toEqual(["UNKNOWN_FIELD"]);
      expect(validateSuite(doc, { allowUnknown: true }).valid).toBe(true);
    });
  }
});

describe("unknown fields: strict default agrees with the raw JSON Schema (testcase, grader)", () => {
  const tcCases: [string, any, boolean][] = [
    ["root", { id: "tc1", input: "hi", graders: ["g1"], extra_tc_key: 1 }, false],
    ["provider", { id: "tc1", input: "hi", graders: ["g1"], provider: { seed: 1 } }, false],
    ["metadata/params open", { id: "tc1", input: "hi", graders: ["g1"], metadata: { extra_tc_key: 1 }, params: { x: 1 } }, true],
  ];
  for (const [name, doc, expected] of tcCases) {
    test(`testcase ${name}`, () => {
      expect(testcaseValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(validateTestCase(doc).valid, "strict hand-rolled").toBe(expected);
      expect(validateTestCase(doc, { allowUnknown: true }).valid).toBe(true);
    });
  }
  const graderCases: [string, any, boolean][] = [
    ["config instead of params", { id: "judge", type: "llm_judge", params: { model: "m", prompt: "{output}" }, config: { temperature: 0 } }, false],
    ["params open", { id: "judge", type: "llm_judge", params: { model: "m", prompt: "{output}", rubric_version: 2 } }, true],
  ];
  for (const [name, doc, expected] of graderCases) {
    test(`grader ${name}`, () => {
      expect(graderValidate(doc) as boolean, "JSON Schema").toBe(expected);
      expect(validateGrader(doc).valid, "strict hand-rolled").toBe(expected);
      expect(validateGrader(doc, { allowUnknown: true }).valid).toBe(true);
    });
  }
});

test("unknown fields: group unknown subfield rejected by both paths under the strict default", () => {
  const doc = withRs((d) => { d.group = { group_id: "g1", not_a_real_field: "oops" }; });
  expect(resultsetValidate(doc) as boolean).toBe(false);
  expect(validateResultSet(doc).errors.map((e) => [e.path, e.code])).toEqual([["$.group.not_a_real_field", "UNKNOWN_FIELD"]]);
});
