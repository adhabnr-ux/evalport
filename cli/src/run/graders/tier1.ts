import type { Grader } from "../../../../sdk/typescript/src/types";
import type { GraderOutcome } from "../types";
import { validateAgainstJsonSchema } from "./jsonSchema";
import { queryJsonPathFirst, JsonPathSyntaxError } from "./jsonPath";

// Tier 1 graders run entirely locally — no network calls, no API keys, no
// external npm dependencies. They're the ones that should always work,
// offline, in CI, for free.

function outcome(grader: Grader, score: number, passed: boolean, reason?: string, metadata?: Record<string, unknown>): GraderOutcome {
  return { graderId: grader.id, type: grader.type, score, passed, reason, metadata };
}

export function gradeExactMatch(grader: Grader, actualOutput: string, expectedOutput: string | undefined): GraderOutcome {
  const params = grader.params ?? {};
  const ignoreCase = params.ignore_case === true;
  const trim = params.trim_whitespace !== false; // default true per spec
  let a = actualOutput, e = expectedOutput ?? "";
  if (trim) { a = a.trim(); e = e.trim(); }
  if (ignoreCase) { a = a.toLowerCase(); e = e.toLowerCase(); }
  const match = a === e;
  return outcome(grader, match ? 1 : 0, match, match ? "exact match" : `expected "${expectedOutput ?? ""}", got "${actualOutput}"`);
}

export function gradeContains(grader: Grader, actualOutput: string): GraderOutcome {
  const params = grader.params ?? {};
  const substring = String(params.substring ?? "");
  const ignoreCase = params.ignore_case === true;
  const found = ignoreCase ? actualOutput.toLowerCase().includes(substring.toLowerCase()) : actualOutput.includes(substring);
  return outcome(grader, found ? 1 : 0, found, found ? `contains "${substring}"` : `does not contain "${substring}"`);
}

export function gradeRegex(grader: Grader, actualOutput: string): GraderOutcome {
  const params = grader.params ?? {};
  const pattern = String(params.pattern ?? "");
  const flags = typeof params.flags === "string" ? params.flags : "";
  try {
    const re = new RegExp(pattern, flags);
    const match = re.test(actualOutput);
    return outcome(grader, match ? 1 : 0, match, match ? `matched /${pattern}/${flags}` : `did not match /${pattern}/${flags}`);
  } catch (e) {
    return outcome(grader, 0, false, `invalid regex: ${(e as Error).message}`);
  }
}

export function gradeJsonSchema(grader: Grader, actualOutput: string): GraderOutcome {
  const params = grader.params ?? {};
  const schema = (params.schema ?? {}) as Record<string, unknown>;
  let parsed: unknown;
  try {
    parsed = JSON.parse(actualOutput);
  } catch (e) {
    return outcome(grader, 0, false, `output is not valid JSON: ${(e as Error).message}`);
  }
  const result = validateAgainstJsonSchema(parsed, schema);
  const reason = result.valid ? "valid against schema" : result.errors.map((er) => `${er.path}: ${er.message}`).join("; ");
  return outcome(grader, result.valid ? 1 : 0, result.valid, reason, { schema_errors: result.errors });
}

function compareValues(actual: unknown, expected: unknown, operator: string): boolean {
  switch (operator) {
    case "ne": return !looseEqual(actual, expected);
    case "gt": return toNumber(actual) > toNumber(expected);
    case "lt": return toNumber(actual) < toNumber(expected);
    case "gte": return toNumber(actual) >= toNumber(expected);
    case "lte": return toNumber(actual) <= toNumber(expected);
    case "contains": return String(actual).includes(String(expected));
    case "eq":
    default:
      return looseEqual(actual, expected);
  }
}

function looseEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  // Allow "3" == 3, "true" == true — JSONPath expected values in suite
  // files are frequently strings even when the extracted value is a number.
  if (typeof a !== typeof b && String(a) === String(b)) return true;
  return false;
}

function toNumber(v: unknown): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isNaN(n) ? NaN : n;
}

export function gradeJsonPath(grader: Grader, actualOutput: string): GraderOutcome {
  const params = grader.params ?? {};
  const path = String(params.path ?? "");
  const expected = params.expected;
  const operator = typeof params.operator === "string" ? params.operator : "eq";

  let parsed: unknown;
  try {
    parsed = JSON.parse(actualOutput);
  } catch (e) {
    return outcome(grader, 0, false, `output is not valid JSON: ${(e as Error).message}`);
  }

  let found: { found: boolean; value: unknown };
  try {
    found = queryJsonPathFirst(parsed, path);
  } catch (e) {
    if (e instanceof JsonPathSyntaxError) return outcome(grader, 0, false, `invalid JSONPath: ${e.message}`);
    throw e;
  }
  if (!found.found) return outcome(grader, 0, false, `path "${path}" matched nothing`);

  const pass = compareValues(found.value, expected, operator);
  return outcome(grader, pass ? 1 : 0, pass, pass
    ? `${JSON.stringify(found.value)} ${operator} ${JSON.stringify(expected)}`
    : `${JSON.stringify(found.value)} not ${operator} ${JSON.stringify(expected)}`);
}

export const TIER1_TYPES = new Set(["exact_match", "contains", "regex", "json_schema", "json_path"]);

export function gradeTier1(grader: Grader, actualOutput: string, expectedOutput: string | undefined): GraderOutcome {
  switch (grader.type) {
    case "exact_match": return gradeExactMatch(grader, actualOutput, expectedOutput);
    case "contains": return gradeContains(grader, actualOutput);
    case "regex": return gradeRegex(grader, actualOutput);
    case "json_schema": return gradeJsonSchema(grader, actualOutput);
    case "json_path": return gradeJsonPath(grader, actualOutput);
    default:
      throw new Error(`gradeTier1 called with non-tier-1 grader type: ${grader.type}`);
  }
}
