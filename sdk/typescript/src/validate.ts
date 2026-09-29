import type { ValidationError, ValidationResult, DocumentType, GraderType } from "./types";

// Mirrors sdk/python/openeval/validate.py rule-for-rule so both SDKs agree on
// what's valid. If you change a rule here, change it there too (and vice versa).

const STANDARD_GRADER_TYPES: ReadonlySet<string> = new Set([
  "exact_match", "contains", "regex", "semantic_similarity", "llm_judge",
  "json_schema", "json_path", "code", "human", "model graded", "custom",
]);

// Full semver 2.0.0 pattern (https://semver.org/#backusnaur-form-grammar-for-valid-semver-versions).
// Was previously hardcoded to accept only "X.Y.Z" or "X.Y.Z-draft" -- rejected legitimate
// prerelease versions like "1.0.0-rc.1" or "1.1.0-beta.2", which is what this project's own
// README and public communications already describe the spec version as.
export const SEMVER_RE =
  /^\d+\.\d+\.\d+(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;

// RFC 3339 section 5.6 `date-time` (what resultset.json's `format: "date-time"` means):
//   full-date "T" full-time, where full-time = partial-time time-offset,
//   partial-time = HH ":" MM ":" SS [time-secfrac], time-offset = "Z" / ("+" / "-") HH ":" MM.
// Seconds and an offset are both mandatory: a date-only value ("2026-01-15") or an
// offset-less local time ("2026-01-15T10:30:00") is NOT an RFC 3339 date-time, even
// though both are valid ISO 8601. RFC 3339 5.6 NOTE: "T" and "Z" may alternatively be
// lower case. time-second allows 60 (leap second) per the RFC 3339 grammar.
// Mirrors RFC3339_DATE_TIME_RE in sdk/python/openeval/validate.py (which uses
// fullmatch; JS `$` without the m flag never matches before a trailing newline).
export const RFC3339_DATE_TIME_RE =
  /^([0-9]{4})-([0-9]{2})-([0-9]{2})[Tt](?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])$/;

const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/** True iff v is a string that is a valid RFC 3339 date-time (incl. calendar-valid day). */
export function isRfc3339DateTime(v: unknown): boolean {
  if (typeof v !== "string") return false;
  const m = RFC3339_DATE_TIME_RE.exec(v);
  if (!m) return false;
  const year = Number(m[1]), month = Number(m[2]), day = Number(m[3]);
  if (month < 1 || month > 12) return false;
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const maxDay = month === 2 && leap ? 29 : DAYS_IN_MONTH[month - 1];
  return day >= 1 && day <= maxDay;
}

const DATE_TIME_MESSAGE = "must be an RFC 3339 date-time with seconds and an offset, e.g. 2026-01-15T10:30:00Z";

function err(path: string, message: string, code: string): ValidationError {
  return { path, message, code };
}

function ok(errors: ValidationError[]): ValidationResult {
  return { valid: errors.length === 0, errors };
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isNonEmptyString(v: unknown): v is string {
  return typeof v === "string" && v.length > 0;
}

// JSON Schema type predicates. typeof already excludes booleans from "number",
// matching JSON Schema ("integer"/"number" never match true/false).
function isNum(v: unknown): v is number {
  return typeof v === "number";
}

function isInt(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v);
}

/** A key counts as present unless it is absent or `undefined` (which JSON cannot carry). */
function has(obj: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(obj, key) && obj[key] !== undefined;
}

type FieldSpec =
  | [string, "string" | "object" | "boolean" | "string_array" | "string_or_integer" | "date_time"]
  | [string, "integer" | "number", (number | null)?, (number | null)?]
  | [string, "enum", readonly string[]];

// Optional-field type checks. A field is checked only when its key is PRESENT --
// including an explicit null, which JSON Schema's `type` keyword rejects for every
// field below (none of them is declared nullable). Unknown keys are deliberately NOT
// policed here (see tests/schema-consistency.test.ts -- that's a separate spec
// question). Mirrors _check_optional() in sdk/python/openeval/validate.py
// rule-for-rule, including error codes and messages.
function checkOptional(obj: Record<string, unknown>, base: string, fields: readonly FieldSpec[], errors: ValidationError[]): void {
  for (const f of fields) {
    const [key, kind] = f;
    if (!has(obj, key)) continue;
    const v = obj[key];
    const path = `${base}.${key}`;
    switch (kind) {
      case "string":
        if (typeof v !== "string") errors.push(err(path, "must be string", "TYPE_ERROR"));
        break;
      case "object":
        if (!isPlainObject(v)) errors.push(err(path, "must be object", "TYPE_ERROR"));
        break;
      case "boolean":
        if (typeof v !== "boolean") errors.push(err(path, "must be boolean", "TYPE_ERROR"));
        break;
      case "string_array":
        if (!Array.isArray(v) || !v.every((x) => typeof x === "string")) errors.push(err(path, "must be array of strings", "TYPE_ERROR"));
        break;
      case "string_or_integer":
        if (typeof v !== "string" && !isInt(v)) errors.push(err(path, "must be string or integer", "TYPE_ERROR"));
        break;
      case "date_time":
        if (typeof v !== "string") errors.push(err(path, "must be string", "TYPE_ERROR"));
        else if (!isRfc3339DateTime(v)) errors.push(err(path, DATE_TIME_MESSAGE, "INVALID_DATE_TIME"));
        break;
      case "enum": {
        const allowed = f[2] as readonly string[];
        if (typeof v !== "string" || !allowed.includes(v)) errors.push(err(path, `must be one of: ${allowed.join(", ")}`, "INVALID_VALUE"));
        break;
      }
      case "integer":
      case "number": {
        const lo = (f[2] ?? null) as number | null;
        const hi = (f[3] ?? null) as number | null;
        if (!(kind === "integer" ? isInt(v) : isNum(v))) {
          errors.push(err(path, `must be ${kind === "integer" ? "an integer" : "a number"}`, "TYPE_ERROR"));
        } else if ((lo !== null && (v as number) < lo) || (hi !== null && (v as number) > hi)) {
          const rng = hi === null ? `>= ${lo}` : lo === null ? `<= ${hi}` : `in [${lo},${hi}]`;
          errors.push(err(path, `must be ${rng}`, "OUT_OF_RANGE"));
        }
        break;
      }
    }
  }
}

// spec/schemas/testcase.json + suite.json `provider` (both declare max_tokens minimum: 1).
const CONFIG_PROVIDER_FIELDS: readonly FieldSpec[] = [["model", "string"], ["api_base", "string"], ["api_key_env", "string"], ["temperature", "number"], ["max_tokens", "integer", 1], ["extra", "object"]];
// spec/schemas/testcase.json optional properties.
const TESTCASE_FIELDS: readonly FieldSpec[] = [
  ["expected_output", "string"], ["context", "string_array"], ["retrieval_context", "string_array"],
  ["tools_called", "string_array"], ["expected_tools", "string_array"], ["metadata", "object"],
  ["tags", "string_array"], ["provider", "object"], ["params", "object"],
  ["timeout_ms", "integer", 1], ["weight", "number", 0],
];
// spec/schemas/grader.json optional properties.
const GRADER_FIELDS: readonly FieldSpec[] = [["params", "object"], ["weight", "number", 0], ["description", "string"]];
// spec/schemas/grader.json per-type allOf/then optional params.
const GRADER_PARAM_FIELDS: Readonly<Record<string, readonly FieldSpec[]>> = {
  contains: [["ignore_case", "boolean"]],
  regex: [["flags", "string"]],
  semantic_similarity: [["model", "string"], ["provider", "string"]],
  llm_judge: [["provider", "string"], ["temperature", "number", 0, 2], ["schema", "object"]],
  json_schema: [["strict", "boolean"]],
  json_path: [["operator", "enum", ["eq", "ne", "gt", "lt", "gte", "lte", "contains"]]],
  code: [["timeout_ms", "integer", 100]],
};
// spec/schemas/suite.json optional properties (graders/test_cases arrays are
// checked inline in validateSuite, since their items are validated there too).
const SUITE_FIELDS: readonly FieldSpec[] = [
  ["$schema", "string"], ["name", "string"], ["description", "string"],
  ["test_cases_file", "string"], ["config", "object"], ["metadata", "object"], ["tags", "string_array"],
];
const SUITE_CONFIG_FIELDS: readonly FieldSpec[] = [["provider", "object"], ["defaults", "object"], ["parallel", "integer", 1], ["retry", "object"]];
const SUITE_DEFAULTS_FIELDS: readonly FieldSpec[] = [["timeout_ms", "integer", 1], ["weight", "number", 0]];
const SUITE_RETRY_FIELDS: readonly FieldSpec[] = [["max_attempts", "integer", 1], ["backoff_ms", "integer", 100]];
// spec/schemas/resultset.json optional top-level properties (group/isolation are
// checked separately below).
const RESULTSET_FIELDS: readonly FieldSpec[] = [
  ["$schema", "string"], ["suite_version", "string"], ["completed_at", "date_time"],
  ["provider", "object"], ["runner", "object"], ["summary", "object"], ["metadata", "object"],
];
const RESULTSET_PROVIDER_FIELDS: readonly FieldSpec[] = [["model", "string"], ["api_base", "string"], ["temperature", "number"], ["max_tokens", "integer"], ["extra", "object"]];
const RUNNER_FIELDS: readonly FieldSpec[] = [["name", "string"], ["version", "string"]];
const SUMMARY_FIELDS: readonly FieldSpec[] = [
  ["total", "integer", 0], ["passed", "integer", 0], ["failed", "integer", 0], ["skipped", "integer", 0],
  ["pass_rate", "number", 0, 1], ["avg_score", "number", 0, 1], ["duration_ms", "integer", 0], ["by_grader", "object"],
];
const BY_GRADER_ENTRY_FIELDS: readonly FieldSpec[] = [["passed", "integer"], ["failed", "integer"], ["avg_score", "number"]];
const RESULT_FIELDS: readonly FieldSpec[] = [
  ["actual_output", "string"], ["duration_ms", "integer", 0], ["completed_at", "date_time"],
  ["error", "object"], ["metadata", "object"],
];
const RESULT_ERROR_FIELDS: readonly FieldSpec[] = [
  ["type", "enum", ["timeout", "provider_error", "runner_error"]], ["message", "string"],
  ["code", "string_or_integer"], ["retryable", "boolean"],
];
const GRADER_RESULT_FIELDS: readonly FieldSpec[] = [["reason", "string"], ["metadata", "object"]];

export function validateTestCase(tc: unknown): ValidationResult {
  if (!isPlainObject(tc)) return ok([err("$", "Must be object", "TYPE_ERROR")]);
  const errors: ValidationError[] = [];

  if (!isNonEmptyString(tc.id)) errors.push(err("$.id", "id required", "REQUIRED"));

  const input = tc.input;
  const isStringInput = typeof input === "string";
  const isStringListInput = Array.isArray(input) && input.every((x) => typeof x === "string");
  if (!isStringInput && !isStringListInput) {
    errors.push(err("$.input", "input required", "REQUIRED"));
  } else if (typeof input === "string" && input.length === 0) {
    errors.push(err("$.input", "empty", "MIN_LENGTH"));
  } else if (isStringListInput && (input as unknown[]).length === 0) {
    errors.push(err("$.input", "empty", "MIN_ITEMS"));
  }

  const graders = tc.graders;
  if (!Array.isArray(graders) || graders.length === 0) {
    errors.push(err("$.graders", "graders required", "REQUIRED"));
  } else {
    graders.forEach((g, i) => {
      if (typeof g === "string") {
        if (g.length === 0) errors.push(err(`$.graders[${i}]`, "empty", "EMPTY_STRING"));
      } else if (isPlainObject(g)) {
        const gv = validateGrader(g);
        if (!gv.valid) gv.errors.forEach((e) => errors.push(err(`$.graders[${i}].${e.path}`, e.message, e.code)));
      } else {
        errors.push(err(`$.graders[${i}]`, "must be string or object", "TYPE_ERROR"));
      }
    });
  }

  checkOptional(tc, "$", TESTCASE_FIELDS, errors);
  if (isPlainObject(tc.provider)) checkOptional(tc.provider, "$.provider", CONFIG_PROVIDER_FIELDS, errors);

  return ok(errors);
}

export function validateGrader(g: unknown): ValidationResult {
  if (!isPlainObject(g)) return ok([err("$", "Must be object", "TYPE_ERROR")]);
  const errors: ValidationError[] = [];

  if (!isNonEmptyString(g.id)) errors.push(err("$.id", "id required", "REQUIRED"));
  checkOptional(g, "$", GRADER_FIELDS, errors);

  const type = g.type;
  if (!isNonEmptyString(type)) {
    errors.push(err("$.type", "type required", "REQUIRED"));
  } else {
    // A non-object params already produced a TYPE_ERROR above; validate the
    // per-type requirements against {} so they're still reported.
    const params = isPlainObject(g.params) ? g.params : {};
    if (STANDARD_GRADER_TYPES.has(type)) {
      validateParams(type as GraderType, params).forEach((e) => errors.push(err(`$.params.${e.path}`, e.message, e.code)));
    } else {
      // Non-standard type name: valid, but treated exactly like "custom" -- must
      // identify a handler so a runner that doesn't recognize it can skip gracefully
      // instead of guessing. This is what lets an adapter use a descriptive type
      // (e.g. "trulens_feedback") instead of the generic "custom" bucket.
      validateParams("custom", params).forEach((e) => errors.push(err(`$.params.${e.path}`, e.message, e.code)));
    }
  }

  return ok(errors);
}

function validateParams(type: GraderType, p: Record<string, unknown>): ValidationError[] {
  const e: ValidationError[] = [];
  switch (type) {
    case "contains":
      if (!isNonEmptyString(p.substring)) e.push(err("substring", "required", "REQUIRED"));
      break;
    case "regex":
      if (!isNonEmptyString(p.pattern)) e.push(err("pattern", "required", "REQUIRED"));
      break;
    case "semantic_similarity": {
      const th = p.threshold;
      if (!isNum(th) || th < 0 || th > 1) e.push(err("threshold", "0-1", "OUT_OF_RANGE"));
      break;
    }
    case "llm_judge": {
      if (!isNonEmptyString(p.model)) e.push(err("model", "required", "REQUIRED"));
      const pr = p.prompt;
      if (!isNonEmptyString(pr)) {
        e.push(err("prompt", "required", "REQUIRED"));
      } else if (!pr.includes("{output}") && !pr.includes("{input}") && !pr.includes("{expected}")) {
        e.push(err("prompt", "missing token", "MISSING_TOKEN"));
      }
      break;
    }
    case "json_schema":
      if (!isPlainObject(p.schema)) e.push(err("schema", "required", "REQUIRED"));
      break;
    case "json_path":
      if (!isNonEmptyString(p.path)) e.push(err("path", "required", "REQUIRED"));
      if (!("expected" in p)) e.push(err("expected", "required", "REQUIRED"));
      break;
    case "code":
      if (p.language !== "python" && p.language !== "javascript") e.push(err("language", "python|javascript", "INVALID_VALUE"));
      if (!isNonEmptyString(p.source)) e.push(err("source", "required", "REQUIRED"));
      break;
    case "custom":
      if (!isNonEmptyString(p.handler)) e.push(err("handler", "required", "REQUIRED"));
      break;
    // exact_match, human, model graded: no required params.
  }
  // Optional, typed per-type params (grader.json allOf/then blocks). Paths here are
  // relative to params (validateGrader prefixes "$.params.").
  const opt: ValidationError[] = [];
  checkOptional(p, "$", GRADER_PARAM_FIELDS[type] ?? [], opt);
  opt.forEach((x) => e.push(err(x.path.slice(2), x.message, x.code)));
  return e;
}

export function validateSuite(s: unknown): ValidationResult {
  if (!isPlainObject(s)) return ok([err("$", "Must be object", "TYPE_ERROR")]);
  const errors: ValidationError[] = [];

  if (!isNonEmptyString(s.version) || !SEMVER_RE.test(s.version)) errors.push(err("$.version", "semver", "INVALID_VERSION"));
  if (!isNonEmptyString(s.id)) errors.push(err("$.id", "required", "REQUIRED"));

  const tcs = s.test_cases;
  const hasTestCasesFile = typeof s.test_cases_file === "string";
  if (!Array.isArray(tcs) && !hasTestCasesFile) errors.push(err("$.test_cases", "required", "REQUIRED"));
  else if (has(s, "test_cases") && !Array.isArray(tcs)) errors.push(err("$.test_cases", "must be array", "TYPE_ERROR"));
  checkOptional(s, "$", SUITE_FIELDS, errors);
  const grs = has(s, "graders") ? s.graders : [];
  if (!Array.isArray(grs)) errors.push(err("$.graders", "must be array", "TYPE_ERROR"));
  const cfg = s.config;
  if (isPlainObject(cfg)) {
    checkOptional(cfg, "$.config", SUITE_CONFIG_FIELDS, errors);
    if (isPlainObject(cfg.provider)) checkOptional(cfg.provider, "$.config.provider", CONFIG_PROVIDER_FIELDS, errors);
    if (isPlainObject(cfg.defaults)) checkOptional(cfg.defaults, "$.config.defaults", SUITE_DEFAULTS_FIELDS, errors);
    if (isPlainObject(cfg.retry)) checkOptional(cfg.retry, "$.config.retry", SUITE_RETRY_FIELDS, errors);
  }

  // Shared graders are validated whether test cases are inline or in
  // test_cases_file -- previously an invalid grader in a test_cases_file suite
  // was never looked at.
  const gids = new Set<string>();
  if (Array.isArray(grs)) {
    grs.forEach((g, i) => {
      const gv = validateGrader(g);
      if (!gv.valid) gv.errors.forEach((e) => errors.push(err(`$.graders[${i}].${e.path}`, e.message, e.code)));
      const gid = isPlainObject(g) ? g.id : undefined;
      if (typeof gid === "string") {
        if (gids.has(gid)) errors.push(err(`$.graders[${i}].id`, `dup:${gid}`, "DUPLICATE_ID"));
        gids.add(gid);
      }
    });
  }

  if (Array.isArray(tcs)) {
    if (tcs.length === 0) errors.push(err("$.test_cases", "empty", "MIN_ITEMS"));

    const ids = new Set<string>();
    tcs.forEach((tc, i) => {
      const tv = validateTestCase(tc);
      if (!tv.valid) tv.errors.forEach((e) => errors.push(err(`$.test_cases[${i}].${e.path}`, e.message, e.code)));
      const tid = isPlainObject(tc) ? tc.id : undefined;
      if (typeof tid === "string") {
        if (ids.has(tid)) errors.push(err(`$.test_cases[${i}].id`, `dup:${tid}`, "DUPLICATE_ID"));
        ids.add(tid);
      }
    });

    if (Array.isArray(grs)) {
      tcs.forEach((tc, i) => {
        if (isPlainObject(tc) && Array.isArray(tc.graders)) {
          tc.graders.forEach((gr, j) => {
            if (typeof gr === "string" && !gids.has(gr)) errors.push(err(`$.test_cases[${i}].graders[${j}]`, `not found:${gr}`, "DANGLING_REFERENCE"));
          });
        }
      });
    }
  }

  return ok(errors);
}

export function validateResultSet(r: unknown): ValidationResult {
  if (!isPlainObject(r)) return ok([err("$", "Must be object", "TYPE_ERROR")]);
  const errors: ValidationError[] = [];

  if (!isNonEmptyString(r.version) || !SEMVER_RE.test(r.version)) errors.push(err("$.version", "semver", "INVALID_VERSION"));
  if (!isNonEmptyString(r.suite_id)) errors.push(err("$.suite_id", "required", "REQUIRED"));
  if (!isNonEmptyString(r.run_id)) errors.push(err("$.run_id", "required", "REQUIRED"));
  if (typeof r.started_at !== "string") errors.push(err("$.started_at", "required", "REQUIRED"));
  else if (!isRfc3339DateTime(r.started_at)) errors.push(err("$.started_at", DATE_TIME_MESSAGE, "INVALID_DATE_TIME"));
  const runId = r.run_id;
  checkOptional(r, "$", RESULTSET_FIELDS, errors);
  if (isPlainObject(r.provider)) checkOptional(r.provider, "$.provider", RESULTSET_PROVIDER_FIELDS, errors);
  if (isPlainObject(r.runner)) checkOptional(r.runner, "$.runner", RUNNER_FIELDS, errors);
  const summary = r.summary;
  if (isPlainObject(summary)) {
    checkOptional(summary, "$.summary", SUMMARY_FIELDS, errors);
    if (isPlainObject(summary.by_grader)) {
      for (const [gk, gv] of Object.entries(summary.by_grader)) {
        if (!isPlainObject(gv)) errors.push(err(`$.summary.by_grader.${gk}`, "must be object", "TYPE_ERROR"));
        else checkOptional(gv, `$.summary.by_grader.${gk}`, BY_GRADER_ENTRY_FIELDS, errors);
      }
    }
  }
  // isolation / group / group.* / attempt below: like every other optional field,
  // an explicit null is a type error (resultset.json declares none of them nullable).
  if (has(r, "isolation") && typeof r.isolation !== "string") {
    errors.push(err("$.isolation", "must be string", "TYPE_ERROR"));
  }

  // Discussion #45 / PR #54: optional group membership joining sibling
  // ResultSets (a sweep, a mutation-testing run, a multi-model comparison).
  // Absent by default -- a no-op for every ResultSet produced before this.
  // Mirrors sdk/python/openeval/validate.py's validate_result_set rule-for-rule:
  // like every other optional sub-object in this file, unknown keys inside
  // `group` are NOT policed here (that's the raw JSON Schema's
  // additionalProperties: false job -- see tests/schema-consistency.test.ts).
  if (has(r, "group")) {
    const group = r.group;
    if (!isPlainObject(group)) {
      errors.push(err("$.group", "must be object", "TYPE_ERROR"));
    } else {
      if (!isNonEmptyString(group.group_id)) errors.push(err("$.group.group_id", "required", "REQUIRED"));
      if (has(group, "parent_group_id")) {
        const pgid = group.parent_group_id;
        if (!isNonEmptyString(pgid)) {
          errors.push(err("$.group.parent_group_id", "must be a non-empty string", "REQUIRED"));
        } else if (isNonEmptyString(group.group_id) && pgid === group.group_id) {
          errors.push(err("$.group.parent_group_id", "a group cannot be its own parent", "SELF_PARENT"));
        }
      }
      if (has(group, "role") && typeof group.role !== "string") {
        errors.push(err("$.group.role", "must be string", "TYPE_ERROR"));
      }
      if (has(group, "label") && typeof group.label !== "string") {
        errors.push(err("$.group.label", "must be string", "TYPE_ERROR"));
      }
      if (has(group, "sequence")) {
        const seq = group.sequence;
        if (!isInt(seq) || seq < 0) {
          errors.push(err("$.group.sequence", "must be an integer >= 0", "OUT_OF_RANGE"));
        }
      }
    }
  }

  const rs = r.results;
  if (!Array.isArray(rs) || rs.length === 0) {
    errors.push(err("$.results", "required", "REQUIRED"));
  } else {
    // Discussion #22 / issue #20: (test_case_id, run_id, attempt) must be
    // unique across results whenever attempt is present -- the join key for
    // repeated trials of the same test case (LangSmith num_repetitions,
    // Promptfoo repeats, Inspect AI epochs). attempt is absent-by-default and
    // single-attempt-per-case ResultSets need no change; this check is a no-op
    // unless a producer actually opts into attempt.
    const seenAttempts = new Set<string>();
    rs.forEach((x, i) => {
      if (!isPlainObject(x)) { errors.push(err(`$.results[${i}]`, "object", "TYPE_ERROR")); return; }
      if (!isNonEmptyString(x.test_case_id)) errors.push(err(`$.results[${i}].test_case_id`, "required", "REQUIRED"));
      if (typeof x.passed !== "boolean") errors.push(err(`$.results[${i}].passed`, "required", "REQUIRED"));
      checkOptional(x, `$.results[${i}]`, RESULT_FIELDS, errors);
      if (isPlainObject(x.error)) checkOptional(x.error, `$.results[${i}].error`, RESULT_ERROR_FIELDS, errors);

      if (has(x, "attempt")) {
        const attempt = x.attempt;
        if (!isInt(attempt) || attempt < 1) {
          errors.push(err(`$.results[${i}].attempt`, "must be an integer >= 1", "OUT_OF_RANGE"));
        } else if (typeof x.test_case_id === "string" && typeof runId === "string") {
          // Only build the uniqueness key once test_case_id/run_id are confirmed
          // strings: JSON.stringify throws on values like BigInt, and either
          // field being malformed already produces its own structured error
          // above ($.run_id / $.results[i].test_case_id), so skipping the
          // dedup check for those entries -- rather than crashing -- is safe.
          const key = JSON.stringify([x.test_case_id, runId, attempt]);
          if (seenAttempts.has(key)) {
            errors.push(err(`$.results[${i}].attempt`, `duplicate (test_case_id, run_id, attempt): ${key}`, "DUPLICATE_ATTEMPT"));
          } else {
            seenAttempts.add(key);
          }
        }
      }

      const grs = x.grader_results;
      if (!Array.isArray(grs)) {
        errors.push(err(`$.results[${i}].grader_results`, "required", "REQUIRED"));
      } else {
        grs.forEach((gr, j) => {
          if (!isPlainObject(gr)) { errors.push(err(`$.results[${i}].grader_results[${j}]`, "object", "TYPE_ERROR")); return; }
          const p = `$.results[${i}].grader_results[${j}]`;
          if (!isNonEmptyString(gr.grader_id)) errors.push(err(`${p}.grader_id`, "required", "REQUIRED"));
          if (typeof gr.type !== "string") errors.push(err(`${p}.type`, "required", "REQUIRED"));
          // score is REQUIRED (resultset.json) -- an absent key is not the same
          // as an explicit null ("not verified", Rule 6).
          if (!has(gr, "score")) {
            errors.push(err(`${p}.score`, "required", "REQUIRED"));
          } else {
            const sc = gr.score;
            if (sc !== null && !isNum(sc)) {
              errors.push(err(`${p}.score`, "number|null", "TYPE_ERROR"));
            } else if (sc !== null && (sc < 0 || sc > 1)) {
              errors.push(err(`${p}.score`, "must be in [0,1] or null", "OUT_OF_RANGE"));
            }
          }
          if (typeof gr.passed !== "boolean") {
            errors.push(err(`${p}.passed`, "required", "REQUIRED"));
          } else if (has(gr, "score") && gr.score === null && gr.passed === true) {
            // SPEC.md Validation Rules -> 6. Result Consistency: "Skipped or
            // not-yet-executed graders (e.g. `human` review pending, an
            // `unsupported_grader_type`, a runner error before scoring) MUST be
            // represented with `score: null` and `passed: false`. `score: null`
            // means "not verified" [...]"
            errors.push(err(`${p}.passed`, "a GraderResult with score null (not verified) MUST have passed false (SPEC Rule 6)", "NULL_SCORE_PASSED"));
          }
          checkOptional(gr, p, GRADER_RESULT_FIELDS, errors);
        });
        // SPEC.md Extension Mechanism -> Aggregation Extension: "A test case
        // whose graders are *all* null-scored has no basis for a pass/fail
        // verdict; runners MUST report such a case's `passed` as `false` [...]"
        if (grs.length > 0 && x.passed === true && grs.every((gr) => isPlainObject(gr) && has(gr, "score") && gr.score === null)) {
          errors.push(err(`$.results[${i}].passed`, "a Result whose grader_results are all null-scored MUST have passed false (SPEC Aggregation Extension / Rule 6)", "UNSCORED_RESULT_PASSED"));
        }
      }
    });
  }

  return ok(errors);
}

export function validateDocument(d: unknown, t: DocumentType): ValidationResult {
  if (t === "testcase") return validateTestCase(d);
  if (t === "grader") return validateGrader(d);
  if (t === "suite") return validateSuite(d);
  if (t === "resultset") return validateResultSet(d);
  throw new Error(`Unknown type: ${t}`);
}
