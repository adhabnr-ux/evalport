import type { ValidationError, ValidationResult, DocumentType, GraderType } from "./types";

// Mirrors sdk/python/openeval/validate.py rule-for-rule so both SDKs agree on
// what's valid. If you change a rule here, change it there too (and vice versa).

const STANDARD_GRADER_TYPES: ReadonlySet<string> = new Set([
  "exact_match", "contains", "regex", "semantic_similarity", "llm_judge",
  "json_schema", "json_path", "code", "human", "model graded", "custom",
]);

const SEMVER_RE = /^\d+\.\d+\.\d+(-draft)?$/;

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

export function validateTestCase(tc: unknown): ValidationResult {
  if (!isPlainObject(tc)) return ok([err("$", "Must be object", "TYPE_ERROR")]);
  const errors: ValidationError[] = [];

  if (!isNonEmptyString(tc.id)) errors.push(err("$.id", "id required", "REQUIRED"));

  const input = tc.input;
  const isStringInput = typeof input === "string";
  const isStringListInput = Array.isArray(input) && input.every((x) => typeof x === "string");
  if (!isStringInput && !isStringListInput) {
    errors.push(err("$.input", "input required", "REQUIRED"));
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

  return ok(errors);
}

export function validateGrader(g: unknown): ValidationResult {
  if (!isPlainObject(g)) return ok([err("$", "Must be object", "TYPE_ERROR")]);
  const errors: ValidationError[] = [];

  if (!isNonEmptyString(g.id)) errors.push(err("$.id", "id required", "REQUIRED"));

  const type = g.type;
  if (typeof type !== "string") {
    errors.push(err("$.type", "type required", "REQUIRED"));
  } else if (!STANDARD_GRADER_TYPES.has(type)) {
    errors.push(err("$.type", `Unknown: ${type}`, "UNKNOWN_TYPE"));
  } else {
    const params = isPlainObject(g.params) ? g.params : {};
    validateParams(type as GraderType, params).forEach((e) => errors.push(err(`$.params.${e.path}`, e.message, e.code)));
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
      if (typeof th !== "number" || th < 0 || th > 1) e.push(err("threshold", "0-1", "OUT_OF_RANGE"));
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

    const grs = Array.isArray(s.graders) ? s.graders : [];
    const gids = new Set<string>();
    grs.forEach((g, i) => {
      const gv = validateGrader(g);
      if (!gv.valid) gv.errors.forEach((e) => errors.push(err(`$.graders[${i}].${e.path}`, e.message, e.code)));
      const gid = isPlainObject(g) ? g.id : undefined;
      if (typeof gid === "string") {
        if (gids.has(gid)) errors.push(err(`$.graders[${i}].id`, `dup:${gid}`, "DUPLICATE_ID"));
        gids.add(gid);
      }
    });

    tcs.forEach((tc, i) => {
      if (isPlainObject(tc) && Array.isArray(tc.graders)) {
        tc.graders.forEach((gr, j) => {
          if (typeof gr === "string" && !gids.has(gr)) errors.push(err(`$.test_cases[${i}].graders[${j}]`, `not found:${gr}`, "DANGLING_REFERENCE"));
        });
      }
    });
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

  const rs = r.results;
  if (!Array.isArray(rs) || rs.length === 0) {
    errors.push(err("$.results", "required", "REQUIRED"));
  } else {
    rs.forEach((x, i) => {
      if (!isPlainObject(x)) { errors.push(err(`$.results[${i}]`, "object", "TYPE_ERROR")); return; }
      if (typeof x.test_case_id !== "string") errors.push(err(`$.results[${i}].test_case_id`, "required", "REQUIRED"));
      if (typeof x.passed !== "boolean") errors.push(err(`$.results[${i}].passed`, "required", "REQUIRED"));

      const grs = x.grader_results;
      if (!Array.isArray(grs)) {
        errors.push(err(`$.results[${i}].grader_results`, "required", "REQUIRED"));
      } else {
        grs.forEach((gr, j) => {
          if (!isPlainObject(gr)) { errors.push(err(`$.results[${i}].grader_results[${j}]`, "object", "TYPE_ERROR")); return; }
          if (typeof gr.grader_id !== "string") errors.push(err(`$.results[${i}].grader_results[${j}].grader_id`, "required", "REQUIRED"));
          if (typeof gr.type !== "string") errors.push(err(`$.results[${i}].grader_results[${j}].type`, "required", "REQUIRED"));
          const sc = gr.score;
          if (typeof sc !== "number" && sc !== null) errors.push(err(`$.results[${i}].grader_results[${j}].score`, "number|null", "TYPE_ERROR"));
          if (typeof gr.passed !== "boolean") errors.push(err(`$.results[${i}].grader_results[${j}].passed`, "required", "REQUIRED"));
        });
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
