import { test, expect } from "vitest";
import { validateSuite, validateGrader, validateTestCase, validateResultSet, validateDocument } from "../src/validate";

// --- suite ---

test("valid suite", () => {
  expect(validateSuite({version:"1.0.0",id:"s",graders:[{id:"g1",type:"exact_match"}],test_cases:[{id:"tc1",input:"hi",graders:["g1"]}]}).valid).toBe(true);
});
test("empty test_cases fails MIN_ITEMS", () => {
  const r = validateSuite({version:"1.0.0",id:"s",test_cases:[]});
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.code === "MIN_ITEMS")).toBe(true);
});
test("missing test_cases and test_cases_file both fail", () => {
  const r = validateSuite({version:"1.0.0",id:"s"});
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.test_cases")).toBe(true);
});
test("test_cases_file alone is accepted in place of test_cases", () => {
  const r = validateSuite({version:"1.0.0",id:"s",test_cases_file:"cases.jsonl"});
  expect(r.errors.some(e => e.path === "$.test_cases")).toBe(false);
});
test("non-object suite fails", () => {
  expect(validateSuite(null).valid).toBe(false);
  expect(validateSuite("not an object").valid).toBe(false);
  expect(validateSuite([1,2,3]).valid).toBe(false);
});
test("invalid version format", () => {
  const r = validateSuite({version:"v1",id:"s",test_cases:[{id:"tc1",input:"hi",graders:["g1"]}],graders:[{id:"g1",type:"exact_match"}]});
  expect(r.errors.some(e => e.code === "INVALID_VERSION")).toBe(true);
});
test("-draft version suffix is accepted", () => {
  const r = validateSuite({version:"1.0.0-draft",id:"s",test_cases:[{id:"tc1",input:"hi",graders:["g1"]}],graders:[{id:"g1",type:"exact_match"}]});
  expect(r.errors.some(e => e.code === "INVALID_VERSION")).toBe(false);
});
test("full semver 2.0.0 prerelease/build metadata versions are accepted", () => {
  // Previously only "X.Y.Z" or "X.Y.Z-draft" passed -- real prerelease versions
  // like "1.0.0-rc.1" (what this project's own README calls its current spec
  // version) were silently rejected as INVALID_VERSION.
  for (const v of ["1.0.0-rc.1", "1.1.0-beta.2", "2.0.0-alpha.1+build.5"]) {
    const r = validateSuite({version:v,id:"s",test_cases:[{id:"tc1",input:"hi",graders:["g1"]}],graders:[{id:"g1",type:"exact_match"}]});
    expect(r.errors.some(e => e.code === "INVALID_VERSION")).toBe(false);
  }
});
test("missing id fails", () => {
  const r = validateSuite({version:"1.0.0",test_cases:[{id:"tc1",input:"hi",graders:["g1"]}],graders:[{id:"g1",type:"exact_match"}]});
  expect(r.errors.some(e => e.path === "$.id")).toBe(true);
});
test("duplicate test case ids flagged", () => {
  const r = validateSuite({version:"1.0.0",id:"s",graders:[{id:"g1",type:"exact_match"}],test_cases:[{id:"tc1",input:"a",graders:["g1"]},{id:"tc1",input:"b",graders:["g1"]}]});
  expect(r.errors.some(e => e.code === "DUPLICATE_ID" && e.path.startsWith("$.test_cases"))).toBe(true);
});
test("duplicate grader ids flagged", () => {
  const r = validateSuite({version:"1.0.0",id:"s",graders:[{id:"g1",type:"exact_match"},{id:"g1",type:"contains",params:{substring:"x"}}],test_cases:[{id:"tc1",input:"a",graders:["g1"]}]});
  expect(r.errors.some(e => e.code === "DUPLICATE_ID" && e.path.startsWith("$.graders"))).toBe(true);
});
test("dangling string grader reference flagged", () => {
  const r = validateSuite({version:"1.0.0",id:"s",graders:[{id:"g1",type:"exact_match"}],test_cases:[{id:"tc1",input:"a",graders:["g_missing"]}]});
  expect(r.errors.some(e => e.code === "DANGLING_REFERENCE")).toBe(true);
});
test("inline dict grader is not treated as a dangling reference", () => {
  const r = validateSuite({version:"1.0.0",id:"s",graders:[],test_cases:[{id:"tc1",input:"a",graders:[{id:"g_inline",type:"contains",params:{substring:"x"}}]}]});
  expect(r.errors.some(e => e.code === "DANGLING_REFERENCE")).toBe(false);
});

// --- grader ---

test("exact_match needs no params", () => {
  expect(validateGrader({id:"g1",type:"exact_match"}).valid).toBe(true);
});
test("non-standard grader type without a handler is rejected (treated like custom)", () => {
  const r = validateGrader({id:"g1",type:"bad"});
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.params.handler")).toBe(true);
});
test("non-standard grader type WITH a handler is a valid, descriptive alternative to type: custom", () => {
  // SPEC.md's "Custom Grader Types" section: "Graders with type: 'custom' or any type
  // not in the standard set are permitted." -- a framework can use a descriptive type
  // name (e.g. one matching its own metric name) instead of the generic "custom" bucket,
  // as long as it still carries a handler for runners that don't recognize it.
  expect(validateGrader({id:"g1",type:"trulens_feedback",params:{handler:"trulens.feedback"}}).valid).toBe(true);
  expect(validateGrader({id:"g1",type:"trulens_feedback"}).valid).toBe(false);
});
test("contains requires substring", () => {
  expect(validateGrader({id:"g1",type:"contains"}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"contains",params:{substring:"x"}}).valid).toBe(true);
});
test("regex requires pattern", () => {
  expect(validateGrader({id:"g1",type:"regex"}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"regex",params:{pattern:"^a.*z$"}}).valid).toBe(true);
});
test("semantic_similarity requires threshold in [0,1]", () => {
  expect(validateGrader({id:"g1",type:"semantic_similarity"}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"semantic_similarity",params:{threshold:1.5}}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"semantic_similarity",params:{threshold:-0.1}}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"semantic_similarity",params:{threshold:0.8}}).valid).toBe(true);
  expect(validateGrader({id:"g1",type:"semantic_similarity",params:{threshold:0}}).valid).toBe(true);
  expect(validateGrader({id:"g1",type:"semantic_similarity",params:{threshold:1}}).valid).toBe(true);
});
test("llm_judge requires model and a prompt with a template token", () => {
  expect(validateGrader({id:"g1",type:"llm_judge",params:{model:"gpt-4o",prompt:"Grade this."}}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"llm_judge",params:{prompt:"Grade {output}."}}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"llm_judge",params:{model:"gpt-4o",prompt:"Grade {output} vs {expected}."}}).valid).toBe(true);
  expect(validateGrader({id:"g1",type:"llm_judge",params:{model:"gpt-4o",prompt:"Given {input}, is this right?"}}).valid).toBe(true);
});
test("json_schema requires a schema object", () => {
  expect(validateGrader({id:"g1",type:"json_schema"}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"json_schema",params:{schema:{type:"object"}}}).valid).toBe(true);
});
test("json_path requires path and expected (expected may be any value including null)", () => {
  expect(validateGrader({id:"g1",type:"json_path",params:{path:"$.a"}}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"json_path",params:{expected:1}}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"json_path",params:{path:"$.a",expected:null}}).valid).toBe(true);
});
test("code requires language in (python, javascript) and source", () => {
  expect(validateGrader({id:"g1",type:"code",params:{language:"ruby",source:"x"}}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"code",params:{language:"python"}}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"code",params:{language:"javascript",source:"assert(true)"}}).valid).toBe(true);
});
test("custom requires handler", () => {
  expect(validateGrader({id:"g1",type:"custom"}).valid).toBe(false);
  expect(validateGrader({id:"g1",type:"custom",params:{handler:"my:handler"}}).valid).toBe(true);
});
test("human and model graded need no extra params", () => {
  expect(validateGrader({id:"g1",type:"human"}).valid).toBe(true);
  expect(validateGrader({id:"g1",type:"model graded",params:{model:"gpt-4o",prompt:"{output}"}}).valid).toBe(true);
});
test("non-object grader fails", () => {
  expect(validateGrader(null).valid).toBe(false);
  expect(validateGrader("g1").valid).toBe(false);
});

// --- test case ---

test("valid test case", () => {
  expect(validateTestCase({id:"tc1",input:"hi",graders:["g1"]}).valid).toBe(true);
});
test("missing input fails", () => {
  expect(validateTestCase({id:"tc1",graders:["g1"]}).valid).toBe(false);
});
test("input may be a non-empty string list", () => {
  expect(validateTestCase({id:"tc1",input:["a","b"],graders:["g1"]}).valid).toBe(true);
  expect(validateTestCase({id:"tc1",input:[],graders:["g1"]}).valid).toBe(false);
});
test("missing or empty graders fails", () => {
  expect(validateTestCase({id:"tc1",input:"hi",graders:[]}).valid).toBe(false);
  expect(validateTestCase({id:"tc1",input:"hi"}).valid).toBe(false);
});
test("empty string grader reference fails", () => {
  expect(validateTestCase({id:"tc1",input:"hi",graders:[""]}).valid).toBe(false);
});
test("inline invalid grader dict surfaces nested errors", () => {
  const r = validateTestCase({id:"tc1",input:"hi",graders:[{id:"g1",type:"contains"}]});
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path.includes("substring"))).toBe(true);
});

// --- result set ---

test("valid result set", () => {
  expect(validateResultSet({version:"1.0.0",suite_id:"s",run_id:"r",started_at:"2026-01-01T00:00:00Z",results:[{test_case_id:"tc1",passed:true,grader_results:[{grader_id:"g1",type:"exact_match",score:1.0,passed:true}]}]}).valid).toBe(true);
});
test("empty results fails", () => {
  expect(validateResultSet({version:"1.0.0",suite_id:"s",run_id:"r",started_at:"2026-01-01T00:00:00Z",results:[]}).valid).toBe(false);
});
test("null score is accepted (e.g. a skipped grader)", () => {
  const r = validateResultSet({version:"1.0.0",suite_id:"s",run_id:"r",started_at:"2026-01-01T00:00:00Z",results:[{test_case_id:"tc1",passed:false,grader_results:[{grader_id:"g1",type:"code",score:null,passed:false}]}]});
  expect(r.valid).toBe(true);
});
test("non-numeric non-null score rejected", () => {
  const r = validateResultSet({version:"1.0.0",suite_id:"s",run_id:"r",started_at:"2026-01-01T00:00:00Z",results:[{test_case_id:"tc1",passed:true,grader_results:[{grader_id:"g1",type:"exact_match",score:"1.0" as unknown as number,passed:true}]}]});
  expect(r.valid).toBe(false);
});
test("out-of-range score rejected even though it type-checks as a number", () => {
  // The JSON Schema (spec/schemas/resultset.json) declares minimum:0/maximum:1 on
  // GraderResult.score, but this hand-written validator wasn't actually enforcing
  // it -- a score of 1.5 previously passed validation despite failing the schema.
  const r = validateResultSet({version:"1.0.0",suite_id:"s",run_id:"r",started_at:"2026-01-01T00:00:00Z",results:[{test_case_id:"tc1",passed:true,grader_results:[{grader_id:"g1",type:"exact_match",score:1.5,passed:true}]}]});
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.code === "OUT_OF_RANGE")).toBe(true);
});
test("boolean is not a valid score", () => {
  const r = validateResultSet({version:"1.0.0",suite_id:"s",run_id:"r",started_at:"2026-01-01T00:00:00Z",results:[{test_case_id:"tc1",passed:true,grader_results:[{grader_id:"g1",type:"exact_match",score:true as unknown as number,passed:true}]}]});
  expect(r.valid).toBe(false);
});
test("missing required top-level fields rejected", () => {
  expect(validateResultSet({}).valid).toBe(false);
});

// --- result set: attempt + isolation (Discussion #22 / issue #20) ---

function graderResult(score = 1.0, passed = true) {
  return { grader_id: "g1", type: "exact_match", score, passed };
}

test("multiple attempts per test_case_id validate", () => {
  const r = validateResultSet({
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: [
      { test_case_id: "tc1", attempt: 1, passed: true, grader_results: [graderResult()] },
      { test_case_id: "tc1", attempt: 2, passed: true, grader_results: [graderResult()] },
      { test_case_id: "tc1", attempt: 3, passed: false, grader_results: [graderResult(0.0, false)] },
    ],
  });
  expect(r.valid, JSON.stringify(r.errors)).toBe(true);
});

test("duplicate (test_case_id, run_id, attempt) rejected", () => {
  const r = validateResultSet({
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: [
      { test_case_id: "tc1", attempt: 1, passed: true, grader_results: [graderResult()] },
      { test_case_id: "tc1", attempt: 1, passed: false, grader_results: [graderResult(0.0, false)] },
    ],
  });
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.code === "DUPLICATE_ATTEMPT")).toBe(true);
});

test("same attempt number on different test_case_id is not a collision", () => {
  const r = validateResultSet({
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: [
      { test_case_id: "tc1", attempt: 1, passed: true, grader_results: [graderResult()] },
      { test_case_id: "tc2", attempt: 1, passed: true, grader_results: [graderResult()] },
    ],
  });
  expect(r.valid).toBe(true);
});

test("attempt must be a positive integer", () => {
  const r = validateResultSet({
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: [{ test_case_id: "tc1", attempt: 0, passed: true, grader_results: [graderResult()] }],
  });
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.code === "OUT_OF_RANGE" && e.path === "$.results[0].attempt")).toBe(true);
});

test("ResultSet-level isolation validates fine", () => {
  const r = validateResultSet({
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    isolation: "fresh",
    results: [{ test_case_id: "tc1", attempt: 1, passed: true, grader_results: [graderResult()] }],
  });
  expect(r.valid).toBe(true);
});

test("isolation is an open string, not a closed enum", () => {
  const r = validateResultSet({
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    isolation: "sandboxed_container_per_attempt",
    results: [{ test_case_id: "tc1", passed: true, grader_results: [graderResult()] }],
  });
  expect(r.valid).toBe(true);
});

test("non-string isolation rejected", () => {
  const r = validateResultSet({
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    isolation: 123 as unknown as string,
    results: [{ test_case_id: "tc1", passed: true, grader_results: [graderResult()] }],
  });
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.isolation")).toBe(true);
});

test("attempt/isolation-free result set still validates (backward compatibility)", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: [{ test_case_id: "tc1", passed: true, grader_results: [graderResult()] }],
  };
  expect(validateResultSet(doc).valid).toBe(true);
});

// --- PR #35 post-merge Copilot review: attempt-uniqueness key must not throw
// on a non-string test_case_id/run_id (github.com/adhabnr-ux/evalport/pull/35) ---

test("non-string test_case_id with attempt reports an error without throwing", () => {
  // Before the fix, JSON.stringify([x.test_case_id, runId, attempt]) could
  // throw (e.g. BigInt anywhere in the tuple) instead of returning a
  // structured validation error. A plain non-string test_case_id exercises
  // the same code path without needing a BigInt fixture.
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: [
      { test_case_id: ["not", "a", "string"] as unknown as string, attempt: 1, passed: true, grader_results: [graderResult()] },
    ],
  };
  let r: ReturnType<typeof validateResultSet> | undefined;
  expect(() => { r = validateResultSet(doc); }).not.toThrow();
  expect(r!.valid).toBe(false);
  expect(r!.errors.some(e => e.path === "$.results[0].test_case_id" && e.code === "REQUIRED")).toBe(true);
});

test("BigInt run_id with attempt reports an error without throwing", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: 1n as unknown as string, started_at: "2026-01-01T00:00:00Z",
    results: [
      { test_case_id: "tc1", attempt: 1, passed: true, grader_results: [graderResult()] },
    ],
  };
  let r: ReturnType<typeof validateResultSet> | undefined;
  expect(() => { r = validateResultSet(doc); }).not.toThrow();
  expect(r!.valid).toBe(false);
  expect(r!.errors.some(e => e.path === "$.run_id" && e.code === "REQUIRED")).toBe(true);
});

test("duplicate attempt still caught when test_case_id and run_id are valid strings", () => {
  // Guard against a regression where the crash fix above accidentally
  // disables the uniqueness check for the normal (all-strings) case.
  const r = validateResultSet({
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: [
      { test_case_id: "tc1", attempt: 1, passed: true, grader_results: [graderResult()] },
      { test_case_id: "tc1", attempt: 1, passed: false, grader_results: [graderResult(0.0, false)] },
    ],
  });
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.code === "DUPLICATE_ATTEMPT")).toBe(true);
});

// --- validateDocument dispatch ---

test("validateDocument dispatches by type", () => {
  expect(validateDocument({id:"g1",type:"exact_match"}, "grader").valid).toBe(true);
  expect(validateDocument({id:"tc1",input:"hi",graders:["g1"]}, "testcase").valid).toBe(true);
  expect(() => validateDocument({}, "bogus" as unknown as "suite").valid).toThrow();
});
