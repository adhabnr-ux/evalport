import { test, expect, describe } from "vitest";
import { validateSuite, validateGrader, validateTestCase, validateResultSet, validateDocument, isRfc3339DateTime } from "../src/validate";

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
test("empty string input fails MIN_LENGTH", () => {
  const r = validateTestCase({id:"tc1",input:"",graders:["g1"]});
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.input" && e.code === "MIN_LENGTH")).toBe(true);
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

// --- result set: group (Discussion #45, proposed -- grouped/sibling ResultSets) ---
// Mirrors sdk/python/tests/test_validate.py's group section rule-for-rule so
// both SDKs' hand-rolled validators agree on what's valid.

function minimalResultList() {
  return [{ test_case_id: "tc1", passed: true, grader_results: [graderResult()] }];
}

test("group absent still validates unchanged (backward compatibility)", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    results: minimalResultList(),
  };
  expect(validateResultSet(doc).valid).toBe(true);
  expect("group" in doc).toBe(false);
});

test("group with only group_id is valid", () => {
  // group_id is the only required sub-field -- role/label/sequence are all optional.
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "mutant-017-run", started_at: "2026-01-01T00:00:00Z",
    group: { group_id: "mutation-sweep-2026-09-01" },
    results: minimalResultList(),
  };
  expect(validateResultSet(doc).valid).toBe(true);
});

test("group with all fields populated is valid", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "mutant-017-run", started_at: "2026-01-01T00:00:00Z",
    group: {
      group_id: "mutation-sweep-2026-09-01",
      role: "mutant",
      label: "mutant_017 (relational-operator-swap in billing.py:42)",
      sequence: 17,
    },
    results: minimalResultList(),
  };
  expect(validateResultSet(doc).valid).toBe(true);
});

test("group missing group_id rejected", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: { role: "mutant" },
    results: minimalResultList(),
  };
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.group_id" && e.code === "REQUIRED")).toBe(true);
});

test("group empty string group_id rejected", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: { group_id: "" },
    results: minimalResultList(),
  };
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.group_id" && e.code === "REQUIRED")).toBe(true);
});

test("group must be an object", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: "mutation-sweep-2026-09-01" as unknown as object, // a bare string is not a valid group
    results: minimalResultList(),
  };
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group" && e.code === "TYPE_ERROR")).toBe(true);
});

test("group role is an open string, not an enum", () => {
  // Mirrors isolation's precedent (Discussion #22): role stays a free string so
  // a new grouping strategy never needs a spec change just to be nameable --
  // not just the mutation-testing-flavored values used in the RFC's examples.
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: { group_id: "grid-search-42", role: "candidate_config_7" },
    results: minimalResultList(),
  };
  expect(validateResultSet(doc).valid).toBe(true);
});

test("group non-string role rejected", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: { group_id: "g1", role: 42 as unknown as string },
    results: minimalResultList(),
  };
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.role" && e.code === "TYPE_ERROR")).toBe(true);
});

test("group non-string label rejected", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: { group_id: "g1", label: ["not", "a", "string"] as unknown as string },
    results: minimalResultList(),
  };
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.label" && e.code === "TYPE_ERROR")).toBe(true);
});

test("group sequence must be a non-negative integer", () => {
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: { group_id: "g1", sequence: -1 },
    results: minimalResultList(),
  };
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.sequence" && e.code === "OUT_OF_RANGE")).toBe(true);
});

test("group sequence zero is valid", () => {
  // sequence is 0-indexed -- the first member of a group is sequence 0, not 1.
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: { group_id: "g1", sequence: 0 },
    results: minimalResultList(),
  };
  expect(validateResultSet(doc).valid).toBe(true);
});

test("group sequence non-integer number rejected", () => {
  // No JS bool/int aliasing gotcha to guard against here (that's Python-
  // specific: bool is an int subclass there). TS/JS instead needs a guard
  // against non-integer numbers, since typeof 2.5 === "number" too.
  const doc = {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-01T00:00:00Z",
    group: { group_id: "g1", sequence: 2.5 },
    results: minimalResultList(),
  };
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.sequence" && e.code === "OUT_OF_RANGE")).toBe(true);
});

test("group hyperparameter sweep (second domain) is valid", () => {
  // Discussion #45's own generalization claim (mutation testing, seed sweeps,
  // model comparisons) is only as credible as its weakest-checked domain --
  // this mirrors spec/conformance/fixtures/group_hyperparameter_sweep_valid.json,
  // a hyperparameter grid search unrelated to mutation testing, with a
  // different role value ("candidate" rather than "mutant") and a sequence
  // populated the way optuna/optuna's real FrozenTrial.number is documented
  // to work ("unique and consecutive... zero-based") for a grid search whose
  // trial count is known upfront. Proves role isn't silently mutation-testing-shaped.
  const doc = {
    version: "1.1.0", suite_id: "rag-retrieval-suite", run_id: "trial-003-run",
    started_at: "2026-09-05T14:00:00Z",
    group: {
      group_id: "lr-batchsize-grid-2026-09-05",
      role: "candidate",
      label: "lr=1e-4, batch_size=32",
      sequence: 3,
    },
    results: [
      { test_case_id: "case_1", passed: true, grader_results: [{ grader_id: "gr1", type: "semantic_similarity", score: 0.88, passed: true }] },
      { test_case_id: "case_2", passed: true, grader_results: [{ grader_id: "gr1", type: "semantic_similarity", score: 0.91, passed: true }] },
    ],
  };
  const r = validateResultSet(doc);
  expect(r.valid, JSON.stringify(r.errors)).toBe(true);
});

test("group multi-model comparison (third domain) is valid", () => {
  // Discussion #45's own generalization claim named TWO use cases beyond
  // mutation testing in issue #36: "a seed-sweep or a multi-model
  // comparison." Hyperparameter sweeps were grounded above via Optuna;
  // this fixture grounds the second, previously-ungrounded one. Verified
  // against promptfoo/promptfoo's actual current source (src/types/index.ts):
  // EvaluateResult.provider: {id, label} identifies which model produced a
  // row, and CompletedPrompt.provider + its .metrics (score, testPassCount,
  // ...) is exactly the consumer-computed per-provider rollup this RFC's
  // group design already assumes -- confirming the join-key-on-member /
  // rollup-computed-by-consumer shape a fifth time (after W&B, MLflow,
  // Stryker, Optuna), independently. providers are configured as an
  // ordered array (providers: z.array(ApiProviderSchema)) and evaluated in
  // that order, so sequence is well-defined here too -- not just for
  // adaptive producers that must omit it. Provider ids and premise are
  // taken directly from promptfoo's own real example built for exactly
  // this purpose: examples/compare-claude-vs-gpt-image/promptfooconfig.yaml.
  // Mirrors spec/conformance/fixtures/group_multi_model_comparison_valid.json.
  const doc = {
    version: "1.1.0", suite_id: "image-description-suite", run_id: "openai-gpt-4.1-run",
    started_at: "2026-09-07T10:00:00Z",
    group: {
      group_id: "claude-vs-gpt-vs-gemini-image-2026-09-07",
      role: "openai:gpt-4.1",
      label: "GPT-4.1 (image description accuracy)",
      sequence: 1,
    },
    results: [
      { test_case_id: "great_wave_off_kanagawa", passed: true, grader_results: [{ grader_id: "gr1", type: "llm_judge", score: 0.92, passed: true }] },
    ],
  };
  const r = validateResultSet(doc);
  expect(r.valid, JSON.stringify(r.errors)).toBe(true);
});

test("group role 'survived' with real-gap metadata is valid", () => {
  // AshwinUgale's Discussion #45 refinement: muteval's "survived" isn't one
  // thing -- a survivor is either a real coverage gap or an inert/equivalent
  // mutant muteval excludes from its effective mutation score. role stays
  // the coarse, conventional "survived" verdict (schema doesn't distinguish
  // further); the real-gap-vs-inert bit rides in free-form metadata instead,
  // reusing muteval's own MutantOutcome.output_changed field name rather than
  // inventing new spec vocabulary. This is the real-coverage-gap branch --
  // mirrors spec/conformance/fixtures/group_role_metadata_real_gap_valid.json.
  const doc = {
    version: "1.1.0", suite_id: "billing-suite", run_id: "mutant-021-run",
    started_at: "2026-09-06T09:00:00Z",
    group: { group_id: "mutation-sweep-2026-09-06", role: "survived", sequence: 21 },
    metadata: { output_changed: true },
    results: [
      { test_case_id: "case_1", passed: true, grader_results: [{ grader_id: "gr1", type: "exact_match", score: 1.0, passed: true }] },
    ],
  };
  const r = validateResultSet(doc);
  expect(r.valid, JSON.stringify(r.errors)).toBe(true);
});

test("group role 'survived' with inert metadata is valid", () => {
  // Paired case: same coarse role: "survived" verdict, but
  // metadata.output_changed: false marks this member as one an
  // effective-mutation-score rollup should exclude -- muteval's own
  // distinction, not folded into role. Mirrors
  // spec/conformance/fixtures/group_role_metadata_inert_survivor_valid.json.
  const doc = {
    version: "1.1.0", suite_id: "billing-suite", run_id: "mutant-022-run",
    started_at: "2026-09-06T09:00:05Z",
    group: { group_id: "mutation-sweep-2026-09-06", role: "survived", sequence: 22 },
    metadata: { output_changed: false },
    results: [
      { test_case_id: "case_1", passed: true, grader_results: [{ grader_id: "gr1", type: "exact_match", score: 1.0, passed: true }] },
    ],
  };
  const r = validateResultSet(doc);
  expect(r.valid, JSON.stringify(r.errors)).toBe(true);
});

test("group role + metadata lets a consumer reconstruct raw and effective mutation score", () => {
  // End-to-end proof of the RFC refinement's actual point: given a full
  // 3-mutant group (one killed, one survived-real-gap, one
  // survived-inert-equivalent), a consumer can derive BOTH muteval's raw
  // mutation score (killed / total) and its effective score
  // (killed / (total - inert-excluded)) purely from group.role +
  // metadata.output_changed -- no schema change, no growth in role's
  // vocabulary, exactly what AshwinUgale's comment asked whether this
  // design could support.
  const groupId = "mutation-sweep-2026-09-06-full";
  const members = [
    {
      version: "1.1.0", suite_id: "billing-suite", run_id: "mutant-020-run",
      started_at: "2026-09-06T08:59:55Z",
      group: { group_id: groupId, role: "killed", sequence: 20 },
      results: [{ test_case_id: "case_1", passed: false, grader_results: [{ grader_id: "gr1", type: "exact_match", score: 0.0, passed: false }] }],
    },
    {
      version: "1.1.0", suite_id: "billing-suite", run_id: "mutant-021-run",
      started_at: "2026-09-06T09:00:00Z",
      group: { group_id: groupId, role: "survived", sequence: 21 },
      metadata: { output_changed: true },
      results: [{ test_case_id: "case_1", passed: true, grader_results: [{ grader_id: "gr1", type: "exact_match", score: 1.0, passed: true }] }],
    },
    {
      version: "1.1.0", suite_id: "billing-suite", run_id: "mutant-022-run",
      started_at: "2026-09-06T09:00:05Z",
      group: { group_id: groupId, role: "survived", sequence: 22 },
      metadata: { output_changed: false },
      results: [{ test_case_id: "case_1", passed: true, grader_results: [{ grader_id: "gr1", type: "exact_match", score: 1.0, passed: true }] }],
    },
  ];

  for (const m of members) {
    const r = validateResultSet(m);
    expect(r.valid, JSON.stringify(r.errors)).toBe(true);
  }

  // Consumer-side rollup math (deliberately NOT part of the schema/validator
  // -- the RFC explicitly declines to standardize this, see "What this
  // deliberately does not do" in SPEC.md). Demonstrated here only to prove
  // both numbers are actually reconstructible from the documents above.
  const total = members.length;
  const killed = members.filter(m => m.group.role === "killed").length;
  const inertExcluded = members.filter(
    m => m.group.role === "survived" && (m as { metadata?: { output_changed?: boolean } }).metadata?.output_changed === false
  ).length;
  const rawScore = killed / total;
  const effectiveScore = killed / (total - inertExcluded);

  expect(rawScore).toBeCloseTo(1 / 3);
  expect(effectiveScore).toBeCloseTo(1 / 2);
  expect(effectiveScore).toBeGreaterThan(rawScore); // excluding the inert survivor raises the score, as it should
});

// --- group.parent_group_id: nested/hierarchical groups (sweep-of-sweeps) ---
//
// Mirrors sdk/python/tests/test_validate.py's parent_group_id section
// rule-for-rule. Grew out of an explicit "is a flat group the right model?"
// audit against real systems: MLflow's nested runs (mlflow/tracking/fluent.py's
// active_run_stack, each run carrying one parent_run_id that can itself point
// to a run with its own parent_run_id) form an arbitrarily deep tree in real
// usage -- confirmed against mlflow/mlflow#16685's actual GrandParent/Parent/
// 150-Child test case, not just the API surface. W&B's Run.sweep_id and its
// separate wandb.init(group=...) primitive are both flat, single-level, with
// no parent construct anywhere in wandb/wandb's source -- so EvalPort's
// flat-only design matched W&B but not MLflow. parent_group_id closes that
// gap the same way group_id itself is modeled: a pointer on the member, not
// an embedded tree.

function rsWithGroup(group: Record<string, unknown>, runId = "mutant-017-run") {
  return {
    version: "1.0.0", suite_id: "s", run_id: runId, started_at: "2026-01-01T00:00:00Z",
    group,
    results: minimalResultList(),
  };
}

test("group parent_group_id absent is valid and unchanged", () => {
  const doc = rsWithGroup({ group_id: "mutation-sweep-2026-09-01" });
  const r = validateResultSet(doc);
  expect(r.valid).toBe(true);
  expect("parent_group_id" in doc.group).toBe(false);
});

test("group parent_group_id valid nested sweep", () => {
  const doc = rsWithGroup({
    group_id: "child-sweep-lr-1e-4",
    parent_group_id: "parent-sweep-lr-batchsize-grid-2026-09-08",
    role: "candidate",
    sequence: 3,
  });
  expect(validateResultSet(doc).valid).toBe(true);
});

test("group parent_group_id empty string rejected", () => {
  const doc = rsWithGroup({ group_id: "g1", parent_group_id: "" });
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.parent_group_id" && e.code === "REQUIRED")).toBe(true);
});

test("group parent_group_id non-string rejected", () => {
  const doc = rsWithGroup({ group_id: "g1", parent_group_id: 42 as unknown as string });
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.parent_group_id" && e.code === "REQUIRED")).toBe(true);
});

test("group parent_group_id equal to group_id rejected", () => {
  const doc = rsWithGroup({ group_id: "sweep-42", parent_group_id: "sweep-42" });
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.some(e => e.path === "$.group.parent_group_id" && e.code === "SELF_PARENT")).toBe(true);
});

test("group three-level nesting matches MLflow GrandParent/Parent/Child shape", () => {
  const grandparent = rsWithGroup({ group_id: "campaign-2026-09-08" }, "grandparent-run");
  const parent = rsWithGroup(
    { group_id: "sweep-lr-grid", parent_group_id: "campaign-2026-09-08" }, "parent-run"
  );
  const child = rsWithGroup(
    { group_id: "trial-003", parent_group_id: "sweep-lr-grid", sequence: 3 }, "child-run"
  );
  for (const doc of [grandparent, parent, child]) {
    const r = validateResultSet(doc);
    expect(r.valid).toBe(true);
  }
  const byGroupId: Record<string, typeof child> = {};
  for (const doc of [grandparent, parent, child]) {
    byGroupId[(doc.group as { group_id: string }).group_id] = doc;
  }
  const chain: string[] = [(child.group as { group_id: string }).group_id];
  let cur = child;
  while (true) {
    const pgid = (cur.group as { parent_group_id?: string }).parent_group_id;
    if (!pgid || !(pgid in byGroupId)) break;
    cur = byGroupId[pgid];
    chain.push((cur.group as { group_id: string }).group_id);
  }
  expect(chain).toEqual(["trial-003", "sweep-lr-grid", "campaign-2026-09-08"]);
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

// --- Validator fidelity: optional-field types, RFC 3339 timestamps, Rule 6 ---
// The hand-rolled validator used to accept documents spec/schemas/*.json rejects:
// an optional field of the wrong type (e.g. Result.actual_output as a list --
// found by ChelseaKR in ChelseaKR/gauntlet#76), a date-only or offset-less
// started_at/completed_at, and a null-scored GraderResult with passed: true.
// Mirrors sdk/python/tests/test_validate.py case-for-case.

function fidRs(): any {
  return {
    version: "1.0.0", suite_id: "s", run_id: "r", started_at: "2026-01-15T10:30:00Z",
    results: [{
      test_case_id: "tc1", passed: true,
      grader_results: [{ grader_id: "g1", type: "exact_match", score: 1.0, passed: true }],
    }],
  };
}

function setPath(doc: any, path: (string | number)[], value: unknown): any {
  let cur = doc;
  for (const k of path.slice(0, -1)) cur = cur[k];
  cur[path[path.length - 1]] = value;
  return doc;
}

const R0: (string | number)[] = ["results", 0];
const GR0: (string | number)[] = ["results", 0, "grader_results", 0];

// [name, path to set, value, expected error path, expected code]
const RESULTSET_REJECT_CASES: [string, (string | number)[], unknown, string, string][] = [
  ["suite_version not string", ["suite_version"], 1, "$.suite_version", "TYPE_ERROR"],
  ["completed_at date-only", ["completed_at"], "2026-01-15", "$.completed_at", "INVALID_DATE_TIME"],
  ["completed_at not string", ["completed_at"], 1700000000, "$.completed_at", "TYPE_ERROR"],
  ["provider not object", ["provider"], "gpt-4o", "$.provider", "TYPE_ERROR"],
  ["provider null", ["provider"], null, "$.provider", "TYPE_ERROR"],
  ["provider.model not string", ["provider"], { model: 4 }, "$.provider.model", "TYPE_ERROR"],
  ["provider.api_base not string", ["provider"], { api_base: ["x"] }, "$.provider.api_base", "TYPE_ERROR"],
  ["provider.temperature string", ["provider"], { temperature: "0.1" }, "$.provider.temperature", "TYPE_ERROR"],
  ["provider.temperature bool", ["provider"], { temperature: true }, "$.provider.temperature", "TYPE_ERROR"],
  ["provider.max_tokens fractional", ["provider"], { max_tokens: 1.5 }, "$.provider.max_tokens", "TYPE_ERROR"],
  ["provider.max_tokens bool", ["provider"], { max_tokens: true }, "$.provider.max_tokens", "TYPE_ERROR"],
  ["provider.extra not object", ["provider"], { extra: [] }, "$.provider.extra", "TYPE_ERROR"],
  ["runner not object", ["runner"], [], "$.runner", "TYPE_ERROR"],
  ["runner.name not string", ["runner"], { name: 1 }, "$.runner.name", "TYPE_ERROR"],
  ["runner.version not string", ["runner"], { version: 1.0 }, "$.runner.version", "TYPE_ERROR"],
  ["summary not object", ["summary"], [], "$.summary", "TYPE_ERROR"],
  ["summary.total negative", ["summary"], { total: -1 }, "$.summary.total", "OUT_OF_RANGE"],
  ["summary.total bool", ["summary"], { total: true }, "$.summary.total", "TYPE_ERROR"],
  ["summary.passed fractional", ["summary"], { passed: 0.5 }, "$.summary.passed", "TYPE_ERROR"],
  ["summary.failed string", ["summary"], { failed: "0" }, "$.summary.failed", "TYPE_ERROR"],
  ["summary.skipped negative", ["summary"], { skipped: -2 }, "$.summary.skipped", "OUT_OF_RANGE"],
  ["summary.pass_rate above 1", ["summary"], { pass_rate: 1.5 }, "$.summary.pass_rate", "OUT_OF_RANGE"],
  ["summary.avg_score string", ["summary"], { avg_score: "0.5" }, "$.summary.avg_score", "TYPE_ERROR"],
  ["summary.duration_ms fractional", ["summary"], { duration_ms: 1.5 }, "$.summary.duration_ms", "TYPE_ERROR"],
  ["summary.by_grader not object", ["summary"], { by_grader: [] }, "$.summary.by_grader", "TYPE_ERROR"],
  ["summary.by_grader entry not object", ["summary"], { by_grader: { g1: 3 } }, "$.summary.by_grader.g1", "TYPE_ERROR"],
  ["summary.by_grader.passed fractional", ["summary"], { by_grader: { g1: { passed: 1.5 } } }, "$.summary.by_grader.g1.passed", "TYPE_ERROR"],
  ["summary.by_grader.failed bool", ["summary"], { by_grader: { g1: { failed: false } } }, "$.summary.by_grader.g1.failed", "TYPE_ERROR"],
  ["summary.by_grader.avg_score string", ["summary"], { by_grader: { g1: { avg_score: "x" } } }, "$.summary.by_grader.g1.avg_score", "TYPE_ERROR"],
  ["metadata not object", ["metadata"], [], "$.metadata", "TYPE_ERROR"],
  ["metadata null", ["metadata"], null, "$.metadata", "TYPE_ERROR"],
  ["isolation null", ["isolation"], null, "$.isolation", "TYPE_ERROR"],
  ["group null", ["group"], null, "$.group", "TYPE_ERROR"],
  ["started_at date-only", ["started_at"], "2026-01-15", "$.started_at", "INVALID_DATE_TIME"],
  ["started_at no offset", ["started_at"], "2026-01-15T10:30:00", "$.started_at", "INVALID_DATE_TIME"],
  ["started_at no seconds", ["started_at"], "2026-01-15T10:30Z", "$.started_at", "INVALID_DATE_TIME"],
  ["started_at space separator", ["started_at"], "2026-01-15 10:30:00Z", "$.started_at", "INVALID_DATE_TIME"],
  ["started_at offset without colon", ["started_at"], "2026-01-15T10:30:00+0530", "$.started_at", "INVALID_DATE_TIME"],
  ["started_at impossible day", ["started_at"], "2026-02-30T10:30:00Z", "$.started_at", "INVALID_DATE_TIME"],
  ["started_at month 13", ["started_at"], "2026-13-01T10:30:00Z", "$.started_at", "INVALID_DATE_TIME"],
  ["started_at hour 24", ["started_at"], "2026-01-15T24:00:00Z", "$.started_at", "INVALID_DATE_TIME"],
  ["started_at epoch string", ["started_at"], "1700000000", "$.started_at", "INVALID_DATE_TIME"],
  // Result
  ["actual_output list (ChelseaKR/gauntlet#76)", [...R0, "actual_output"], ["a", "b"], "$.results[0].actual_output", "TYPE_ERROR"],
  ["actual_output object", [...R0, "actual_output"], { text: "a" }, "$.results[0].actual_output", "TYPE_ERROR"],
  ["actual_output null", [...R0, "actual_output"], null, "$.results[0].actual_output", "TYPE_ERROR"],
  ["test_case_id empty", [...R0, "test_case_id"], "", "$.results[0].test_case_id", "REQUIRED"],
  ["duration_ms fractional", [...R0, "duration_ms"], 1.5, "$.results[0].duration_ms", "TYPE_ERROR"],
  ["duration_ms negative", [...R0, "duration_ms"], -1, "$.results[0].duration_ms", "OUT_OF_RANGE"],
  ["duration_ms bool", [...R0, "duration_ms"], true, "$.results[0].duration_ms", "TYPE_ERROR"],
  ["result completed_at offset-less", [...R0, "completed_at"], "2026-01-15T10:30:05", "$.results[0].completed_at", "INVALID_DATE_TIME"],
  ["error not object", [...R0, "error"], "boom", "$.results[0].error", "TYPE_ERROR"],
  ["error.type not in enum", [...R0, "error"], { type: "crash" }, "$.results[0].error.type", "INVALID_VALUE"],
  ["error.message not string", [...R0, "error"], { message: 1 }, "$.results[0].error.message", "TYPE_ERROR"],
  ["error.code fractional", [...R0, "error"], { code: 1.5 }, "$.results[0].error.code", "TYPE_ERROR"],
  ["error.code bool", [...R0, "error"], { code: true }, "$.results[0].error.code", "TYPE_ERROR"],
  ["error.retryable string", [...R0, "error"], { retryable: "yes" }, "$.results[0].error.retryable", "TYPE_ERROR"],
  ["result metadata not object", [...R0, "metadata"], "trace", "$.results[0].metadata", "TYPE_ERROR"],
  ["attempt null", [...R0, "attempt"], null, "$.results[0].attempt", "OUT_OF_RANGE"],
  // GraderResult
  ["grader_id empty", [...GR0, "grader_id"], "", "$.results[0].grader_results[0].grader_id", "REQUIRED"],
  ["reason not string", [...GR0, "reason"], 1, "$.results[0].grader_results[0].reason", "TYPE_ERROR"],
  ["grader metadata not object", [...GR0, "metadata"], [], "$.results[0].grader_results[0].metadata", "TYPE_ERROR"],
];

describe("resultset optional-field rejections", () => {
  for (const [name, path, value, errPath, code] of RESULTSET_REJECT_CASES) {
    test(name, () => {
      const r = validateResultSet(setPath(fidRs(), path, value));
      expect(r.valid).toBe(false);
      expect(r.errors.some(e => e.path === errPath && e.code === code), JSON.stringify(r.errors)).toBe(true);
    });
  }
});

test("GraderResult with no score key is REQUIRED, not treated as null", () => {
  const doc = fidRs();
  delete doc.results[0].grader_results[0].score;
  doc.results[0].grader_results[0].passed = false;
  doc.results[0].passed = false;
  const r = validateResultSet(doc);
  expect(r.errors.some(e => e.path === "$.results[0].grader_results[0].score" && e.code === "REQUIRED")).toBe(true);
});

test("Rule 6: null score with passed true is rejected (NULL_SCORE_PASSED)", () => {
  const doc = fidRs();
  doc.results[0].grader_results.push({ grader_id: "g2", type: "human", score: null, passed: true });
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.map(e => e.code)).toEqual(["NULL_SCORE_PASSED"]);
  expect(r.errors[0].path).toBe("$.results[0].grader_results[1].passed");
});

test("Rule 6: all null-scored Result with passed true is rejected (UNSCORED_RESULT_PASSED)", () => {
  const doc = fidRs();
  doc.results[0].grader_results = [
    { grader_id: "g1", type: "human", score: null, passed: false },
    { grader_id: "g2", type: "llm_judge", score: null, passed: false },
  ];
  const r = validateResultSet(doc);
  expect(r.valid).toBe(false);
  expect(r.errors.map(e => [e.path, e.code])).toEqual([["$.results[0].passed", "UNSCORED_RESULT_PASSED"]]);
});

test("Rule 6: all null-scored Result with passed false is valid", () => {
  const doc = fidRs();
  doc.results[0].passed = false;
  doc.results[0].grader_results = [{ grader_id: "g1", type: "human", score: null, passed: false }];
  expect(validateResultSet(doc).valid).toBe(true);
});

test("Rule 6: mixed null and scored graders can still pass", () => {
  // Null-scored graders are excluded from aggregation, so a Result can pass on
  // its scored graders alone -- only the ALL-null case must be passed: false.
  const doc = fidRs();
  doc.results[0].grader_results.push({ grader_id: "g2", type: "human", score: null, passed: false });
  expect(validateResultSet(doc).valid).toBe(true);
});

test("Rule 6: empty grader_results is not 'all null-scored'", () => {
  const doc = fidRs();
  doc.results[0].grader_results = [];
  expect(validateResultSet(doc).valid).toBe(true);
});

test("ResultSet with every optional field well-typed is valid", () => {
  const doc = fidRs();
  Object.assign(doc, {
    $schema: "https://evalport.org/schema/resultset.json",
    suite_version: "1.0.0",
    completed_at: "2026-01-15T10:31:45.123+05:30",
    provider: { model: "gpt-4o", api_base: "https://api.example.com", temperature: 0, max_tokens: 256, extra: { seed: 1 } },
    runner: { name: "evalport-cli", version: "1.3.1" },
    summary: { total: 1, passed: 1, failed: 0, skipped: 0, pass_rate: 1, avg_score: 1.0, duration_ms: 1200,
      by_grader: { g1: { passed: 1, failed: 0, avg_score: 1.0 } } },
    metadata: { "openeval.partial": false },
  });
  Object.assign(doc.results[0], {
    actual_output: "Paris", duration_ms: 1200, completed_at: "2026-01-15t10:31:44z",
    error: { type: "provider_error", message: "rate limited", code: 429, retryable: true },
    metadata: { trace_id: "abc" },
  });
  Object.assign(doc.results[0].grader_results[0], { reason: "exact", metadata: { k: "v" } });
  const r = validateResultSet(doc);
  expect(r.errors).toEqual([]);
  expect(r.valid).toBe(true);
});

test("error.code may be a string or an integer", () => {
  for (const code of ["RATE_LIMIT", 429]) {
    expect(validateResultSet(setPath(fidRs(), [...R0, "error"], { type: "timeout", code })).valid).toBe(true);
  }
});

test("undefined-valued optional keys count as absent (JSON.stringify drops them)", () => {
  const doc = fidRs();
  doc.provider = undefined;
  doc.results[0].actual_output = undefined;
  expect(validateResultSet(doc).valid).toBe(true);
});

const RFC3339_ACCEPT = [
  "2026-01-15T10:30:00Z", "2026-01-15T10:30:00+05:30", "2026-01-15T10:30:00-08:00",
  "2026-01-15t10:30:00z", "2026-01-15T10:30:00.123456Z", "2024-02-29T00:00:00Z",
  "2026-01-15T10:30:00.5+00:00", "2016-12-31T23:59:60Z",
];
const RFC3339_REJECT = [
  "2026-01-15", "2026-01-15T10:30:00", "2026-01-15T10:30Z", "2026-01-15 10:30:00Z",
  "2026-01-15T10:30:00+0530", "2026-01-15T10:30:00+05", "2025-02-29T00:00:00Z", "2026-04-31T00:00:00Z",
  "2026-00-10T00:00:00Z", "2026-01-00T00:00:00Z", "2026-01-15T10:60:00Z", "2026-01-15T10:30:61Z",
  "2026-01-15T10:30:00.Z", "2026-01-15T10:30:00Z\n", "20260115T103000Z", "", "now",
];

describe("RFC 3339 date-time", () => {
  for (const v of RFC3339_ACCEPT) {
    test(`accepts ${JSON.stringify(v)}`, () => {
      expect(isRfc3339DateTime(v)).toBe(true);
      expect(validateResultSet(setPath(fidRs(), ["started_at"], v)).valid).toBe(true);
    });
  }
  for (const v of RFC3339_REJECT) {
    test(`rejects ${JSON.stringify(v)}`, () => {
      expect(isRfc3339DateTime(v)).toBe(false);
      const r = validateResultSet(setPath(fidRs(), ["started_at"], v));
      expect(r.errors.some(e => e.path === "$.started_at" && e.code === "INVALID_DATE_TIME")).toBe(true);
    });
  }
  test("Date.prototype.toISOString() output is valid", () => {
    expect(isRfc3339DateTime(new Date().toISOString())).toBe(true);
  });
});

// TestCase / Grader / Suite optional fields

const TC_BASE = { id: "tc1", input: "hi", graders: ["g1"] };
const TESTCASE_REJECT_CASES: [string, unknown, string, string][] = [
  ["expected_output", 1, "$.expected_output", "TYPE_ERROR"],
  ["expected_output", null, "$.expected_output", "TYPE_ERROR"],
  ["context", "doc", "$.context", "TYPE_ERROR"],
  ["context", ["ok", 2], "$.context", "TYPE_ERROR"],
  ["retrieval_context", [null], "$.retrieval_context", "TYPE_ERROR"],
  ["tools_called", "search", "$.tools_called", "TYPE_ERROR"],
  ["expected_tools", [{ name: "search" }], "$.expected_tools", "TYPE_ERROR"],
  ["metadata", [], "$.metadata", "TYPE_ERROR"],
  ["tags", "smoke", "$.tags", "TYPE_ERROR"],
  ["provider", "gpt-4o", "$.provider", "TYPE_ERROR"],
  ["provider", { max_tokens: 0 }, "$.provider.max_tokens", "OUT_OF_RANGE"],
  ["provider", { api_key_env: 1 }, "$.provider.api_key_env", "TYPE_ERROR"],
  ["provider", { temperature: "hot" }, "$.provider.temperature", "TYPE_ERROR"],
  ["params", [], "$.params", "TYPE_ERROR"],
  ["timeout_ms", 0, "$.timeout_ms", "OUT_OF_RANGE"],
  ["timeout_ms", 1.5, "$.timeout_ms", "TYPE_ERROR"],
  ["weight", -1, "$.weight", "OUT_OF_RANGE"],
  ["weight", true, "$.weight", "TYPE_ERROR"],
];

describe("testcase optional-field rejections", () => {
  for (const [key, value, errPath, code] of TESTCASE_REJECT_CASES) {
    test(`${key}=${JSON.stringify(value)}`, () => {
      const r = validateTestCase({ ...TC_BASE, [key]: value });
      expect(r.valid).toBe(false);
      expect(r.errors.some(e => e.path === errPath && e.code === code), JSON.stringify(r.errors)).toBe(true);
    });
  }
});

test("testcase with every optional field well-typed is valid", () => {
  const r = validateTestCase({
    ...TC_BASE, expected_output: "Paris", context: ["a"], retrieval_context: ["b"],
    tools_called: ["search"], expected_tools: ["search"], metadata: { k: 1 }, tags: ["smoke"],
    provider: { model: "gpt-4o", api_base: "x", api_key_env: "OPENAI_API_KEY", temperature: 0.2, max_tokens: 1, extra: {} },
    params: { top_p: 1 }, timeout_ms: 1, weight: 0,
  });
  expect(r.errors).toEqual([]);
});

const GRADER_REJECT_CASES: [Record<string, unknown>, string, string][] = [
  [{ id: "g", type: "exact_match", params: [] }, "$.params", "TYPE_ERROR"],
  [{ id: "g", type: "custom", params: ["x"] }, "$.params", "TYPE_ERROR"],
  [{ id: "g", type: "exact_match", weight: -0.5 }, "$.weight", "OUT_OF_RANGE"],
  [{ id: "g", type: "exact_match", weight: "1" }, "$.weight", "TYPE_ERROR"],
  [{ id: "g", type: "exact_match", description: 1 }, "$.description", "TYPE_ERROR"],
  [{ id: "g", type: "contains", params: { substring: "a", ignore_case: "yes" } }, "$.params.ignore_case", "TYPE_ERROR"],
  [{ id: "g", type: "regex", params: { pattern: "a", flags: 1 } }, "$.params.flags", "TYPE_ERROR"],
  [{ id: "g", type: "semantic_similarity", params: { threshold: true } }, "$.params.threshold", "OUT_OF_RANGE"],
  [{ id: "g", type: "semantic_similarity", params: { threshold: 0.8, model: 1 } }, "$.params.model", "TYPE_ERROR"],
  [{ id: "g", type: "semantic_similarity", params: { threshold: 0.8, provider: 1 } }, "$.params.provider", "TYPE_ERROR"],
  [{ id: "g", type: "llm_judge", params: { model: "m", prompt: "{output}", temperature: 3 } }, "$.params.temperature", "OUT_OF_RANGE"],
  [{ id: "g", type: "llm_judge", params: { model: "m", prompt: "{output}", schema: "x" } }, "$.params.schema", "TYPE_ERROR"],
  [{ id: "g", type: "llm_judge", params: { model: "m", prompt: "{output}", provider: 1 } }, "$.params.provider", "TYPE_ERROR"],
  [{ id: "g", type: "json_schema", params: { schema: {}, strict: "true" } }, "$.params.strict", "TYPE_ERROR"],
  [{ id: "g", type: "json_path", params: { path: "$.a", expected: "1", operator: "like" } }, "$.params.operator", "INVALID_VALUE"],
  [{ id: "g", type: "code", params: { language: "python", source: "x", timeout_ms: 50 } }, "$.params.timeout_ms", "OUT_OF_RANGE"],
];

describe("grader optional-field rejections", () => {
  for (const [doc, errPath, code] of GRADER_REJECT_CASES) {
    test(`${doc.type}:${errPath}`, () => {
      const r = validateGrader(doc);
      expect(r.valid).toBe(false);
      expect(r.errors.some(e => e.path === errPath && e.code === code), JSON.stringify(r.errors)).toBe(true);
    });
  }
});

test("grader with non-object params still reports missing handler", () => {
  const r = validateGrader({ id: "g", type: "custom", params: ["x"] });
  expect(new Set(r.errors.map(e => `${e.path}|${e.code}`))).toEqual(new Set(["$.params|TYPE_ERROR", "$.params.handler|REQUIRED"]));
});

test("grader optional params well-typed are valid", () => {
  for (const g of [
    { id: "g", type: "contains", params: { substring: "a", ignore_case: true }, weight: 2, description: "d" },
    { id: "g", type: "regex", params: { pattern: "a", flags: "i" } },
    { id: "g", type: "semantic_similarity", params: { threshold: 0.8, model: "m", provider: "p" } },
    { id: "g", type: "llm_judge", params: { model: "m", prompt: "{output}", provider: "p", temperature: 2, schema: {} } },
    { id: "g", type: "json_schema", params: { schema: {}, strict: false } },
    { id: "g", type: "json_path", params: { path: "$.a", expected: "1", operator: "gte" } },
    { id: "g", type: "code", params: { language: "python", source: "x", timeout_ms: 100 } },
  ]) {
    expect(validateGrader(g).errors, JSON.stringify(g)).toEqual([]);
  }
});

const SUITE_BASE = { version: "1.0.0", id: "s", graders: [{ id: "g1", type: "exact_match" }], test_cases: [TC_BASE] };
const SUITE_REJECT_CASES: [string, unknown, string, string][] = [
  ["name", 1, "$.name", "TYPE_ERROR"],
  ["description", ["x"], "$.description", "TYPE_ERROR"],
  ["graders", { g1: {} }, "$.graders", "TYPE_ERROR"],
  ["test_cases_file", 1, "$.test_cases_file", "TYPE_ERROR"],
  ["metadata", "x", "$.metadata", "TYPE_ERROR"],
  ["tags", [1], "$.tags", "TYPE_ERROR"],
  ["config", [], "$.config", "TYPE_ERROR"],
  ["config", { parallel: 0 }, "$.config.parallel", "OUT_OF_RANGE"],
  ["config", { provider: [] }, "$.config.provider", "TYPE_ERROR"],
  ["config", { provider: { max_tokens: 0 } }, "$.config.provider.max_tokens", "OUT_OF_RANGE"],
  ["config", { defaults: { timeout_ms: 0 } }, "$.config.defaults.timeout_ms", "OUT_OF_RANGE"],
  ["config", { defaults: { weight: -1 } }, "$.config.defaults.weight", "OUT_OF_RANGE"],
  ["config", { retry: { max_attempts: 0 } }, "$.config.retry.max_attempts", "OUT_OF_RANGE"],
  ["config", { retry: { backoff_ms: 10 } }, "$.config.retry.backoff_ms", "OUT_OF_RANGE"],
  ["$schema", 1, "$.$schema", "TYPE_ERROR"],
];

describe("suite optional-field rejections", () => {
  for (const [key, value, errPath, code] of SUITE_REJECT_CASES) {
    test(`${key}=${JSON.stringify(value)}`, () => {
      const r = validateSuite({ ...SUITE_BASE, [key]: value });
      expect(r.valid).toBe(false);
      expect(r.errors.some(e => e.path === errPath && e.code === code), JSON.stringify(r.errors)).toBe(true);
    });
  }
});

test("suite nested test case optional-field rejection is reported", () => {
  // Nested errors keep validateSuite's existing "$.test_cases[i]." + "$.<field>" prefixing.
  const r = validateSuite({ ...SUITE_BASE, test_cases: [{ ...TC_BASE, tags: "smoke" }] });
  expect(r.errors.map(e => [e.path, e.code])).toEqual([["$.test_cases[0].$.tags", "TYPE_ERROR"]]);
});

test("suite test_cases of the wrong type is rejected even with test_cases_file", () => {
  const r = validateSuite({ version: "1.0.0", id: "s", test_cases_file: "cases.jsonl", test_cases: "cases.jsonl" });
  expect(r.errors.some(e => e.path === "$.test_cases" && e.code === "TYPE_ERROR")).toBe(true);
});

test("suite graders are validated when test cases come from test_cases_file", () => {
  const r = validateSuite({ version: "1.0.0", id: "s", test_cases_file: "cases.jsonl", graders: [{ id: "g1", type: "custom" }] });
  expect(r.errors.some(e => e.path === "$.graders[0].$.params.handler")).toBe(true);
});

test("suite with every optional field well-typed is valid", () => {
  const r = validateSuite({
    ...SUITE_BASE, $schema: "https://evalport.org/schema/suite.json", name: "n", description: "d",
    metadata: { k: 1 }, tags: ["a"],
    config: { provider: { model: "m", max_tokens: 10 }, defaults: { timeout_ms: 1000, weight: 1 },
      parallel: 4, retry: { max_attempts: 3, backoff_ms: 100 } },
  });
  expect(r.errors).toEqual([]);
});
