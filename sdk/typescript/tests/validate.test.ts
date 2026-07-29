import { validateSuite, validateGrader, validateTestCase, validateResultSet } from "../src/validate";

test("valid suite", () => {
  expect(validateSuite({version:"1.0.0",id:"s",graders:[{id:"g1",type:"exact_match"}],test_cases:[{id:"tc1",input:"hi",graders:["g1"]}]}).valid).toBe(true);
});
test("empty", () => {
  expect(validateSuite({version:"1.0.0",id:"s",test_cases:[]}).valid).toBe(false);
});
test("grader", () => {
  expect(validateGrader({id:"g1",type:"exact_match"}).valid).toBe(true);
  expect(validateGrader({id:"g1",type:"bad"}).valid).toBe(false);
});
test("testcase", () => {
  expect(validateTestCase({id:"tc1",input:"hi",graders:["g1"]}).valid).toBe(true);
  expect(validateTestCase({id:"tc1",graders:["g1"]}).valid).toBe(false);
});
test("resultset", () => {
  expect(validateResultSet({version:"1.0.0",suite_id:"s",run_id:"r",started_at:"2026-01-01T00:00:00Z",results:[{test_case_id:"tc1",passed:true,grader_results:[{grader_id:"g1",type:"exact_match",score:1.0,passed:true}]}]}).valid).toBe(true);
});
