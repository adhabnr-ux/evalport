#!/usr/bin/env node
import { validateSuite, validateResultSet, validateGrader, validateTestCase } from "../../sdk/typescript/src/validate";
import { fromPromptfoo, computeSummary } from "../../sdk/typescript/src/convert";
import { readFileSync, writeFileSync, existsSync } from "fs";

const args = process.argv.slice(2);
const cmd = args[0];

function loadJson(p: string): unknown {
  if (!existsSync(p)) { console.error("File not found: " + p); process.exit(1); }
  return JSON.parse(readFileSync(p, "utf-8"));
}

if (cmd === "validate") {
  const file = args[1];
  if (!file) { console.error("Usage: openeval validate <file> [--type=suite|testcase|grader|resultset]"); process.exit(1); }
  const doc = loadJson(file);
  const tf = args.find(a => a.startsWith("--type="));
  const type = tf ? tf.split("=")[1] : "suite";
  const r = type==="testcase"?validateTestCase(doc):type==="grader"?validateGrader(doc):type==="resultset"?validateResultSet(doc):validateSuite(doc);
  if (r.valid) { console.log("Valid"); process.exit(0); }
  console.error("Invalid:");
  r.errors.forEach(e => console.error("  "+e.path+": "+e.message+" ["+e.code+"]"));
  process.exit(1);
}
else if (cmd === "convert") {
  const [from,to,input,output] = args.slice(1);
  if (!from||!to||!input) { console.error("Usage: openeval convert <from> <to> <input> [output]"); process.exit(1); }
  const doc = loadJson(input);
  if (from==="promptfoo"&&to==="openeval") {
    const r = fromPromptfoo(doc);
    const j = JSON.stringify(r,null,2);
    if (output) { writeFileSync(output,j); console.log("Written to "+output); }
    else console.log(j);
  } else { console.error("Unsupported: "+from+" -> "+to); process.exit(1); }
}
else if (cmd === "init") {
  const name = args[1]||"my-eval-suite";
  const suite = {
    "$schema":"https://evalport.org/schema/suite.json",
    "version":"1.0.0","id":name,"name":name,
    "graders":[{"id":"gr_exact","type":"exact_match","params":{"ignore_case":true}}],
    "test_cases":[{"id":"tc_001","input":"Example?","expected_output":"Answer","graders":["gr_exact"]}],
    "config":{"provider":{"model":"gpt-4o","temperature":0}}
  };
  writeFileSync(name+".json", JSON.stringify(suite,null,2));
  console.log("Created "+name+".json");
}
else if (cmd === "summary") {
  const file = args[1];
  if (!file) { console.error("Usage: openeval summary <resultset.json>"); process.exit(1); }
  const rs: any = loadJson(file);
  const s = rs.summary || computeSummary(rs.results);
  console.log("Total: "+(s.total||s.passed+s.failed));
  console.log("Passed: "+s.passed+"  Failed: "+s.failed);
  console.log("Pass rate: "+(s.pass_rate*100).toFixed(1)+"%");
  if (s.avg_score) console.log("Avg score: "+s.avg_score.toFixed(3));
}
else {
  console.log("EvalPort CLI v1.0.0\n\nCommands:\n  validate <file> [--type=suite|testcase|grader|resultset]  Validate an EvalPort document\n  convert <from> <to> <input> [output]                       Convert between formats\n  init [name]                                                 Create a starter eval suite\n  summary <resultset.json>                                    Print summary of a result set");
}
