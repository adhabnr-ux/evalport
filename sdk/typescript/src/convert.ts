import type { EvalSuite, TestCase, Grader, ResultSet, Result, Summary } from "./types";
import { OPENEVAL_VERSION } from "./types";

export function fromPromptfoo(pf: unknown): EvalSuite {
  const p = pf as Record<string, unknown>;
  const tests = (p.tests ?? []) as Record<string, unknown>[];
  const providers = (p.providers ?? []) as Record<string, unknown>[];
  const graders: Grader[] = [];
  const test_cases: TestCase[] = tests.map((t, i) => {
    const vars = (t.vars ?? {}) as Record<string, unknown>;
    const asserts = (t.assert ?? []) as Record<string, unknown>[];
    const tcGraders: string[] = [];
    for (let j = 0; j < asserts.length; j++) {
      const gId = `gr_${i}_${j}`;
      graders.push(promptfooAssertToGrader(gId, asserts[j]));
      tcGraders.push(gId);
    }
    const input = typeof vars.query === "string" ? vars.query : typeof vars.prompt === "string" ? vars.prompt : JSON.stringify(vars);
    return { id: `tc_${i}`, input, expected_output: typeof vars.expected === "string" ? vars.expected : undefined, context: Array.isArray(vars.context) ? vars.context as string[] : undefined, graders: tcGraders.length > 0 ? tcGraders : ["gr_default"] };
  });
  const provider = providers[0];
  return { version: OPENEVAL_VERSION, id: "suite_promptfoo_import", name: "Imported from Promptfoo", graders: graders.length > 0 ? graders : [{ id: "gr_default", type: "exact_match" }], test_cases, config: provider ? { provider: { model: typeof provider.model === "string" ? provider.model : undefined } } : undefined, metadata: { openeval: { source: "promptfoo" } } };
}

function promptfooAssertToGrader(id: string, a: Record<string, unknown>): Grader {
  const type = a.type as string;
  switch (type) {
    case "equals": return { id, type: "exact_match" };
    case "contains": return { id, type: "contains", params: { substring: String(a.value ?? "") } };
    case "regex": return { id, type: "regex", params: { pattern: String(a.value ?? "") } };
    case "contains-json": return { id, type: "json_schema", params: { schema: {} } };
    case "ic": return { id, type: "llm_judge", params: { model: "gpt-4o", prompt: String(a.value ?? "Evaluate the output.") } };
    default: return { id, type: "custom", params: { handler: `promptfoo:${type}` } };
  }
}

export function computeSummary(results: Result[]): Summary {
  const total = results.length;
  let passed = 0, failed = 0, skipped = 0, scoreSum = 0, scoreCount = 0;
  const byGrader: Record<string, { passed: number; failed: number; scoreSum: number; scoreCount: number }> = {};
  for (const r of results) {
    if (r.passed) passed++; else failed++;
    for (const gr of r.grader_results) {
      if (!byGrader[gr.grader_id]) byGrader[gr.grader_id] = { passed: 0, failed: 0, scoreSum: 0, scoreCount: 0 };
      if (gr.passed) byGrader[gr.grader_id].passed++;
      else if (gr.score === null) skipped++;
      else byGrader[gr.grader_id].failed++;
      if (gr.score !== null) { scoreSum += gr.score; scoreCount++; byGrader[gr.grader_id].scoreSum += gr.score; byGrader[gr.grader_id].scoreCount++; }
    }
  }
  const byGraderSummary: Record<string, { passed: number; failed: number; avg_score: number }> = {};
  for (const [id, v] of Object.entries(byGrader)) byGraderSummary[id] = { passed: v.passed, failed: v.failed, avg_score: v.scoreCount > 0 ? v.scoreSum / v.scoreCount : 0 };
  return { total, passed, failed, skipped, pass_rate: total > 0 ? passed / total : 0, avg_score: scoreCount > 0 ? scoreSum / scoreCount : 0, by_grader: byGraderSummary };
}

export function createResultSet(suite: EvalSuite, results: Result[], runId: string, runnerName = "openeval-sdk", runnerVersion = "1.0.0"): ResultSet {
  return { version: OPENEVAL_VERSION, suite_id: suite.id, suite_version: suite.version, run_id: runId, started_at: new Date().toISOString(), completed_at: new Date().toISOString(), provider: suite.config?.provider, runner: { name: runnerName, version: runnerVersion }, results, summary: computeSummary(results) };
}
