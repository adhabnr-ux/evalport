import type { EvalSuite, Grader, TestCase } from "../../../../sdk/typescript/src/types";
import type { GraderOutcome, ProviderClient } from "../types";
import { TIER1_TYPES, gradeTier1 } from "./tier1";
import { TIER2_TYPES, gradeLlmJudge, gradeSemanticSimilarity } from "./tier2";

export { TIER1_TYPES } from "./tier1";
export { TIER2_TYPES } from "./tier2";

/** Types with no runner support in this version. Per SPEC.md's "Custom
 * grader handling" rule: never fail the suite, record score:null,
 * passed:false, metadata.skip_reason:"unsupported_grader_type" instead. */
export const UNSUPPORTED_TYPES = new Set(["code", "human", "custom"]);

/** Resolve a test case's grader list (string refs into suite.graders, or
 * inline dicts) into concrete Grader objects. Suites are validated with
 * validateSuite() before a run starts, so a dangling string reference here
 * should never happen in practice — but a suite loaded with
 * validation skipped (or hand-edited after validation) could still produce
 * one, so unresolvable refs are dropped rather than crashing the run. */
export function resolveGraders(suite: EvalSuite, testCase: TestCase): Grader[] {
  const suiteGraders = new Map((suite.graders ?? []).map((g) => [g.id, g] as const));
  const resolved: Grader[] = [];
  for (const ref of testCase.graders ?? []) {
    if (typeof ref === "string") {
      const found = suiteGraders.get(ref);
      if (found) resolved.push(found);
    } else {
      resolved.push(ref);
    }
  }
  return resolved;
}

export interface GradeContext {
  actualOutput: string;
  input: string;
  expectedOutput?: string;
  context?: string[];
}

export interface GraderClients {
  /** Used for llm_judge / model graded completions. */
  judgeClient: ProviderClient;
  /** Used for semantic_similarity embeddings — always OpenAI-compatible,
   * since Anthropic has no public embeddings API (see providers.ts). */
  embeddingClient: ProviderClient;
  defaultEmbeddingModel: string;
}

function skipOutcome(grader: Grader): GraderOutcome {
  return {
    graderId: grader.id,
    type: grader.type,
    score: null,
    passed: false,
    reason: `grader type "${grader.type}" is not supported by evalport run — skipped cleanly per spec (see SPEC.md "Custom grader handling")`,
    metadata: { skip_reason: "unsupported_grader_type" },
  };
}

export async function gradeOne(grader: Grader, ctx: GradeContext, clients: GraderClients): Promise<GraderOutcome> {
  if (TIER1_TYPES.has(grader.type)) {
    return gradeTier1(grader, ctx.actualOutput, ctx.expectedOutput);
  }
  if (TIER2_TYPES.has(grader.type)) {
    if (grader.type === "semantic_similarity") {
      return gradeSemanticSimilarity(grader, { actualOutput: ctx.actualOutput, expectedOutput: ctx.expectedOutput }, clients.embeddingClient, clients.defaultEmbeddingModel);
    }
    // llm_judge and its "model graded" alias
    return gradeLlmJudge(grader, { actualOutput: ctx.actualOutput, input: ctx.input, expectedOutput: ctx.expectedOutput, context: ctx.context }, clients.judgeClient);
  }
  // code / human / custom / anything unrecognized
  return skipOutcome(grader);
}
