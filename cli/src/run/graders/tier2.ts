import type { Grader } from "../../../../sdk/typescript/src/types";
import type { GraderOutcome, ProviderClient } from "../types";
import { estimateCostUsd } from "../cost";

// Tier 2 graders make real API calls (a judge LLM completion, or an
// embedding call for semantic similarity). Both take a ProviderClient so
// tests can inject a mock and never touch a live key or the network.

export const TIER2_TYPES = new Set(["llm_judge", "model graded", "semantic_similarity"]);

function interpolatePrompt(template: string, vars: { output: string; input: string; expected: string; context: string }): string {
  return template
    .replace(/\{output\}/g, vars.output)
    .replace(/\{input\}/g, vars.input)
    .replace(/\{expected\}/g, vars.expected)
    .replace(/\{context\}/g, vars.context);
}

/** Judge responses are supposed to be JSON like {"score": 0.8, "reason": "..."}
 * per the spec's example, but real models don't always comply — this parses
 * generously: JSON object with a score field, a bare number, or PASS/FAIL
 * text, in that order of preference. */
function parseJudgeResponse(text: string): { score: number; reason?: string } {
  const trimmed = text.trim();
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && typeof parsed.score === "number") {
      return { score: parsed.score, reason: typeof parsed.reason === "string" ? parsed.reason : undefined };
    }
  } catch {
    // not JSON — fall through to looser parsing below
  }
  const jsonBlockMatch = trimmed.match(/\{[\s\S]*\}/);
  if (jsonBlockMatch) {
    try {
      const parsed = JSON.parse(jsonBlockMatch[0]);
      if (parsed && typeof parsed === "object" && typeof parsed.score === "number") {
        return { score: parsed.score, reason: typeof parsed.reason === "string" ? parsed.reason : undefined };
      }
    } catch {
      // still not parseable JSON — fall through
    }
  }
  const bareNumber = trimmed.match(/-?\d+(\.\d+)?/);
  if (bareNumber && /^-?\d+(\.\d+)?$/.test(trimmed)) return { score: Number(bareNumber[0]) };
  if (/\bpass\b/i.test(trimmed)) return { score: 1, reason: trimmed };
  if (/\bfail\b/i.test(trimmed)) return { score: 0, reason: trimmed };
  return { score: 0, reason: `could not parse judge response: ${trimmed.slice(0, 200)}` };
}

export async function gradeLlmJudge(
  grader: Grader,
  args: { actualOutput: string; input: string; expectedOutput?: string; context?: string[] },
  client: ProviderClient,
): Promise<GraderOutcome> {
  const params = grader.params ?? {};
  const model = String(params.model ?? "");
  const promptTemplate = String(params.prompt ?? "");
  const temperature = typeof params.temperature === "number" ? params.temperature : 0;
  const threshold = typeof params.threshold === "number" ? params.threshold : 1.0;

  const prompt = interpolatePrompt(promptTemplate, {
    output: args.actualOutput,
    input: args.input,
    expected: args.expectedOutput ?? "",
    context: (args.context ?? []).join("\n"),
  });

  const completion = await client.complete({ model, prompt, temperature });
  const { score, reason } = parseJudgeResponse(completion.text);
  const clampedScore = Math.max(0, Math.min(1, score));
  const passed = score >= threshold;
  const costUsd = estimateCostUsd(model, completion.inputTokens, completion.outputTokens);

  return {
    graderId: grader.id,
    type: grader.type,
    score: clampedScore,
    passed,
    reason: reason ?? `judge score ${score} vs threshold ${threshold}`,
    metadata: { judge_model: model, judge_raw_response: completion.text },
    costUsd,
    inputTokens: completion.inputTokens,
    outputTokens: completion.outputTokens,
  };
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i]; }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function gradeSemanticSimilarity(
  grader: Grader,
  args: { actualOutput: string; expectedOutput?: string },
  embeddingClient: ProviderClient,
  defaultModel: string,
): Promise<GraderOutcome> {
  const params = grader.params ?? {};
  const threshold = typeof params.threshold === "number" ? params.threshold : 0.8;
  const model = typeof params.model === "string" ? params.model : defaultModel;

  if (!args.expectedOutput) {
    return { graderId: grader.id, type: grader.type, score: null, passed: false, reason: "semantic_similarity requires expected_output on the test case", metadata: { skip_reason: "missing_expected_output" } };
  }

  const [actualEmb, expectedEmb] = await Promise.all([
    embeddingClient.embed({ model, input: args.actualOutput }),
    embeddingClient.embed({ model, input: args.expectedOutput }),
  ]);
  const similarity = cosineSimilarity(actualEmb.vector, expectedEmb.vector);
  const passed = similarity >= threshold;
  const totalInputTokens = actualEmb.inputTokens + expectedEmb.inputTokens;
  const costUsd = estimateCostUsd(model, totalInputTokens, 0);

  return {
    graderId: grader.id,
    type: grader.type,
    score: similarity,
    passed,
    reason: `cosine similarity ${similarity.toFixed(4)} vs threshold ${threshold}`,
    metadata: { embedding_model: model },
    costUsd,
    inputTokens: totalInputTokens,
    outputTokens: 0,
  };
}
