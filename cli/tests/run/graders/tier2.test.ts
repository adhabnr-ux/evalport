import { describe, expect, test, vi } from "vitest";
import { TIER2_TYPES, gradeLlmJudge, gradeSemanticSimilarity } from "../../../src/run/graders/tier2";
import type { Grader } from "../../../../sdk/typescript/src/types";
import type { CompletionResult, EmbeddingResult, ProviderClient } from "../../../src/run/types";

function grader(type: Grader["type"], params: Record<string, unknown> = {}, id = "g1"): Grader {
  return { id, type, params };
}

function mockClient(overrides: Partial<ProviderClient> = {}): ProviderClient {
  return {
    complete: vi.fn(async (): Promise<CompletionResult> => ({ text: "", inputTokens: 0, outputTokens: 0 })),
    embed: vi.fn(async (): Promise<EmbeddingResult> => ({ vector: [], inputTokens: 0 })),
    ...overrides,
  };
}

describe("TIER2_TYPES", () => {
  test("includes llm_judge, its 'model graded' alias, and semantic_similarity", () => {
    expect(TIER2_TYPES.has("llm_judge")).toBe(true);
    expect(TIER2_TYPES.has("model graded")).toBe(true);
    expect(TIER2_TYPES.has("semantic_similarity")).toBe(true);
    expect(TIER2_TYPES.has("exact_match")).toBe(false);
  });
});

describe("gradeLlmJudge", () => {
  const g = grader("llm_judge", { model: "gpt-4o-mini", prompt: "Score {output} vs {expected} given {input} and {context}.", threshold: 0.7 });

  test("parses a clean JSON {score, reason} response", async () => {
    const client = mockClient({ complete: vi.fn(async () => ({ text: '{"score": 0.9, "reason": "great"}', inputTokens: 10, outputTokens: 5 })) });
    const outcome = await gradeLlmJudge(g, { actualOutput: "out", input: "in", expectedOutput: "exp" }, client);
    expect(outcome.score).toBe(0.9);
    expect(outcome.passed).toBe(true);
    expect(outcome.reason).toBe("great");
    expect(outcome.inputTokens).toBe(10);
    expect(outcome.outputTokens).toBe(5);
    expect(typeof outcome.costUsd).toBe("number");
  });

  test("interpolates the prompt template with output/input/expected/context", async () => {
    const complete = vi.fn(async (_args: { model: string; prompt: string }) => ({ text: '{"score": 1}', inputTokens: 1, outputTokens: 1 }));
    const client = mockClient({ complete });
    await gradeLlmJudge(g, { actualOutput: "OUT", input: "IN", expectedOutput: "EXP", context: ["c1", "c2"] }, client);
    const sentPrompt = complete.mock.calls[0][0].prompt;
    expect(sentPrompt).toContain("OUT");
    expect(sentPrompt).toContain("IN");
    expect(sentPrompt).toContain("EXP");
    expect(sentPrompt).toContain("c1\nc2");
  });

  test("extracts JSON embedded in a larger text response", async () => {
    const client = mockClient({ complete: vi.fn(async () => ({ text: 'Sure, here you go: {"score": 0.8} thanks!', inputTokens: 1, outputTokens: 1 })) });
    const outcome = await gradeLlmJudge(g, { actualOutput: "x", input: "y" }, client);
    expect(outcome.score).toBe(0.8);
  });

  test("falls back to a bare numeric response", async () => {
    const client = mockClient({ complete: vi.fn(async () => ({ text: "0.85", inputTokens: 1, outputTokens: 1 })) });
    const outcome = await gradeLlmJudge(g, { actualOutput: "x", input: "y" }, client);
    expect(outcome.score).toBe(0.85);
  });

  test("falls back to PASS/FAIL keyword parsing", async () => {
    const client = mockClient({ complete: vi.fn(async () => ({ text: "This is a PASS.", inputTokens: 1, outputTokens: 1 })) });
    const pass = await gradeLlmJudge(g, { actualOutput: "x", input: "y" }, client);
    expect(pass.score).toBe(1);

    const client2 = mockClient({ complete: vi.fn(async () => ({ text: "FAIL — wrong answer.", inputTokens: 1, outputTokens: 1 })) });
    const fail = await gradeLlmJudge(g, { actualOutput: "x", input: "y" }, client2);
    expect(fail.score).toBe(0);
  });

  test("unparseable garbage scores 0 with a diagnostic reason, never throws", async () => {
    const client = mockClient({ complete: vi.fn(async () => ({ text: "¯\\_(ツ)_/¯", inputTokens: 1, outputTokens: 1 })) });
    const outcome = await gradeLlmJudge(g, { actualOutput: "x", input: "y" }, client);
    expect(outcome.score).toBe(0);
    expect(outcome.reason).toMatch(/could not parse/);
  });

  test("score is clamped into [0,1] even if the judge returns out-of-range numbers", async () => {
    const client = mockClient({ complete: vi.fn(async () => ({ text: '{"score": 5}', inputTokens: 1, outputTokens: 1 })) });
    const outcome = await gradeLlmJudge(g, { actualOutput: "x", input: "y" }, client);
    expect(outcome.score).toBe(1);
  });

  test("passed compares the raw (unclamped) score against threshold, not the clamped score", async () => {
    // threshold 0.7; raw score -1 should fail even though clamped score is 0.
    const client = mockClient({ complete: vi.fn(async () => ({ text: '{"score": -1}', inputTokens: 1, outputTokens: 1 })) });
    const outcome = await gradeLlmJudge(g, { actualOutput: "x", input: "y" }, client);
    expect(outcome.passed).toBe(false);
  });

  test("default threshold is 1.0 when not specified", async () => {
    const gNoThreshold = grader("llm_judge", { model: "gpt-4o-mini", prompt: "{output}" });
    const client = mockClient({ complete: vi.fn(async () => ({ text: '{"score": 0.99}', inputTokens: 1, outputTokens: 1 })) });
    const outcome = await gradeLlmJudge(gNoThreshold, { actualOutput: "x", input: "y" }, client);
    expect(outcome.passed).toBe(false);
  });
});

describe("gradeSemanticSimilarity", () => {
  const g = grader("semantic_similarity", { threshold: 0.8 });

  test("cosine similarity of identical vectors is 1 and passes", async () => {
    const client = mockClient({ embed: vi.fn(async () => ({ vector: [1, 0, 0], inputTokens: 3 })) });
    const outcome = await gradeSemanticSimilarity(g, { actualOutput: "a", expectedOutput: "a" }, client, "text-embedding-3-small");
    expect(outcome.score).toBeCloseTo(1, 5);
    expect(outcome.passed).toBe(true);
  });

  test("orthogonal vectors score 0 and fail", async () => {
    let call = 0;
    const client = mockClient({
      embed: vi.fn(async () => {
        call++;
        return { vector: call === 1 ? [1, 0] : [0, 1], inputTokens: 2 };
      }),
    });
    const outcome = await gradeSemanticSimilarity(g, { actualOutput: "a", expectedOutput: "b" }, client, "text-embedding-3-small");
    expect(outcome.score).toBeCloseTo(0, 5);
    expect(outcome.passed).toBe(false);
  });

  test("missing expected_output skips cleanly instead of calling the embedding API", async () => {
    const embed = vi.fn(async () => ({ vector: [1], inputTokens: 1 }));
    const client = mockClient({ embed });
    const outcome = await gradeSemanticSimilarity(g, { actualOutput: "a", expectedOutput: undefined }, client, "text-embedding-3-small");
    expect(outcome.score).toBeNull();
    expect(outcome.passed).toBe(false);
    expect(outcome.metadata?.skip_reason).toBe("missing_expected_output");
    expect(embed).not.toHaveBeenCalled();
  });

  test("uses grader params.model over the default embedding model when given", async () => {
    const embed = vi.fn(async (_args: { model: string; input: string }) => ({ vector: [1, 0], inputTokens: 1 }));
    const client = mockClient({ embed });
    await gradeSemanticSimilarity(grader("semantic_similarity", { model: "custom-embed" }), { actualOutput: "a", expectedOutput: "b" }, client, "text-embedding-3-small");
    expect(embed.mock.calls[0][0].model).toBe("custom-embed");
  });

  test("default threshold is 0.8 when not specified", async () => {
    const gNoThreshold = grader("semantic_similarity");
    let call = 0;
    // similarity ~0.6, below default 0.8
    const client = mockClient({
      embed: vi.fn(async () => {
        call++;
        return { vector: call === 1 ? [1, 0] : [0.6, 0.8], inputTokens: 1 };
      }),
    });
    const outcome = await gradeSemanticSimilarity(gNoThreshold, { actualOutput: "a", expectedOutput: "b" }, client, "text-embedding-3-small");
    expect(outcome.passed).toBe(false);
  });
});
