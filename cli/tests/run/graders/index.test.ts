import { describe, expect, test, vi } from "vitest";
import { UNSUPPORTED_TYPES, gradeOne, resolveGraders } from "../../../src/run/graders/index";
import type { EvalSuite, Grader, TestCase } from "../../../../sdk/typescript/src/types";
import type { CompletionResult, EmbeddingResult, ProviderClient } from "../../../src/run/types";

function mockClient(overrides: Partial<ProviderClient> = {}): ProviderClient {
  return {
    complete: vi.fn(async (): Promise<CompletionResult> => ({ text: "", inputTokens: 0, outputTokens: 0 })),
    embed: vi.fn(async (): Promise<EmbeddingResult> => ({ vector: [], inputTokens: 0 })),
    ...overrides,
  };
}

describe("resolveGraders", () => {
  const suiteGraders: Grader[] = [
    { id: "gr_a", type: "exact_match" },
    { id: "gr_b", type: "contains", params: { substring: "x" } },
  ];
  const suite: EvalSuite = { version: "1.0.0", id: "s", graders: suiteGraders, test_cases: [] };

  test("resolves string references against suite.graders", () => {
    const tc: TestCase = { id: "tc1", input: "hi", graders: ["gr_a", "gr_b"] };
    const resolved = resolveGraders(suite, tc);
    expect(resolved.map((g) => g.id)).toEqual(["gr_a", "gr_b"]);
  });

  test("passes through inline grader objects untouched", () => {
    const inline: Grader = { id: "gr_inline", type: "regex", params: { pattern: "x" } };
    const tc: TestCase = { id: "tc1", input: "hi", graders: [inline] };
    expect(resolveGraders(suite, tc)).toEqual([inline]);
  });

  test("supports mixing string refs and inline graders", () => {
    const inline: Grader = { id: "gr_inline", type: "regex", params: { pattern: "x" } };
    const tc: TestCase = { id: "tc1", input: "hi", graders: ["gr_a", inline] };
    expect(resolveGraders(suite, tc).map((g) => g.id)).toEqual(["gr_a", "gr_inline"]);
  });

  test("drops an unresolvable string reference instead of throwing", () => {
    const tc: TestCase = { id: "tc1", input: "hi", graders: ["gr_does_not_exist"] };
    expect(resolveGraders(suite, tc)).toEqual([]);
  });

  test("empty graders list on the test case resolves to empty", () => {
    const tc: TestCase = { id: "tc1", input: "hi", graders: [] };
    expect(resolveGraders(suite, tc)).toEqual([]);
  });
});

describe("UNSUPPORTED_TYPES", () => {
  test("covers exactly code, human, custom", () => {
    expect([...UNSUPPORTED_TYPES].sort()).toEqual(["code", "custom", "human"]);
  });
});

describe("gradeOne dispatcher", () => {
  const clients = { judgeClient: mockClient(), embeddingClient: mockClient(), defaultEmbeddingModel: "text-embedding-3-small" };

  test("routes exact_match (tier 1) synchronously, using ctx.actualOutput/expectedOutput", async () => {
    const outcome = await gradeOne({ id: "g1", type: "exact_match" }, { actualOutput: "4", input: "2+2?", expectedOutput: "4" }, clients);
    expect(outcome.passed).toBe(true);
  });

  test("routes llm_judge (tier 2) through the injected judge client", async () => {
    const judgeClient = mockClient({ complete: vi.fn(async () => ({ text: '{"score": 1}', inputTokens: 1, outputTokens: 1 })) });
    const outcome = await gradeOne(
      { id: "g1", type: "llm_judge", params: { model: "gpt-4o-mini", prompt: "{output}" } },
      { actualOutput: "hi", input: "q" },
      { ...clients, judgeClient },
    );
    expect(outcome.passed).toBe(true);
    expect(judgeClient.complete).toHaveBeenCalledOnce();
  });

  test("routes 'model graded' alias through the same llm_judge path", async () => {
    const judgeClient = mockClient({ complete: vi.fn(async () => ({ text: '{"score": 1}', inputTokens: 1, outputTokens: 1 })) });
    const outcome = await gradeOne(
      { id: "g1", type: "model graded", params: { model: "gpt-4o-mini", prompt: "{output}" } },
      { actualOutput: "hi", input: "q" },
      { ...clients, judgeClient },
    );
    expect(outcome.passed).toBe(true);
  });

  test("routes semantic_similarity through the injected embedding client, not the judge client", async () => {
    const embeddingClient = mockClient({ embed: vi.fn(async () => ({ vector: [1, 0], inputTokens: 1 })) });
    const judgeClient = mockClient();
    await gradeOne({ id: "g1", type: "semantic_similarity" }, { actualOutput: "a", input: "q", expectedOutput: "a" }, { ...clients, judgeClient, embeddingClient });
    expect(embeddingClient.embed).toHaveBeenCalled();
    expect(judgeClient.complete).not.toHaveBeenCalled();
  });

  test("skips code/human/custom cleanly with skip_reason:unsupported_grader_type", async () => {
    for (const type of ["code", "human", "custom"] as const) {
      const outcome = await gradeOne({ id: "g1", type }, { actualOutput: "x", input: "y" }, clients);
      expect(outcome.score).toBeNull();
      expect(outcome.passed).toBe(false);
      expect(outcome.metadata?.skip_reason).toBe("unsupported_grader_type");
    }
  });
});
