import { describe, expect, test } from "vitest";
import { KNOWN_PRICING, estimateCostUsd, estimateTokens, pricingFor } from "../../src/run/cost";

describe("pricingFor", () => {
  test("exact match returns the known pricing entry", () => {
    const { pricing, isKnown } = pricingFor("gpt-4o");
    expect(isKnown).toBe(true);
    expect(pricing).toEqual(KNOWN_PRICING["gpt-4o"]);
  });

  test("dated model id prefix-matches to the base model", () => {
    const { pricing, isKnown } = pricingFor("gpt-4o-2024-08-06");
    expect(isKnown).toBe(true);
    expect(pricing).toEqual(KNOWN_PRICING["gpt-4o"]);
  });

  test("unknown model falls back to a conservative rate, not $0", () => {
    const { pricing, isKnown } = pricingFor("some-brand-new-model-nobody-has-priced-yet");
    expect(isKnown).toBe(false);
    expect(pricing.inputPer1M).toBeGreaterThan(0);
    expect(pricing.outputPer1M).toBeGreaterThan(0);
  });
});

describe("estimateCostUsd", () => {
  test("computes cost from known per-1M pricing", () => {
    // gpt-4o-mini: 0.15 in / 0.6 out per 1M
    const cost = estimateCostUsd("gpt-4o-mini", 1_000_000, 1_000_000);
    expect(cost).toBeCloseTo(0.15 + 0.6, 6);
  });

  test("zero tokens costs zero", () => {
    expect(estimateCostUsd("gpt-4o", 0, 0)).toBe(0);
  });

  test("embedding models have zero output cost by design", () => {
    const cost = estimateCostUsd("text-embedding-3-small", 1_000_000, 1_000_000);
    expect(cost).toBeCloseTo(0.02, 6); // output tokens shouldn't matter for an embedding model
  });
});

describe("estimateTokens", () => {
  test("uses the ~4 chars/token heuristic", () => {
    expect(estimateTokens("a".repeat(400))).toBe(100);
  });

  test("empty string is zero tokens", () => {
    expect(estimateTokens("")).toBe(0);
  });

  test("rounds up for partial tokens", () => {
    expect(estimateTokens("abc")).toBe(1);
  });
});
