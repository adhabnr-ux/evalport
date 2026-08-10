import type { ModelPricing } from "./types";

// USD per 1M tokens. Not exhaustive, not guaranteed current — pricing pages
// change without notice and this table WILL drift. Treat --dry-run numbers
// as an estimate to sanity-check spend, not an invoice. Unknown models fall
// back to a conservative default rather than silently estimating $0.
export const KNOWN_PRICING: Record<string, ModelPricing> = {
  "gpt-4o": { inputPer1M: 2.5, outputPer1M: 10 },
  "gpt-4o-mini": { inputPer1M: 0.15, outputPer1M: 0.6 },
  "gpt-4-turbo": { inputPer1M: 10, outputPer1M: 30 },
  "gpt-3.5-turbo": { inputPer1M: 0.5, outputPer1M: 1.5 },
  "text-embedding-3-small": { inputPer1M: 0.02, outputPer1M: 0 },
  "text-embedding-3-large": { inputPer1M: 0.13, outputPer1M: 0 },
  "claude-3-5-sonnet-20241022": { inputPer1M: 3, outputPer1M: 15 },
  "claude-3-5-haiku-20241022": { inputPer1M: 0.8, outputPer1M: 4 },
  "claude-3-opus-20240229": { inputPer1M: 15, outputPer1M: 75 },
};

// Any model prefix not in KNOWN_PRICING gets this — deliberately at the
// higher end of observed frontier-model pricing so --dry-run never
// under-promises spend for a model we don't have real numbers for.
const FALLBACK_PRICING: ModelPricing = { inputPer1M: 10, outputPer1M: 30 };

export function pricingFor(model: string): { pricing: ModelPricing; isKnown: boolean } {
  if (KNOWN_PRICING[model]) return { pricing: KNOWN_PRICING[model], isKnown: true };
  // Loose prefix match for dated model ids, e.g. "gpt-4o-2024-08-06" -> "gpt-4o".
  const prefixMatch = Object.keys(KNOWN_PRICING).find((k) => model.startsWith(k));
  if (prefixMatch) return { pricing: KNOWN_PRICING[prefixMatch], isKnown: true };
  return { pricing: FALLBACK_PRICING, isKnown: false };
}

export function estimateCostUsd(model: string, inputTokens: number, outputTokens: number): number {
  const { pricing } = pricingFor(model);
  return (inputTokens / 1_000_000) * pricing.inputPer1M + (outputTokens / 1_000_000) * pricing.outputPer1M;
}

/** Rough token estimate for cost planning before any call is made. Not a
 * real tokenizer — ~4 chars/token is the standard back-of-envelope ratio
 * for English text and is good enough for a --dry-run budget check. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export interface DryRunLineItem {
  testCaseId: string;
  graderIds: string[];
  estimatedInputTokens: number;
  estimatedOutputTokens: number;
  estimatedCostUsd: number;
}

export interface DryRunReport {
  model: string;
  modelPricingKnown: boolean;
  testCaseCount: number;
  lineItems: DryRunLineItem[];
  totalEstimatedInputTokens: number;
  totalEstimatedOutputTokens: number;
  totalEstimatedCostUsd: number;
  judgeModelsUsed: string[];
  warnings: string[];
}
