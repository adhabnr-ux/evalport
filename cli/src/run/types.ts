// Internal types for the `evalport run` command. These are runner
// implementation details, not part of the EvalPort spec itself — the spec
// types (EvalSuite, TestCase, Grader, Result, ResultSet, ...) live in
// ../../../sdk/typescript/src/types.ts and are what actually gets written
// to disk / validated.

export type ProviderName = "openai" | "anthropic";

/** Resolved, ready-to-use provider configuration for a single run. */
export interface ResolvedProviderConfig {
  provider: ProviderName;
  model: string;
  apiBase: string;
  apiKeyEnv: string;
  temperature?: number;
  maxTokens?: number;
}

/** What a provider call actually returns, independent of vendor shape. */
export interface CompletionResult {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

export interface EmbeddingResult {
  vector: number[];
  inputTokens: number;
}

/** The seam mocked in tests — no live HTTP, no live API keys required. */
export interface ProviderClient {
  complete(args: { model: string; prompt: string; temperature?: number; maxTokens?: number }): Promise<CompletionResult>;
  embed(args: { model: string; input: string }): Promise<EmbeddingResult>;
}

export interface RunOptions {
  suitePath: string;
  provider: ProviderName;
  model: string;
  apiBase?: string;
  apiKeyEnv?: string;
  temperature?: number;
  maxTokens?: number;
  parallel: number;
  limit?: number;
  output?: string;
  dryRun: boolean;
  maxAttempts: number;
  backoffMs: number;
  embeddingApiBase?: string;
  embeddingApiKeyEnv?: string;
  embeddingModel?: string;
  runId?: string;
}

export interface GraderOutcome {
  graderId: string;
  type: string;
  score: number | null;
  passed: boolean;
  reason?: string;
  metadata?: Record<string, unknown>;
  costUsd?: number;
  inputTokens?: number;
  outputTokens?: number;
}

/** Per-model pricing, USD per 1M tokens. Used only for --dry-run estimates
 * and for populating result cost metadata — never sent anywhere. */
export interface ModelPricing {
  inputPer1M: number;
  outputPer1M: number;
}
