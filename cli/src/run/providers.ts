import type { CompletionResult, EmbeddingResult, ProviderClient, ProviderName, ResolvedProviderConfig } from "./types";

export const DEFAULT_API_BASE: Record<ProviderName, string> = {
  openai: "https://api.openai.com/v1",
  anthropic: "https://api.anthropic.com/v1",
};

export const DEFAULT_API_KEY_ENV: Record<ProviderName, string> = {
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
};

export const ANTHROPIC_VERSION = "2023-06-01";

export class MissingApiKeyError extends Error {
  constructor(public readonly envVar: string, public readonly provider: ProviderName) {
    super(`No API key found in $${envVar} for provider "${provider}". Set it before running, or pass --api-key-env <VAR> to point at a different environment variable.`);
    this.name = "MissingApiKeyError";
  }
}

export class ProviderHttpError extends Error {
  constructor(public readonly status: number, public readonly body: string, public readonly retryable: boolean) {
    super(`Provider request failed with HTTP ${status}: ${body.slice(0, 500)}`);
    this.name = "ProviderHttpError";
  }
}

function requireApiKey(envVar: string, provider: ProviderName): string {
  const key = process.env[envVar];
  if (!key) throw new MissingApiKeyError(envVar, provider);
  return key;
}

/** 429 and 5xx are worth retrying; everything else (bad request, auth, etc.)
 * is not — retrying a 400 just burns the same error three times. */
function isRetryableStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

/** Real HTTP client for the OpenAI chat-completions + embeddings APIs, and
 * for any OpenAI-compatible endpoint reachable via --api-base (local
 * inference servers, proxies, other vendors that mirror the OpenAI shape). */
export class OpenAICompatibleClient implements ProviderClient {
  constructor(private readonly apiBase: string, private readonly apiKeyEnv: string) {}

  async complete(args: { model: string; prompt: string; temperature?: number; maxTokens?: number }): Promise<CompletionResult> {
    const apiKey = requireApiKey(this.apiKeyEnv, "openai");
    const res = await fetch(`${this.apiBase}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({
        model: args.model,
        messages: [{ role: "user", content: args.prompt }],
        temperature: args.temperature,
        max_tokens: args.maxTokens,
      }),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new ProviderHttpError(res.status, body, isRetryableStatus(res.status));
    }
    const json = (await res.json()) as {
      choices: { message: { content: string } }[];
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    return {
      text: json.choices[0]?.message?.content ?? "",
      inputTokens: json.usage?.prompt_tokens ?? 0,
      outputTokens: json.usage?.completion_tokens ?? 0,
    };
  }

  async embed(args: { model: string; input: string }): Promise<EmbeddingResult> {
    const apiKey = requireApiKey(this.apiKeyEnv, "openai");
    const res = await fetch(`${this.apiBase}/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model: args.model, input: args.input }),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new ProviderHttpError(res.status, body, isRetryableStatus(res.status));
    }
    const json = (await res.json()) as { data: { embedding: number[] }[]; usage?: { prompt_tokens?: number } };
    return { vector: json.data[0]?.embedding ?? [], inputTokens: json.usage?.prompt_tokens ?? 0 };
  }
}

/** Real HTTP client for the Anthropic Messages API. Anthropic has no public
 * embeddings endpoint, so `embed()` throws rather than silently degrading —
 * semantic_similarity always routes through an OpenAI-compatible embedding
 * endpoint regardless of which provider generates completions (see
 * resolveEmbeddingClient in runner.ts). */
export class AnthropicClient implements ProviderClient {
  constructor(private readonly apiBase: string, private readonly apiKeyEnv: string) {}

  async complete(args: { model: string; prompt: string; temperature?: number; maxTokens?: number }): Promise<CompletionResult> {
    const apiKey = requireApiKey(this.apiKeyEnv, "anthropic");
    const res = await fetch(`${this.apiBase}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": apiKey, "anthropic-version": ANTHROPIC_VERSION },
      body: JSON.stringify({
        model: args.model,
        max_tokens: args.maxTokens ?? 1024,
        temperature: args.temperature,
        messages: [{ role: "user", content: args.prompt }],
      }),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new ProviderHttpError(res.status, body, isRetryableStatus(res.status));
    }
    const json = (await res.json()) as {
      content: { type: string; text?: string }[];
      usage?: { input_tokens?: number; output_tokens?: number };
    };
    const text = json.content?.find((c) => c.type === "text")?.text ?? "";
    return { text, inputTokens: json.usage?.input_tokens ?? 0, outputTokens: json.usage?.output_tokens ?? 0 };
  }

  async embed(): Promise<EmbeddingResult> {
    throw new Error("Anthropic has no public embeddings API. Use an OpenAI-compatible embedding endpoint for semantic_similarity graders (the default, or set --embedding-api-base).");
  }
}

export function createProviderClient(provider: ProviderName, apiBase: string, apiKeyEnv: string): ProviderClient {
  return provider === "anthropic" ? new AnthropicClient(apiBase, apiKeyEnv) : new OpenAICompatibleClient(apiBase, apiKeyEnv);
}

export function resolveProviderConfig(args: {
  provider: ProviderName;
  model: string;
  apiBase?: string;
  apiKeyEnv?: string;
  temperature?: number;
  maxTokens?: number;
}): ResolvedProviderConfig {
  return {
    provider: args.provider,
    model: args.model,
    apiBase: args.apiBase ?? DEFAULT_API_BASE[args.provider],
    apiKeyEnv: args.apiKeyEnv ?? DEFAULT_API_KEY_ENV[args.provider],
    temperature: args.temperature,
    maxTokens: args.maxTokens,
  };
}
