import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  ANTHROPIC_VERSION,
  AnthropicClient,
  DEFAULT_API_BASE,
  DEFAULT_API_KEY_ENV,
  MissingApiKeyError,
  OpenAICompatibleClient,
  ProviderHttpError,
  createProviderClient,
  resolveProviderConfig,
} from "../../src/run/providers";
import type { ProviderClient } from "../../src/run/types";

function jsonResponse(body: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(body), json: async () => body } as Response;
}

describe("resolveProviderConfig", () => {
  test("fills in provider-specific defaults when not overridden", () => {
    const cfg = resolveProviderConfig({ provider: "openai", model: "gpt-4o-mini" });
    expect(cfg.apiBase).toBe(DEFAULT_API_BASE.openai);
    expect(cfg.apiKeyEnv).toBe(DEFAULT_API_KEY_ENV.openai);
  });

  test("honors explicit overrides", () => {
    const cfg = resolveProviderConfig({ provider: "anthropic", model: "claude-3-5-sonnet-20241022", apiBase: "http://localhost:1234/v1", apiKeyEnv: "MY_KEY", temperature: 0.2, maxTokens: 500 });
    expect(cfg.apiBase).toBe("http://localhost:1234/v1");
    expect(cfg.apiKeyEnv).toBe("MY_KEY");
    expect(cfg.temperature).toBe(0.2);
    expect(cfg.maxTokens).toBe(500);
  });
});

describe("createProviderClient", () => {
  test("returns an AnthropicClient for 'anthropic', OpenAICompatibleClient otherwise", () => {
    expect(createProviderClient("anthropic", "base", "KEY")).toBeInstanceOf(AnthropicClient);
    expect(createProviderClient("openai", "base", "KEY")).toBeInstanceOf(OpenAICompatibleClient);
  });
});

describe("OpenAICompatibleClient", () => {
  const ENV_VAR = "TEST_OPENAI_KEY";
  beforeEach(() => {
    process.env[ENV_VAR] = "sk-test";
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    delete process.env[ENV_VAR];
    vi.unstubAllGlobals();
  });

  test("complete() posts to /chat/completions and maps the response", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      jsonResponse({ choices: [{ message: { content: "hello" } }], usage: { prompt_tokens: 3, completion_tokens: 2 } }),
    );
    const client = new OpenAICompatibleClient("https://api.example.com/v1", ENV_VAR);
    const result = await client.complete({ model: "gpt-4o-mini", prompt: "hi" });
    expect(result).toEqual({ text: "hello", inputTokens: 3, outputTokens: 2 });
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("https://api.example.com/v1/chat/completions");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer sk-test");
  });

  test("embed() posts to /embeddings and maps the response", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce(jsonResponse({ data: [{ embedding: [1, 2, 3] }], usage: { prompt_tokens: 4 } }));
    const client = new OpenAICompatibleClient("https://api.example.com/v1", ENV_VAR);
    const result = await client.embed({ model: "text-embedding-3-small", input: "hi" });
    expect(result).toEqual({ vector: [1, 2, 3], inputTokens: 4 });
  });

  test("missing API key throws MissingApiKeyError before any fetch call", async () => {
    delete process.env[ENV_VAR];
    const client = new OpenAICompatibleClient("https://api.example.com/v1", ENV_VAR);
    await expect(client.complete({ model: "gpt-4o-mini", prompt: "hi" })).rejects.toThrow(MissingApiKeyError);
    expect(fetch).not.toHaveBeenCalled();
  });

  test("HTTP 429 maps to a retryable ProviderHttpError", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ ok: false, status: 429, text: async () => "rate limited" } as Response);
    const client = new OpenAICompatibleClient("https://api.example.com/v1", ENV_VAR);
    try {
      await client.complete({ model: "gpt-4o-mini", prompt: "hi" });
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(ProviderHttpError);
      expect((e as ProviderHttpError).retryable).toBe(true);
      expect((e as ProviderHttpError).status).toBe(429);
    }
  });

  test("HTTP 500 maps to a retryable ProviderHttpError", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ ok: false, status: 503, text: async () => "down" } as Response);
    const client = new OpenAICompatibleClient("https://api.example.com/v1", ENV_VAR);
    await expect(client.complete({ model: "gpt-4o-mini", prompt: "hi" })).rejects.toMatchObject({ retryable: true, status: 503 });
  });

  test("HTTP 400 maps to a non-retryable ProviderHttpError", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ ok: false, status: 400, text: async () => "bad request" } as Response);
    const client = new OpenAICompatibleClient("https://api.example.com/v1", ENV_VAR);
    await expect(client.complete({ model: "gpt-4o-mini", prompt: "hi" })).rejects.toMatchObject({ retryable: false, status: 400 });
  });

  test("HTTP 401 maps to a non-retryable ProviderHttpError", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ ok: false, status: 401, text: async () => "unauthorized" } as Response);
    const client = new OpenAICompatibleClient("https://api.example.com/v1", ENV_VAR);
    await expect(client.complete({ model: "gpt-4o-mini", prompt: "hi" })).rejects.toMatchObject({ retryable: false, status: 401 });
  });
});

describe("AnthropicClient", () => {
  const ENV_VAR = "TEST_ANTHROPIC_KEY";
  beforeEach(() => {
    process.env[ENV_VAR] = "sk-ant-test";
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    delete process.env[ENV_VAR];
    vi.unstubAllGlobals();
  });

  test("complete() posts to /messages with anthropic headers and maps the text block", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      jsonResponse({ content: [{ type: "text", text: "hi there" }], usage: { input_tokens: 5, output_tokens: 3 } }),
    );
    const client = new AnthropicClient("https://api.anthropic.com/v1", ENV_VAR);
    const result = await client.complete({ model: "claude-3-5-sonnet-20241022", prompt: "hi" });
    expect(result).toEqual({ text: "hi there", inputTokens: 5, outputTokens: 3 });
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("https://api.anthropic.com/v1/messages");
    const headers = init.headers as Record<string, string>;
    expect(headers["x-api-key"]).toBe("sk-ant-test");
    expect(headers["anthropic-version"]).toBe(ANTHROPIC_VERSION);
  });

  test("embed() always throws — Anthropic has no public embeddings API", async () => {
    const client: ProviderClient = new AnthropicClient("https://api.anthropic.com/v1", ENV_VAR);
    await expect(client.embed({ model: "x", input: "y" })).rejects.toThrow(/no public embeddings API/);
    expect(fetch).not.toHaveBeenCalled();
  });

  test("HTTP error from Anthropic maps to ProviderHttpError with correct retryable flag", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ ok: false, status: 529, text: async () => "overloaded" } as Response);
    const client = new AnthropicClient("https://api.anthropic.com/v1", ENV_VAR);
    await expect(client.complete({ model: "x", prompt: "y" })).rejects.toMatchObject({ retryable: true, status: 529 });
  });
});
