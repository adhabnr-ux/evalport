# @evalport/cli

Command-line tool for EvalPort — run eval suites against real providers, validate, convert, and init.

## Install

```bash
npm install -g @evalport/cli
```

## `run` — execute a suite against a real provider

This is the CLI's headline command: it loads an EvalPort suite, calls a real model provider for every test case, grades the outputs, and writes a spec-valid, self-validated `ResultSet`.

```bash
openeval run <suite.json> --provider <openai|anthropic> [options]
```

**Always estimate cost before spending anything:**

```bash
openeval run suite.json --provider openai --model gpt-4o-mini --dry-run
```

`--dry-run` makes zero API calls. It prints a per-test-case token/cost estimate and a total, using the built-in pricing table (falling back to a conservative rate — deliberately on the high side — for unlisted models, with a warning). Treat the number as a budget sanity check, not an invoice: it's a heuristic (`chars/4` token approximation, an assumed completion length), not a real tokenizer.

Once you've reviewed the estimate and are ready to spend real money:

```bash
openeval run suite.json --provider openai --model gpt-4o-mini --output results.json
```

### Providers

Two providers ship built in:

- `--provider openai` — the OpenAI chat-completions and embeddings APIs, or **any OpenAI-compatible endpoint** via `--api-base` (local inference servers like Ollama/vLLM, proxies, other vendors that mirror the OpenAI request/response shape).
- `--provider anthropic` — the Anthropic Messages API. Anthropic has no public embeddings API, so `semantic_similarity` graders always call an OpenAI-compatible embedding endpoint regardless of `--provider` (configurable via `--embedding-api-base` / `--embedding-api-key-env` / `--embedding-model`, defaulting to OpenAI's `text-embedding-3-small`).

API keys are read from an environment variable — `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` by default, or whatever `--api-key-env` points at. Keys are never logged, written to output, or passed on the command line.

### Graders

| Tier | Types | Notes |
|---|---|---|
| 1 (local, zero external deps) | `exact_match`, `contains`, `regex`, `json_schema`, `json_path` | Run entirely in-process — no network call, no API key needed, work offline and in CI for free. `json_schema` is a hand-written draft-07-ish validator; `json_path` is a hand-written evaluator covering `$`, `.field`, `[index]`/`[-1]`, and `[*]`/`.*` wildcards. Neither pulls in `ajv` or a JSONPath package. |
| 2 (API calls) | `llm_judge` / `model graded`, `semantic_similarity` | `llm_judge` sends your `{output}`/`{input}`/`{expected}`/`{context}` prompt template to a judge model and parses its response generously (JSON `{"score":...}`, a bare number, or PASS/FAIL text, in that order). `semantic_similarity` embeds the actual and expected outputs and compares cosine similarity against a threshold; it's skipped cleanly (not an error) when the test case has no `expected_output`. |
| Unsupported in this runner | `code`, `human`, `custom` | Per SPEC.md's "Custom grader handling" rule: recorded as `skipped` with `score: null`, `passed: false`, `metadata.skip_reason: "unsupported_grader_type"`. A run is never aborted because of a grader type it doesn't know how to execute. |

A grader implementation that throws unexpectedly is caught and recorded as a `GRADER_ERROR`-style skipped result (`metadata.error`), not a crashed run.

### Flags

```
Required:
  --provider <name>          "openai" or "anthropic" (or any OpenAI-compatible
                              endpoint via --api-base with --provider openai)

Model / provider:
  --model <name>              Falls back to config.provider.model in the suite if omitted
  --api-base <url>            Falls back to the provider's default endpoint
  --api-key-env <VAR>         Env var holding the API key (default: OPENAI_API_KEY / ANTHROPIC_API_KEY)
  --temperature <n>
  --max-tokens <n>

Execution:
  --parallel <n>               Concurrent test cases in flight (default: 1)
  --limit <n>                  Only run the first <n> test cases
  --output <path>               Write the ResultSet JSON here (also written incrementally as cases complete)
  --dry-run                     Estimate cost and exit — makes zero API calls
  --max-attempts <n>            Retry attempts for retryable (429/5xx) provider errors (default: 3)
  --backoff-ms <n>               Base backoff between retries, doubles each attempt (default: 1000)
  --run-id <id>                  Override the generated run_id

Semantic similarity embeddings (always OpenAI-compatible, independent of --provider):
  --embedding-api-base <url>
  --embedding-api-key-env <VAR>
  --embedding-model <name>       Default: text-embedding-3-small
```

A test case's own `provider` block (model / `api_base` / `api_key_env` / `temperature` / `max_tokens`) overrides the run-level configuration for that one test case, per the spec.

### Behavior notes

- **Suites are validated before anything runs.** A suite that fails `validateSuite()` is rejected with the exact validation errors — `evalport run` never executes against a suite the SDK itself considers invalid.
- **`test_cases_file` (JSONL) is supported** for large suites, resolved relative to the suite file, exactly as documented in [SPEC.md](../spec/SPEC.md#example-5-jsonl-streaming-format).
- **Retries are selective.** Only HTTP 429 and 5xx responses are retried, with exponential backoff (`--backoff-ms`, doubling each attempt up to `--max-attempts`). Missing API keys, bad requests, and auth errors fail immediately — retrying them would just repeat the same error.
- **`timeout_ms`** (per test case, or `config.defaults.timeout_ms` at the suite level) is enforced; an exceeded call is recorded with `error.type: "timeout"`.
- **Results are written incrementally** to `--output` as each test case completes, so a long run's progress survives an interruption instead of being lost until the very end.
- **The final `ResultSet` is self-validated** against the SDK's own `validateResultSet()` before being written. If the runner ever produced spec-invalid output, it refuses to ship it — that's treated as a runner bug, not something to paper over.
- Exit code is `1` if any test case failed (or the run itself errored), `0` otherwise — safe to use in CI gating.

## Other commands

### validate

Validate an EvalPort document against its schema.

```bash
openeval validate my-suite.json
openeval validate my-suite.json --type=resultset
```

### convert

Convert between evaluation formats.

```bash
openeval convert promptfoo openeval config.json output.json
```

Supported conversions:
- `promptfoo` → `openeval`

### init

Create a starter eval suite.

```bash
openeval init my-eval-suite
# Creates my-eval-suite.json
```

### summary

Print summary statistics of a result set.

```bash
openeval summary results.json
```

## Development

```bash
npm install
npm run typecheck   # tsc --noEmit
npm test            # vitest — unit tests for Tier 1 graders, mocked-API tests for
                     # providers/Tier 2 graders/the full runner. No live API keys required or used.
```

## License

Apache 2.0
