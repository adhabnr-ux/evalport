# EvalPort — The Open Evaluation Standard

[![CI](https://github.com/adhabnr-ux/evalport/actions/workflows/ci.yml/badge.svg)](https://github.com/adhabnr-ux/evalport/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalport-sdk.svg?label=PyPI)](https://pypi.org/project/evalport-sdk/)
[![npm](https://img.shields.io/npm/v/evalport-sdk.svg?label=npm)](https://www.npmjs.com/package/evalport-sdk)
[![Discussions](https://img.shields.io/github/discussions/adhabnr-ux/evalport?label=Discussions)](https://github.com/adhabnr-ux/evalport/discussions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Version:** 1.0.0-rc.5 | **License:** Apache 2.0

**Adopted by:** [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) — merged to the official community extensions list, [PR #4797](https://github.com/UKGovernmentBEIS/inspect_ai/pull/4797), August 2026. Approved with a clean CI run at [TruLens](https://github.com/truera/trulens) — [PR #2697](https://github.com/truera/trulens/pull/2697), a full `to_openeval()`/`from_openeval()` module, 17/17 tests passing, awaiting merge. 32 framework adapters shipped and independently tested — including [Agenta](adapters/agenta-openeval-adapter/) (built at the explicit request of the Agenta-AI maintainer, [Agenta-AI/agenta#6222](https://github.com/Agenta-AI/agenta/issues/6222)), [Parea AI](adapters/parea-openeval-adapter/) (a first-time external contributor's [PR #19](https://github.com/adhabnr-ux/evalport/pull/19), merged after review and an independent re-run of its test suite), [Hugging Face `evaluate`](adapters/huggingface-evaluate-openeval-adapter/), [EleutherAI `lm-evaluation-harness`](adapters/lm-eval-harness-openeval-adapter/), [Hugging Face `lighteval`](adapters/lighteval-openeval-adapter/) (the library behind the HF Open LLM Leaderboard), [OpenCompass](adapters/opencompass-openeval-adapter/), [FinanceBench](adapters/financebench-openeval-adapter/) (a real benchmark dataset, not a live SDK), [Athina](adapters/athina-openeval-adapter/), [DeepEval](adapters/deepeval-openeval-adapter/), and [Galileo](adapters/galileo-openeval-adapter/) — plus a Literal AI adapter in review right now from another new contributor ([PR #25](https://github.com/adhabnr-ux/evalport/pull/25)) — see [Framework Adapters](#framework-adapters) below.

EvalPort is an open specification for portable LLM evaluation test cases, graders, suites, and results. It enables evaluation datasets to be shared across frameworks (DeepEval, Promptfoo, Ragas, Inspect AI, LangSmith, Braintrust, OpenAI Evals, MLflow) without loss of semantic fidelity.

## Quick Start

```bash
# Install CLI
npm install -g evalport-cli

# Create an eval suite
openeval init my-eval-suite

# Validate
openeval validate my-eval-suite.json

# Convert from Promptfoo
openeval convert promptfoo openeval config.json output.json

# Run a suite against a real provider — always dry-run first to see estimated cost
openeval run examples/basic-suite.json --provider openai --model gpt-4o-mini --dry-run
openeval run examples/basic-suite.json --provider openai --model gpt-4o-mini --output results.json

# Python SDK
pip install evalport-sdk
```

```python
from openeval.validate import validate_suite

result = validate_suite({
    "version": "1.0.0",
    "id": "my_suite",
    "graders": [{"id": "gr1", "type": "exact_match"}],
    "test_cases": [{"id": "tc1", "input": "Hello", "expected_output": "Hi", "graders": ["gr1"]}]
})
print(result.valid)  # True
```

## Contribute

EvalPort is pre-1.0 and actively shaped by outside contributors — the fastest ways in, from least to most involved:

- **Have an opinion on an open design question?** [`spec/SPEC.md`](spec/SPEC.md#open-design-questions--rfc-topics-we-need-help-with) tracks what's genuinely unresolved. All five questions raised on that list so far — suite/result signing, a formal conformance test suite, resuming interrupted runs, whether `llm_judge` injection mitigations should be mandatory, and (most recently, [Discussion #22](https://github.com/adhabnr-ux/evalport/discussions/22)) how `ResultSet` should represent repeated-attempt evals (`num_repetitions`, epochs) — have gone through this exact process and landed as shipped spec changes; the table shows how each one resolved. There's no genuinely open item on the list right now, but the process is the same for the next one: no prior EvalPort contribution required, just a considered opinion.
- **Want to ship a framework adapter?** [Issue #6](https://github.com/adhabnr-ux/evalport/issues/6) is the map. [`adapters/autogen-openeval-adapter`](adapters/autogen-openeval-adapter/) is the reference shape to copy: `to_openeval()`, `from_openeval()`, tests against the real validator, a README.
- **Want to propose a spec change?** Open a Discussion titled `[Spec Change] <what and why>` — the full process (comment period, what "consensus" means, when the spec lead's sign-off is required) is in [`spec/SPEC.md`'s Governance section](spec/SPEC.md#governance).
- **How people actually become collaborators:** ship something real and tested, engage substantively, get invited — see [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for who's done that so far. It's not gatekept; a merged, tested PR is what clears the bar.

Full details in the [Contributing Guide](.github/CONTRIBUTING.md).

## Repository Structure

```
openeval/
├── spec/
│   ├── SPEC.md                # Full specification
│   ├── ADOPTION.md            # Adoption strategy
│   ├── CRITIQUE.md            # Self-critique and hostile review
│   ├── schemas/               # JSON Schemas (4 files)
│   ├── examples/              # Example suites and conversions
├── sdk/
│   ├── typescript/            # evalport-sdk (npm)
│   └── python/                # openeval (PyPI)
├── cli/                      # evalport-cli
├── api/                     # Example REST API server
├── docs/
│   ├── getting-started/       # 5-minute quickstart
│   ├── grader-reference/      # All 11 grader types
│   ├── migration-guides/      # Promptfoo, DeepEval, Inspect AI
│   ├── api/                   # REST API docs
│   ├── blog/                  # Launch posts
│   └── landing-page.html      # Landing page
├── examples/                  # Example eval suites
├── adapters/                  # Standalone to_openeval()/from_openeval() packages per framework
├── benchmarks/                # 14 public benchmarks converted to validated EvalPort suites
├── .github/                   # CI, CONTRIBUTING, issue templates
├── LICENSE                    # Apache 2.0
└── README.md                  # This file
```

## Run Evals

`evalport run` executes an EvalPort suite against a real model provider and produces a spec-valid, self-validated `ResultSet` — no separate harness, no glue code. It's the CLI's headline command:

```bash
openeval run suite.json --provider openai --model gpt-4o-mini --dry-run   # estimate cost first, spend nothing
openeval run suite.json --provider anthropic --model claude-3-5-sonnet-20241022 --output results.json
```

- **Two providers out of the box** — OpenAI and Anthropic — plus any OpenAI-compatible endpoint (local inference servers, proxies, other vendors) via `--api-base`.
- **Tier 1 graders run locally, with zero external dependencies**: `exact_match`, `contains`, `regex`, `json_schema` (a hand-written draft-07-ish validator), `json_path` (a hand-written JSONPath subset evaluator).
- **Tier 2 graders call an API**: `llm_judge` / `model graded`, and `semantic_similarity` (cosine similarity over embeddings — always via an OpenAI-compatible endpoint, independent of `--provider`, since Anthropic has no public embeddings API).
- **Unsupported grader types clean-skip**, per the spec's "Custom grader handling" rule (`code`, `human`, `custom` are recorded as `skipped` with `metadata.skip_reason: "unsupported_grader_type"` — the run never aborts because of one grader it doesn't know how to execute).
- **`--dry-run` estimates cost before spending anything** — token and USD estimates per test case and in total, with warnings when a model's pricing isn't in the known table. Always run this first and get sign-off on the estimate before running for real.
- **Retries with backoff** on retryable provider errors (HTTP 429/5xx); non-retryable errors (bad auth, malformed request) fail fast instead of repeating.
- **`--parallel <n>`** for concurrent test cases, **`--limit <n>`** to run a subset, **`--output <path>`** to write results incrementally as cases complete (so a long run's progress survives an interruption).
- Every `ResultSet` it produces is validated against the SDK's own `validateResultSet()` before being written — `evalport run` refuses to emit output that fails its own spec.

See [`cli/README.md`](cli/README.md) for the full flag reference.

## Grader Types

| Type | Description |
|------|-------------|
| `exact_match` | String equality |
| `contains` | Substring check |
| `regex` | Regex match |
| `semantic_similarity` | Embedding cosine similarity |
| `llm_judge` | LLM-as-judge with prompt template |
| `json_schema` | JSON Schema validation |
| `json_path` | JSONPath extraction + comparison |
| `code` | Custom grading function (sandboxed) |
| `human` | Human review |
| `model graded` | Alias for llm_judge (OpenAI Evals compat) |
| `custom` | Framework-specific handler |

## Converters

| From | Status |
|------|--------|
| Promptfoo → EvalPort | ✅ CLI + SDK |
| DeepEval → EvalPort | ✅ Python SDK |
| Inspect AI → EvalPort | ✅ Python SDK |
| OpenAI Evals → EvalPort | ✅ Python SDK |

## Benchmark Hub

[`benchmarks/`](benchmarks/) converts 14 well-known public benchmarks — GSM8K, ARC, BoolQ, HellaSwag, WinoGrande, CommonsenseQA, PIQA, TruthfulQA, MMLU, HumanEval, MBPP, SQuAD 2.0, DROP, and BIG-Bench Hard — into 22 individually-valid EvalPort suites (8,012 test cases total), every one of them passing the real `validate_suite()` validator in CI. Every benchmark's license was independently verified before inclusion; see [`benchmarks/LICENSES.md`](benchmarks/LICENSES.md) for the full attribution table. Start with [`benchmarks/README.md`](benchmarks/README.md) for the full index and quickstart.

## Framework Adapters

[`adapters/`](adapters/) has standalone `to_openeval()`/`from_openeval()` packages for converting real evaluation results to and from EvalPort, one per framework, each independently installable and tested against the real validator:

| Framework | Package |
|---|---|
| Agenta | [agenta-openeval-adapter](adapters/agenta-openeval-adapter/) |
| Parea AI | [parea-openeval-adapter](adapters/parea-openeval-adapter/) |
| AutoGen | [autogen-openeval-adapter](adapters/autogen-openeval-adapter/) |
| CrewAI | [crewai-openeval-adapter](adapters/crewai-openeval-adapter/) |
| Ragas | [ragas-openeval-adapter](adapters/ragas-openeval-adapter/) |
| LangSmith | [langsmith-openeval-adapter](adapters/langsmith-openeval-adapter/) |
| Braintrust | [braintrust-openeval-adapter](adapters/braintrust-openeval-adapter/) |
| MLflow | [mlflow-openeval-adapter](adapters/mlflow-openeval-adapter/) |
| Opik | [opik-openeval-adapter](adapters/opik-openeval-adapter/) |
| Arize Phoenix | [phoenix-openeval-adapter](adapters/phoenix-openeval-adapter/) |
| Weights & Biases Weave | [weave-openeval-adapter](adapters/weave-openeval-adapter/) |
| UpTrain | [uptrain-openeval-adapter](adapters/uptrain-openeval-adapter/) |
| Langfuse | [langfuse-openeval-adapter](adapters/langfuse-openeval-adapter/) |
| Giskard | [giskard-openeval-adapter](adapters/giskard-openeval-adapter/) |
| LlamaIndex | [llamaindex-openeval-adapter](adapters/llamaindex-openeval-adapter/) |
| Patronus AI | [patronus-openeval-adapter](adapters/patronus-openeval-adapter/) |
| Vertex AI | [vertexai-openeval-adapter](adapters/vertexai-openeval-adapter/) |
| DSPy | [dspy-openeval-adapter](adapters/dspy-openeval-adapter/) |
| Haystack | [haystack-openeval-adapter](adapters/haystack-openeval-adapter/) |
| Evidently | [evidently-openeval-adapter](adapters/evidently-openeval-adapter/) |
| Guardrails AI | [guardrails-openeval-adapter](adapters/guardrails-openeval-adapter/) |
| Argilla | [argilla-openeval-adapter](adapters/argilla-openeval-adapter/) |
| Azure AI Evaluation | [azure-ai-evaluation-openeval-adapter](adapters/azure-ai-evaluation-openeval-adapter/) |
| Arthur Bench | [arthur-bench-openeval-adapter](adapters/arthur-bench-openeval-adapter/) |
| Hugging Face `evaluate` | [huggingface-evaluate-openeval-adapter](adapters/huggingface-evaluate-openeval-adapter/) |
| EleutherAI `lm-evaluation-harness` | [lm-eval-harness-openeval-adapter](adapters/lm-eval-harness-openeval-adapter/) |
| Hugging Face `lighteval` | [lighteval-openeval-adapter](adapters/lighteval-openeval-adapter/) |
| OpenCompass | [opencompass-openeval-adapter](adapters/opencompass-openeval-adapter/) |
| FinanceBench | [financebench-openeval-adapter](adapters/financebench-openeval-adapter/) |
| Athina | [athina-openeval-adapter](adapters/athina-openeval-adapter/) |
| DeepEval | [deepeval-openeval-adapter](adapters/deepeval-openeval-adapter/) |
| Galileo | [galileo-openeval-adapter](adapters/galileo-openeval-adapter/) |

## Documentation

- [Getting Started](docs/getting-started/README.md)
- [Grader Type Reference](docs/grader-reference/README.md)
- [Migration Guides](docs/migration-guides/)
- [REST API](docs/api/README.md)
- [Full Specification](spec/SPEC.md)
- [Adoption Strategy](spec/ADOPTION.md)
- [Hostile Critique](spec/CRITIQUE.md)

## Community

- [Contributing Guide](.github/CONTRIBUTING.md)
- [Governance & the RFC process](spec/SPEC.md#governance)
- [Open Design Questions](spec/SPEC.md#open-design-questions--rfc-topics-we-need-help-with) — live Discussions on unresolved spec questions, no prior contribution required
- [GitHub Discussions](https://github.com/adhabnr-ux/evalport/discussions)
- [Contributors](CONTRIBUTORS.md)
- [Launch Blog Post](docs/blog/launch-post.md)
- [Landing Page](docs/landing-page.html)

## License

Apache 2.0 — see [LICENSE](LICENSE)
