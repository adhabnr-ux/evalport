# EvalPort — The Open Evaluation Standard

**Version:** 1.0.0-rc.1 | **License:** Apache 2.0

EvalPort is an open specification for portable LLM evaluation test cases, graders, suites, and results. It enables evaluation datasets to be shared across frameworks (DeepEval, Promptfoo, Ragas, Inspect AI, LangSmith, Braintrust, OpenAI Evals, MLflow) without loss of semantic fidelity.

## Quick Start

```bash
# Install CLI
npm install -g @evalport/cli

# Create an eval suite
openeval init my-eval-suite

# Validate
openeval validate my-eval-suite.json

# Convert from Promptfoo
openeval convert promptfoo openeval config.json output.json

# Python SDK
pip install openeval
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

## Repository Structure

```
openeval/
├── spec/
│   ├── SPEC.md                # Full specification
│   ├── ADOPTION.md            # Adoption strategy
│   ├── CRITIQUE.md            # Self-critique and hostile review
│   ├── schemas/               # JSON Schemas (4 files)
│   └── examples/              # Example suites and conversions
├── sdk/
│   ├── typescript/            # @evalport/sdk (npm)
│   └── python/                # openeval (PyPI)
├── cli/                       # @evalport/cli
├── api/                       # Example REST API server
├── docs/
│   ├── getting-started/       # 5-minute quickstart
│   ├── grader-reference/      # All 11 grader types
│   ├── migration-guides/      # Promptfoo, DeepEval, Inspect AI
│   ├── api/                   # REST API docs
│   ├── blog/                  # Launch posts
│   └── landing-page.html      # Landing page
├── examples/                  # Example eval suites
├── .github/                   # CI, CONTRIBUTING, issue templates
├── LICENSE                    # Apache 2.0
└── README.md                  # This file
```

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
- [Launch Blog Post](docs/blog/launch-post.md)
- [Landing Page](docs/landing-page.html)

## License

Apache 2.0 — see [LICENSE](LICENSE)
