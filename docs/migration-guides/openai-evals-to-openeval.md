# Migrating from OpenAI Evals to EvalPort

## Overview

OpenAI Evals uses registry configurations (YAML) and sample datasets (JSONL) with task-specific Python eval classes (`Match`, `Includes`, `FuzzyMatch`, `ModelGradedSpec`). EvalPort provides a portable, JSON-Schema-validated specification for evaluation suites. This guide explains how to convert OpenAI Evals datasets and configurations into EvalPort suites.

## Field Mapping

| OpenAI Evals | EvalPort | Notes |
|---|---|---|
| `id` | `EvalSuite.id` | Prefixed with `suite_` |
| `test_data[].input` / `prompt` (string) | `TestCase.input` | Direct mapping |
| `test_data[].input` (chat message objects) | `TestCase.input` | Flattened as `"role: content"` lines; original messages preserved in `metadata.openai_evals.messages` |
| `test_data[].ideal` / `target` (string) | `TestCase.expected_output` | Direct mapping |
| `test_data[].ideal` (list of strings) | `TestCase.expected_output` | Primary variant used; complete list preserved in `metadata.openai_evals.ideal_variants` |
| `test_data[].context` | `TestCase.context` | Wrapped in string list if scalar |
| `test_data[].metadata` | `TestCase.metadata` | Preserved directly |
| `config.sampling.model` | `config.provider.model` | Mapped to EvalPort suite provider config |
| `config.sampling.temperature` | `config.provider.temperature` | Mapped to EvalPort suite provider config |
| `config.grader` | `EvalSuite.graders[]` | Mapped to corresponding EvalPort grader |

## Grader Mapping

| OpenAI Evals Grader / Eval Class | EvalPort Grader Type | Parameters |
|---|---|---|
| `match`, `exact` | `exact_match` | Standard string comparison |
| `includes`, `contains` | `contains` | `params.substring` |
| `pattern`, `regex` | `regex` | `params.pattern` |
| `json` | `json_schema` | `params.schema` |
| `model_graded`, `modelgraded`, `llm` | `model graded` | `params.model`, `params.prompt` |
| Other / custom classes | `custom` | `params.handler: "openai_evals:<type>"` |

## SDK Conversion

### Python

```python
from openeval.converters_openai import from_openai_evals
import json

eval_data = json.load(open("openai_eval.json"))
suite = from_openai_evals(eval_data)
json.dump(suite, open("output.json", "w"), indent=2)
```

## Example

### Before (OpenAI Evals export)

```json
{
  "id": "math_qa",
  "config": {
    "sampling": {
      "model": "gpt-4o",
      "temperature": 0.0
    },
    "grader": {
      "type": "match"
    }
  },
  "test_data": [
    {
      "id": "tc_1",
      "input": [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 15 * 6?"}
      ],
      "ideal": ["90", "90.0"],
      "metadata": {
        "topic": "multiplication"
      }
    }
  ]
}
```

### After (EvalPort)

```json
{
  "version": "1.0.0-rc.5",
  "id": "suite_math_qa",
  "name": "Imported from OpenAI Evals: math_qa",
  "graders": [
    {
      "id": "gr_0",
      "type": "exact_match"
    }
  ],
  "test_cases": [
    {
      "id": "tc_1",
      "input": "system: You are a math tutor.\nuser: What is 15 * 6?",
      "graders": [
        "gr_0"
      ],
      "expected_output": "90",
      "metadata": {
        "topic": "multiplication",
        "openai_evals": {
          "messages": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 15 * 6?"}
          ],
          "ideal_variants": ["90", "90.0"]
        }
      }
    }
  ],
  "config": {
    "provider": {
      "model": "gpt-4o",
      "temperature": 0.0
    }
  },
  "metadata": {
    "openeval": {
      "source": "openai_evals"
    }
  }
}
```

## Limitations

- **Chat inputs**: EvalPort `TestCase.input` must be a string or array of strings. Chat message lists (`[{"role": ..., "content": ...}]`) are flattened into human-readable newline-separated messages, while retaining the raw message array in `metadata.openai_evals.messages` for lossless reconstruction.
- **Multiple ideal targets**: When `ideal` contains multiple acceptable target strings, the primary item is set as `TestCase.expected_output`, and all candidates are preserved in `metadata.openai_evals.ideal_variants`.
- **Complex completion workflows**: Multi-turn completion loops and specialized execution hooks require custom grader implementations or execution harness runners.
