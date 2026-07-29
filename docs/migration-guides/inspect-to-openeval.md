# Migrating from Inspect AI to OpenEval

## Overview

Inspect AI (UK AISI) uses Python `Sample` objects and solver functions for evaluation. OpenEval provides a portable JSON format. This guide helps you convert Inspect tasks to OpenEval suites.

## Field Mapping

| Inspect AI | OpenEval |
|------------|----------|
| `Sample.input` | `TestCase.input` |
| `Sample.target` | `TestCase.expected_output` |
| `Sample.context` | `TestCase.context` |
| `Sample.metadata` | `TestCase.metadata` |
| `Sample.id` | `TestCase.id` |
| Solver functions | Not directly mapped (framework-specific execution) |
| `score()` return | `GraderResult.score` + `GraderResult.passed` |
| Scorer functions | `Grader` definitions |

## SDK Conversion

### Python

```python
from openeval.convert import from_inspect
import json

inspect_data = json.load(open("inspect_export.json"))
suite = from_inspect(inspect_data)
json.dump(suite, open("output.json", "w"), indent=2)
```

## Inspect Scorer → OpenEval Grader Mapping

| Inspect Scorer | OpenEval Grader Type |
|----------------|---------------------|
| `exact()` | `exact_match` |
| `pattern()` | `regex` |
| `includes()` | `contains` |
| `json_scorer()` | `json_schema` |
| `model_graded_*()` | `llm_judge` |
| `manual_review()` | `human` |
| Custom scorers | `code` or `custom` |

## Example

### Before (Inspect AI export)

```json
{
  "task": "math_qa",
  "samples": [
    {
      "id": "s1",
      "input": "What is 5 * 7?",
      "target": "35",
      "metadata": {"difficulty": "easy"}
    }
  ],
  "scorers": ["exact"]
}
```

### After (OpenEval)

```json
{
  "version": "1.0.0",
  "id": "suite_inspect_math_qa",
  "name": "Imported from Inspect AI",
  "graders": [{"id": "gr_exact", "type": "exact_match"}],
  "test_cases": [
    {"id": "s1", "input": "What is 5 * 7?", "expected_output": "35", "graders": ["gr_exact"], "metadata": {"difficulty": "easy"}}
  ],
  "metadata": {"openeval": {"source": "inspect_ai"}}
}
```

## Limitations

- Inspect's solver pipeline (generate, regex_extract, etc.) is execution logic, not eval data — not mapped
- Inspect's `Task` configuration (model, config) maps to suite-level `config.provider`
- Multi-step agent tasks require manual conversion of intermediate steps to `metadata`
