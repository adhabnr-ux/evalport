# Getting Started with EvalPort

## 5-Minute Quickstart

### 1. Install

```bash
# CLI
npm install -g @evalport/cli

# TypeScript SDK
npm install @evalport/sdk

# Python SDK
pip install openeval
```

### 2. Create an Eval Suite

```bash
openeval init my-eval-suite
```

This creates `my-eval-suite.json`:

```json
{
  "$schema": "https://evalport.org/schema/suite.json",
  "version": "1.0.0",
  "id": "my-eval-suite",
  "name": "my-eval-suite",
  "graders": [
    {"id": "gr_exact", "type": "exact_match", "params": {"ignore_case": true}}
  ],
  "test_cases": [
    {"id": "tc_001", "input": "Example?", "expected_output": "Answer", "graders": ["gr_exact"]}
  ],
  "config": {"provider": {"model": "gpt-4o", "temperature": 0}}
}
```

### 3. Validate

```bash
openeval validate my-eval-suite.json
# Output: Valid
```

### 4. Convert from Promptfoo

```bash
openeval convert promptfoo openeval promptfoo-config.json output.json
```

### 5. Use the Python SDK

```python
from openeval.validate import validate_suite
from openeval.convert import from_promptfoo

# Validate a suite
result = validate_suite({
    "version": "1.0.0",
    "id": "my_suite",
    "graders": [{"id": "gr1", "type": "exact_match"}],
    "test_cases": [{"id": "tc1", "input": "Hello", "expected_output": "Hi", "graders": ["gr1"]}]
})
print(result.valid)  # True

# Convert from Promptfoo
import json
pf_config = json.load(open("promptfoo-config.json"))
openeval_suite = from_promptfoo(pf_config)
print(openeval_suite["id"])  # suite_promptfoo_import
```

### 6. Use the TypeScript SDK

```typescript
import { validateSuite } from "@evalport/sdk";

const result = validateSuite({
  version: "1.0.0",
  id: "my_suite",
  graders: [{ id: "gr1", type: "exact_match" }],
  test_cases: [{ id: "tc1", input: "Hello", expected_output: "Hi", graders: ["gr1"] }]
});
console.log(result.valid); // true
```

## Key Concepts

- **TestCase**: One eval input + expected output + grader references
- **Grader**: A scoring criterion (exact_match, semantic_similarity, llm_judge, etc.)
- **EvalSuite**: A collection of test cases + shared graders + config
- **ResultSet**: Results from running a suite (scores, pass/fail, summary)

## Next Steps

- [Grader Type Reference](../grader-reference/README.md) — all 11 grader types
- [Migration Guides](../migration-guides/) — convert from DeepEval, Promptfoo, Inspect AI
- [API Reference](../api/README.md) — REST API for serving eval suites
- [Full Specification](../../spec/SPEC.md) — the complete EvalPort spec
