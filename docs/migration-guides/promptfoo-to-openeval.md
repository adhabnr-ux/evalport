# Migrating from Promptfoo to OpenEval

## Overview

Promptfoo and OpenEval serve different purposes: Promptfoo is a full eval runner, OpenEval is a portable data format. This guide helps you convert Promptfoo configs to OpenEval suites.

## Field Mapping

| Promptfoo | OpenEval |
|-----------|----------|
| `vars.query` / `vars.prompt` | `TestCase.input` |
| `vars.expected` | `TestCase.expected_output` |
| `vars.context` | `TestCase.context` |
| `assert[].type: "equals"` | `Grader.type: "exact_match"` |
| `assert[].type: "contains"` | `Grader.type: "contains"` |
| `assert[].type: "regex"` | `Grader.type: "regex"` |
| `assert[].type: "contains-json"` | `Grader.type: "json_schema"` |
| `assert[].type: "ic"` | `Grader.type: "llm_judge"` |
| `providers[].model` | `config.provider.model` |

## CLI Conversion

```bash
openeval convert promptfoo openeval promptfoo-config.yaml output.json
```

## SDK Conversion

### Python

```python
from openeval.convert import from_promptfoo
import json

pf = json.load(open("promptfoo-config.json"))
suite = from_promptfoo(pf)
json.dump(suite, open("output.json", "w"), indent=2)
```

### TypeScript

```typescript
import { fromPromptfoo } from "@openeval/sdk";
import * as fs from "fs";

const pf = JSON.parse(fs.readFileSync("promptfoo-config.json", "utf-8"));
const suite = fromPromptfoo(pf);
fs.writeFileSync("output.json", JSON.stringify(suite, null, 2));
```

## Example

### Before (Promptfoo)

```json
{
  "description": "Q&A eval",
  "providers": [{"model": "gpt-4o"}],
  "tests": [
    {"vars": {"query": "What is 2+2?", "expected": "4"}, "assert": [{"type": "equals", "value": "{{expected}}"}]}
  ]
}
```

### After (OpenEval)

```json
{
  "version": "1.0.0",
  "id": "suite_promptfoo_import",
  "name": "Imported from Promptfoo",
  "graders": [{"id": "gr_0_0", "type": "exact_match"}],
  "test_cases": [{"id": "tc_0", "input": "What is 2+2?", "expected_output": "4", "graders": ["gr_0_0"]}],
  "config": {"provider": {"model": "gpt-4o"}}
}
```

## Limitations

- Promptfoo's `{{var}}` template syntax is not preserved; variables are resolved at conversion time
- Custom assertion types map to `custom` graders with `handler: "promptfoo:TYPE"`
- Promptfoo's `providerConfig` is not fully mapped; only model and temperature are preserved
