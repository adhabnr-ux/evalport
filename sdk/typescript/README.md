# @evalport/sdk

TypeScript SDK for EvalPort — The Open Evaluation Standard.

## Install

```bash
npm install @evalport/sdk
```

## Usage

### Validate a suite

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

### Convert from Promptfoo

```typescript
import { fromPromptfoo } from "@evalport/sdk";

const suite = fromPromptfoo(promptfooConfig);
```

### Compute summary

```typescript
import { computeSummary, createResultSet } from "@evalport/sdk";

const summary = computeSummary(results);
const resultSet = createResultSet(suite, results, "run_001");
```

## API

- `validateSuite(doc)` → `ValidationResult`
- `validateTestCase(doc)` → `ValidationResult`
- `validateGrader(doc)` → `ValidationResult`
- `validateResultSet(doc)` → `ValidationResult`
- `fromPromptfoo(config)` → `EvalSuite`
- `computeSummary(results)` → `Summary`
- `createResultSet(suite, results, runId)` → `ResultSet`

## License

Apache 2.0
