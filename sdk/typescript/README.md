# evalport-sdk

TypeScript SDK for EvalPort — The Open Evaluation Standard.

## Install

```bash
npm install evalport-sdk
```

## Usage

### Validate a suite

```typescript
import { validateSuite } from "evalport-sdk";

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
import { fromPromptfoo } from "evalport-sdk";

const suite = fromPromptfoo(promptfooConfig);
```

### Compute summary

```typescript
import { computeSummary, createResultSet } from "evalport-sdk";

const summary = computeSummary(results);
const resultSet = createResultSet(suite, results, "run_001");
```

## API

- `validateSuite(doc, opts?)` → `ValidationResult`
- `validateTestCase(doc, opts?)` → `ValidationResult`
- `validateGrader(doc, opts?)` → `ValidationResult`
- `validateResultSet(doc, opts?)` → `ValidationResult`
- `validateDocument(doc, type, opts?)` → `ValidationResult`
- `fromPromptfoo(config)` → `EvalSuite`
- `computeSummary(results)` → `Summary`
- `createResultSet(suite, results, runId)` → `ResultSet`

Validation is strict by default (**proposed** in
[Discussion #108](https://github.com/adhabnr-ux/evalport/discussions/108)): a
property the schema doesn't define is an `UNKNOWN_FIELD` error at its path,
except inside `metadata` objects (and `provider.extra`, `params`,
`summary.by_grader` entries), which stay open. Pass `{ allowUnknown: true }`
when reading a document from a newer minor spec version.

## License

Apache 2.0
