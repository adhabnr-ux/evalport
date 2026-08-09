# EvalPort — The Open Evaluation Standard

**Version:** 1.0.0-draft  
**Status:** Draft for Community Review  
**License:** Apache 2.0  
**Specification Lead:** EvalPort Working Group

---

## Abstract

EvalPort is an open, language-agnostic specification for representing LLM evaluation test cases, scoring criteria (graders), evaluation suites, and result sets. It defines a portable data format that enables evaluation datasets and results to be shared across evaluation frameworks (DeepEval, Promptfoo, Ragas, Inspect AI, LangSmith, Braintrust, OpenAI Evals, MLflow, and others) without loss of semantic fidelity.

The specification consists of four JSON document types — **TestCase**, **Grader**, **EvalSuite**, and **ResultSet** — each defined by a JSON Schema, together with a grader type system, validation rules, versioning policy, and extension mechanism. Reference implementations are provided as TypeScript and Python SDKs, a CLI tool, and example integrations.

---

## Motivation

The LLM evaluation ecosystem is fragmented across 10+ major frameworks, each with its own data model, field names, grader representation, and result schema. This fragmentation creates concrete, recurring problems:

1. **Eval datasets are not portable.** A test suite built in DeepEval cannot be run in Promptfoo without manual conversion. Teams using different tools cannot share evaluation datasets.
2. **Graders are not interoperable.** Each framework represents scoring criteria differently — DeepEval uses metric classes, Promptfoo uses assertion objects, Inspect uses solver functions, OpenAI Evals uses grader specs. A semantic-similarity grader written for one framework must be reimplemented for another.
3. **Results cannot be compared.** Eval results from LangSmith, Braintrust, and Arize use incompatible schemas, making cross-tool comparison and benchmarking impossible without manual normalization.
4. **Vendor lock-in.** Teams that invest hundreds of hours building eval datasets in one framework face switching costs that lock them in, even when another framework would be a better fit.
5. **No shared benchmark format.** The community cannot publish reproducible benchmark datasets that work across eval frameworks, the way ImageNet or GLUE did for ML.

EvalPort addresses these problems by defining a minimal, extensible data format that preserves the full semantics of an evaluation — test inputs, expected outputs, graders, and results — in a way that any framework can import, export, and natively support.

---

## Problem Statement

**There is no widely adopted, standard, portable format for LLM evaluation test cases, scoring criteria, and results.** Every major evaluation framework uses a proprietary format, and existing standardization attempts are narrow (covering only results metadata) or have minimal adoption (under 100 GitHub stars).

A practitioner who builds a 500-case RAG evaluation suite in DeepEval and wants to run it in Promptfoo must:
- Rewrite each test case to Promptfoo's assertion format
- Reimplement each grader (faithfulness, answer relevancy, context precision) as a Promptfoo custom assertion
- Manually map result fields when comparing outputs

This is not a one-time cost — it repeats for every framework transition, every team handoff, and every benchmark reproduction.

**EvalPort solves this by defining a standard format that frameworks can natively read and write, making evaluation datasets and results portable.**

---

## Terminology

| Term | Definition |
|------|-----------|
| **Test Case** | A single evaluation input with its expected output, context, and grader references. The atomic unit of evaluation. |
| **Grader** | A scoring criterion that evaluates a test case's actual output against expected output. Has a type (e.g., `exact_match`, `llm_judge`), parameters, and a pass threshold. |
| **Eval Suite** | A named collection of test cases and shared grader definitions, with execution configuration. Analogous to a test suite in traditional software testing. |
| **Result Set** | The output of running an eval suite — actual outputs, scores, pass/fail status, and summary statistics for each test case. |
| **Eval Runner** | A framework or tool that executes an eval suite and produces a result set (e.g., DeepEval, Promptfoo, Inspect AI). |
| **Eval Consumer** | A tool that imports or displays eval data — dashboards (Arize, LangSmith), CI systems, benchmark aggregators. |
| **Provider** | The LLM or service being evaluated. May be specified per-test-case or at the suite level. |
| **Context** | Supplementary data provided to the LLM during evaluation (retrieved documents, conversation history, tool call results). |

---

## Goals

1. **Portability.** An eval suite authored once can be imported and executed by any compliant eval runner.
2. **Semantic fidelity.** Grader definitions preserve enough detail that a runner can execute them natively or flag unsupported grader types.
3. **Extensibility.** Custom grader types, metadata fields, and extensions can be added without breaking interoperability.
4. **Simplicity.** The core format is JSON and can be authored by hand or generated programmatically.
5. **Bidirectional conversion.** Existing framework formats can be converted to and from EvalPort with minimal loss.
6. **Reproducibility.** Result sets capture enough information (provider, model, timestamps, config) to reproduce an evaluation run.
7. **Human-readable.** Eval suites and results are readable as JSON/YAML without proprietary tooling.

---

## Non-goals

1. **EvalPort does not define an evaluation runner.** It is a data format, not an execution engine. Runners are framework-specific.
2. **EvalPort does not mandate specific grader implementations.** A `semantic_similarity` grader specifies the threshold and model, but the implementation of embedding comparison is runner-specific.
3. **EvalPort does not define a trace format.** Execution traces are covered by OpenTelemetry GenAI semantic conventions. EvalPort references trace IDs but does not define trace structure.
4. **EvalPort does not define a UI.** Dashboards and visualization are framework-specific.
5. **EvalPort does not define model APIs.** How a runner calls an LLM provider is out of scope.
6. **EvalPort does not define access control.** File-level permissions are the deployer's responsibility.

---

## Architecture

### Document Model

```
┌─────────────────────────────────────────────────────┐
│                    Eval Suite                         │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Test Case │  │ Test Case │  │ Test Case │  ...    │
│  │  #1       │  │  #2       │  │  #3       │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                 │
│       └──────────────┼──────────────┘                 │
│                      ▼                                │
│              ┌──────────────┐                        │
│              │   Graders    │  (shared definitions)   │
│              │  (referenced │                         │
│              │   by ID)     │                         │
│              └──────────────┘                        │
│                                                      │
│  ┌──────────────────────────────────┐               │
│  │       Suite Configuration         │               │
│  │  (provider, model, defaults)      │               │
│  └──────────────────────────────────┘               │
└─────────────────────────────────────────────────────┘

                        │ run

                        ▼

┌─────────────────────────────────────────────────────┐
│                    Result Set                        │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Result #1  │  │  Result #2  │  │  Result #3  │   │
│  │  (score,    │  │  (score,    │  │  (score,    │   │
│  │   pass/fail)│  │   pass/fail)│  │   pass/fail)│   │
│  └────────────┘  └────────────┘  └────────────┘    │
│                                                      │
│  ┌──────────────────────────────────┐               │
│  │       Summary Statistics          │               │
│  │  (pass rate, avg score, per-grader│               │
│  │   breakdown, duration)            │               │
│  └──────────────────────────────────┘               │
└─────────────────────────────────────────────────────┘
```

### Document Relationships

- An **Eval Suite** contains 1..N **Test Cases** and 0..N **Grader** definitions.
- Each **Test Case** references 1..N graders by ID.
- A **Result Set** contains 1..N **Results**, one per test case in the source suite.
- Each **Result** contains 1..N **Grader Results**, one per grader applied to that test case.

### File Formats

- **JSON** is the canonical format. All schemas are defined against JSON Schema 2020-12.
- **YAML** is supported as an alternative serialization. YAML files MUST be convertible to semantically identical JSON.
- **JSONL** (JSON Lines) is supported for streaming test cases. Each line is a complete `TestCase` document.

---

## Data Model

### 1. TestCase

A test case is the atomic unit of evaluation. It represents a single input to an LLM system and the criteria for evaluating the output.

```json
{
  "$schema": "https://evalport.org/schema/testcase.json",
  "id": "tc_001",
  "input": "What is the capital of France?",
  "expected_output": "Paris",
  "context": [
    "France is a country in Western Europe. Its capital is Paris."
  ],
  "graders": ["gr_exact_match", "gr_semantic_sim"],
  "metadata": {
    "category": "geography",
    "difficulty": "easy",
    "source": "manual"
  },
  "tags": ["rag", "factual"]
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (unique within suite) | Unique identifier for the test case. |
| `input` | string \| array | The input prompt(s) sent to the LLM. Array form represents conversational turns. |
| `graders` | array of string | IDs of graders to apply. Must reference graders defined in the enclosing suite, or be inline grader objects. |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `expected_output` | string | The reference/golden output. Required for graders that compare against a ground truth. |
| `context` | array of string | Supplementary context (retrieved docs, conversation history, tool results). |
| `retrieval_context` | array of string | Documents retrieved by a RAG system, separated from general context for RAG-specific graders. |
| `tools_called` | array of string | Names of tools expected to be called (for agent evaluation). |
| `expected_tools` | array of string | Names of tools that SHOULD be called (for agent evaluation). |
| `metadata` | object | Free-form metadata (category, difficulty, source, etc.). Keys `openeval.*` are reserved. |
| `tags` | array of string | Categorization tags for filtering and grouping. |
| `provider` | object | Per-test-case provider override (see Suite Configuration). |
| `params` | object | Per-test-case generation parameters (temperature, max_tokens, etc.). |
| `timeout_ms` | integer | Maximum execution time for this test case in milliseconds. |
| `weight` | number | Relative weight for score aggregation (default: 1.0). |

---

### 2. Grader

A grader defines how a test case's actual output is scored. Graders are defined once in the eval suite and referenced by ID from test cases.

```json
{
  "$schema": "https://evalport.org/schema/grader.json",
  "id": "gr_semantic_sim",
  "type": "semantic_similarity",
  "params": {
    "model": "text-embedding-3-small",
    "threshold": 0.85
  },
  "weight": 1.0
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (unique within suite) | Unique identifier for the grader. |
| `type` | string | The grader type (see Grader Type System). |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `params` | object | Type-specific parameters (threshold, model, schema, etc.). |
| `weight` | number | Relative weight when aggregating scores (default: 1.0). |
| `description` | string | Human-readable description of what this grader checks. |

#### Grader Type System

| Type | Description | Required Params | Optional Params |
|------|-------------|----------------|-----------------|
| `exact_match` | String equality (case-sensitive or insensitive) | — | `ignore_case` (bool, default false), `trim_whitespace` (bool, default true) |
| `contains` | Checks if actual output contains a substring | `substring` (string) | `ignore_case` (bool) |
| `regex` | Matches actual output against a regex pattern | `pattern` (string, RE2 syntax) | `flags` (string) |
| `semantic_similarity` | Cosine similarity of embeddings | `threshold` (number, 0-1) | `model` (string), `provider` (string) |
| `llm_judge` | An LLM evaluates the output against a rubric | `model` (string), `prompt` (string) | `provider` (string), `temperature` (number), `schema` (object) |
| `json_schema` | Validates actual output against a JSON Schema | `schema` (object) | `strict` (bool) |
| `json_path` | Extracts a value via JSONPath and compares it | `path` (string), `expected` (string) | `operator` (string: eq, ne, gt, lt, gte, lte, contains) |
| `code` | Executes a custom grading function | `language` (string: "python", "javascript"), `source` (string) | `timeout_ms` (integer) |
| `human` | Defers to human review | — | `instructions` (string) |
| `model graded` | Alias for `llm_judge` (OpenAI Evals compatibility) | Same as `llm_judge` | Same as `llm_judge` |
| `custom` | Framework-specific grader not in the standard set | `handler` (string) | any |

**Custom grader handling:** When a runner encounters a `custom` grader type or any unrecognized type, it MUST:
1. Check if it has a handler registered for the `handler` string or type name.
2. If no handler is available, mark the grader result as `skipped` with reason `unsupported_grader_type`.
3. Never fail the entire suite due to an unsupported grader.

---

### 3. EvalSuite

An eval suite is a named collection of test cases and shared grader definitions.

```json
{
  "$schema": "https://evalport.org/schema/suite.json",
  "version": "1.0.0",
  "id": "suite_rag_eval_001",
  "name": "RAG Evaluation Suite — Knowledge Base v2",
  "description": "Evaluates RAG pipeline against 50 factual questions",
  "graders": [
    {
      "id": "gr_exact_match",
      "type": "exact_match",
      "params": { "ignore_case": true }
    },
    {
      "id": "gr_semantic_sim",
      "type": "semantic_similarity",
      "params": { "model": "text-embedding-3-small", "threshold": 0.85 }
    }
  ],
  "test_cases": [
    {
      "id": "tc_001",
      "input": "What is the capital of France?",
      "expected_output": "Paris",
      "context": ["France is a country in Western Europe. Its capital is Paris."],
      "graders": ["gr_exact_match", "gr_semantic_sim"]
    }
  ],
  "config": {
    "provider": {
      "model": "gpt-4o",
      "temperature": 0.0
    },
    "defaults": {
      "timeout_ms": 30000,
      "weight": 1.0
    }
  },
  "metadata": {
    "author": "jane@example.com",
    "created": "2026-01-15T10:00:00Z",
    "version": "1.0.0"
  }
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | EvalPort specification version (semver). |
| `id` | string | Unique identifier for the suite. |
| `test_cases` | array of TestCase | One or more test cases. |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable suite name. |
| `description` | string | Longer description of the suite's purpose. |
| `graders` | array of Grader | Shared grader definitions referenced by test cases. |
| `config` | object | Suite-level configuration (provider, defaults). |
| `metadata` | object | Free-form metadata. Keys `openeval.*` are reserved. |
| `tags` | array of string | Suite-level tags. |

#### Suite Configuration (`config`)

| Field | Type | Description |
|-------|------|-------------|
| `provider` | object | Default provider and model settings. |
| `provider.model` | string | Model identifier (e.g., `gpt-4o`, `claude-sonnet-4-20250514`). |
| `provider.api_base` | string | Base URL for API calls (for self-hosted models). |
| `provider.api_key_env` | string | Name of environment variable containing the API key. Never the key itself. |
| `provider.temperature` | number | Sampling temperature. |
| `provider.max_tokens` | integer | Max output tokens. |
| `provider.extra` | object | Provider-specific parameters. |
| `defaults` | object | Default values for optional test case fields. |
| `defaults.timeout_ms` | integer | Default timeout. |
| `defaults.weight` | number | Default test case weight. |
| `parallel` | integer | Number of test cases to run in parallel (runner may ignore). |
| `retry` | object | Retry configuration. |
| `retry.max_attempts` | integer | Max retry attempts on provider errors. |
| `retry.backoff_ms` | integer | Initial backoff in milliseconds. |

---

### 4. ResultSet

A result set is the output of running an eval suite. It contains one result per test case, plus summary statistics.

```json
{
  "$schema": "https://evalport.org/schema/resultset.json",
  "version": "1.0.0",
  "suite_id": "suite_rag_eval_001",
  "suite_version": "1.0.0",
  "run_id": "run_20260115_103000",
  "started_at": "2026-01-15T10:30:00Z",
  "completed_at": "2026-01-15T10:31:45Z",
  "provider": {
    "model": "gpt-4o",
    "temperature": 0.0
  },
  "results": [
    {
      "test_case_id": "tc_001",
      "actual_output": "The capital of France is Paris.",
      "grader_results": [
        {
          "grader_id": "gr_exact_match",
          "type": "exact_match",
          "score": 0.0,
          "passed": false,
          "reason": "Expected 'Paris', got 'The capital of France is Paris.'"
        },
        {
          "grader_id": "gr_semantic_sim",
          "type": "semantic_similarity",
          "score": 0.92,
          "passed": true,
          "metadata": {
            "similarity": 0.92,
            "threshold": 0.85
          }
        }
      ],
      "passed": false,
      "duration_ms": 1200,
      "metadata": {
        "trace_id": "trace_abc123"
      }
    }
  ],
  "summary": {
    "total": 1,
    "passed": 0,
    "failed": 1,
    "skipped": 0,
    "pass_rate": 0.0,
    "avg_score": 0.46,
    "duration_ms": 1200,
    "by_grader": {
      "gr_exact_match": { "passed": 0, "failed": 1, "avg_score": 0.0 },
      "gr_semantic_sim": { "passed": 1, "failed": 0, "avg_score": 0.92 }
    }
  }
}
```

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | EvalPort specification version. |
| `suite_id` | string | ID of the eval suite that was run. |
| `run_id` | string | Unique identifier for this run. |
| `started_at` | string (ISO 8601) | Run start timestamp. |
| `results` | array of Result | One result per test case. |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `suite_version` | string | Version of the eval suite that was run. |
| `completed_at` | string (ISO 8601) | Run completion timestamp. |
| `provider` | object | Provider configuration used for the run. |
| `runner` | object | Information about the runner (name, version). |
| `runner.name` | string | Runner name (e.g., "deepeval", "promptfoo"). |
| `runner.version` | string | Runner version. |
| `summary` | object | Aggregated statistics. |
| `metadata` | object | Free-form metadata. |

#### Result Object

| Field | Type | Description |
|-------|------|-------------|
| `test_case_id` | string (required) | ID of the test case this result corresponds to. |
| `actual_output` | string | The output produced by the LLM. |
| `grader_results` | array (required) | Results from each grader. |
| `passed` | boolean (required) | Overall pass/fail (all graders passed). |
| `duration_ms` | integer | Execution time. |
| `error` | object | Error details if the test case errored. |
| `error.message` | string | Error message. |
| `error.type` | string | Error type (`timeout`, `provider_error`, `runner_error`). |
| `metadata` | object | Free-form metadata (trace ID, cost, tokens). |

#### GraderResult Object

| Field | Type | Description |
|-------|------|-------------|
| `grader_id` | string (required) | ID of the grader. |
| `type` | string (required) | Grader type. |
| `score` | number (required) | Numeric score, typically 0.0-1.0. |
| `passed` | boolean (required) | Whether the grader's threshold was met. |
| `reason` | string | Human-readable explanation. |
| `metadata` | object | Grader-specific details (similarity value, judge response, etc.). |

---

## Validation Rules

### 1. Schema Validation

All documents MUST validate against their respective JSON Schemas. Runners MUST reject documents that fail schema validation with a clear error message identifying the failing field.

### 2. Referential Integrity

- Every grader ID referenced in a `TestCase.graders` array MUST exist in the suite's `graders` array, OR be an inline grader object.
- Every `test_case_id` in a `ResultSet` MUST correspond to a test case in the source suite.

### 3. Uniqueness

- Test case IDs MUST be unique within a suite.
- Grader IDs MUST be unique within a suite.
- Run IDs SHOULD be globally unique (recommend UUID or timestamp + random suffix).

### 4. Type-Specific Grader Validation

- `exact_match`: No required params. If `ignore_case` is true, comparison is case-insensitive.
- `contains`: `substring` param is required and MUST be a non-empty string.
- `regex`: `pattern` param is required and MUST be a valid RE2 regex.
- `semantic_similarity`: `threshold` param is required and MUST be between 0.0 and 1.0.
- `llm_judge`: `model` and `prompt` params are required. `prompt` MUST contain the token `{output}` or `{input}` or `{expected}` for variable substitution.
- `json_schema`: `schema` param is required and MUST be a valid JSON Schema object.
- `json_path`: `path` and `expected` params are required. `path` MUST be a valid JSONPath expression.
- `code`: `language` and `source` params are required. `language` MUST be one of `python`, `javascript`.

### 5. Score Range

- Grader scores MUST be in the range [0.0, 1.0] unless a `score_range` extension is specified.
- Pass/fail is determined by comparing the score to the grader's threshold (default threshold: 1.0 for exact match, specified via params for other types).

### 6. Result Consistency

- Every grader referenced by a test case MUST have a corresponding `GraderResult` in the result set, unless the grader was `skipped`.
- Skipped graders MUST be represented with `score: null`, `passed: false`, and `metadata.skip_reason`.

---

## Versioning

### Specification Version

EvalPort follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes to the data model (removed fields, changed semantics).
- **MINOR**: Backward-compatible additions (new optional fields, new grader types).
- **PATCH**: Backward-compatible fixes (clarifications, schema corrections).

The `version` field in each document specifies the EvalPort spec version the document conforms to.

### Compatibility Policy

- Runners MUST accept documents with a higher minor version than their own implementation, ignoring unknown optional fields.
- Runners MUST reject documents with a higher major version, producing an error: `Unsupported EvalPort major version: {version}. Supported: {supported}.`
- Runners SHOULD warn on unknown grader types but continue execution.

### Schema Evolution

JSON Schemas are versioned and published at:
- `https://evalport.org/schema/testcase.json` (latest)
- `https://evalport.org/schema/v1.0.0/testcase.json` (pinned)

---

## Extension Mechanism

### Custom Fields

Any document may include a `metadata` object with arbitrary keys. Keys with the `openeval.*` prefix are reserved for future specification use. Custom keys SHOULD use a reverse-DNS prefix (e.g., `com.example.myfield`).

### Custom Grader Types

Graders with `type: "custom"` or any type not in the standard set are permitted. The `handler` field in `params` identifies the custom grader implementation. Runners that don't recognize the handler MUST mark the result as `skipped`.

### Extensions Registry

EvalPort maintains an extensions registry at `https://evalport.org/extensions` where the community can register:
- Custom grader types with handler identifiers
- Provider-specific configuration extensions
- Metadata field conventions

### Profile Extensions

A **profile** is a named set of conventions for a specific use case. For example:
- `profile:rag` — conventions for RAG evaluation (retrieval_context field usage, standard RAG graders)
- `profile:agent` — conventions for agent evaluation (tools_called, expected_tools fields)
- `profile:safety` — conventions for safety/toxicity evaluation

Profiles are declared in the suite metadata: `"metadata": { "openeval.profile": "rag" }`. Profiles do not change the schema — they document conventions for field usage.

---

## Error Handling

### Document-Level Errors

| Error Code | Condition | Action |
|------------|-----------|--------|
| `SCHEMA_INVALID` | Document fails JSON Schema validation | Runner rejects the document with field-level error details. |
| `VERSION_UNSUPPORTED` | Major version exceeds runner's supported version | Runner rejects the document. |
| `DUPLICATE_ID` | Test case or grader ID is not unique | Runner rejects the suite. |
| `DANGLING_REFERENCE` | Grader ID referenced in test case not found in suite | Runner rejects the suite. |

### Execution-Level Errors

| Error Code | Condition | Action |
|------------|-----------|--------|
| `TIMEOUT` | Test case exceeded `timeout_ms` | Result recorded with `error.type: "timeout"`, `passed: false`. |
| `PROVIDER_ERROR` | LLM provider returned an error | Result recorded with `error.type: "provider_error"`, `passed: false`. |
| `GRADER_ERROR` | Grader implementation threw an exception | GraderResult recorded with `score: null`, `passed: false`, `metadata.error`. |
| `UNSUPPORTED_GRADER` | Runner doesn't support the grader type | GraderResult recorded with `score: null`, `passed: false`, `metadata.skip_reason: "unsupported_grader_type"`. |

### Error Reporting

Errors MUST be structured, not string messages. The `error` object in results contains:
```json
{
  "error": {
    "type": "provider_error",
    "message": "Rate limit exceeded",
    "code": 429,
    "retryable": true
  }
}
```

---

## Security Considerations

### Prompt Injection in Graders

`llm_judge` graders use an LLM to evaluate outputs, which creates a prompt injection risk: a malicious test case input could cause the judge LLM to produce incorrect scores. Mitigations:
- Judge prompts SHOULD use structured output (JSON schema) to constrain the judge's response.
- Judge prompts SHOULD include the output in a delimited section, not concatenated with instructions.
- Runners SHOULD cap judge LLM output length.

### Code Grader Execution

`code` graders execute arbitrary code. Runners MUST:
- Execute code graders in an isolated sandbox (container, WASM, or subprocess with restricted permissions).
- Enforce `timeout_ms` (default: 5000ms).
- NOT execute code graders by default in CI environments unless explicitly enabled via `--allow-code-graders` flag or equivalent.

### API Key Handling

- API keys MUST NEVER be stored in EvalPort documents. Use `api_key_env` to reference an environment variable.
- Runners MUST NOT log or expose API keys when serializing suite configuration.

### Supply Chain Risks

- Eval suites shared between organizations may contain malicious test cases (e.g., inputs designed to trigger harmful outputs).
- Runners SHOULD validate that test case inputs don't contain known exploit patterns when importing third-party suites.
- The `metadata.source` field SHOULD indicate the origin of test cases.

---

## Privacy Considerations

### PII in Test Cases

Eval test cases may contain personally identifiable information (PII). EvalPort does not redact or encrypt PII — it is the deployer's responsibility to:
- Classify eval suites containing PII as confidential.
- Store and transmit eval suites over encrypted channels.
- Avoid sharing eval suites containing real user data without consent.

### PII in Results

Result sets contain `actual_output` which may include PII from the LLM's response. The same precautions apply.

### Right to Be Forgotten

To support GDPR/right-to-be-forgotten requests:
- Test cases and results include a `metadata.source` field to trace data origin.
- Runners SHOULD support deletion of individual test cases and their corresponding results from a result set.

### Metadata Minimization

The `metadata` field is free-form. Producers SHOULD minimize the inclusion of identifying information in metadata. Reserved key `openeval.pii_flags` can list fields containing PII for downstream tooling:
```json
"metadata": {
  "openeval.pii_flags": ["input", "expected_output"]
}
```

---

## Backward Compatibility

### Migration from Framework Formats

EvalPort is designed to be a superset of common framework formats. Conversion guides are provided for:

- **DeepEval** → EvalPort: `LLMTestCase` maps to `TestCase`; metrics map to graders.
- **Promptfoo** → EvalPort: test case `vars` map to `input`/`context`; `assert` maps to inline graders.
- **OpenAI Evals** → EvalPort: `test` maps to `TestCase`; `grader` maps to `Grader`.
- **Inspect AI** → EvalPort: `Sample` maps to `TestCase`; solvers map to graders.
- **LangSmith** → EvalPort: dataset examples map to `TestCase`; evaluators map to graders.
- **Braintrust** → EvalPort: test cases and scores map directly; scorer functions map to graders.

### Forward Compatibility

- New optional fields may be added in minor versions. Runners MUST ignore unknown fields rather than failing.
- New grader types may be added in minor versions. Runners MUST handle unknown grader types gracefully (skip, don't fail).

### Deprecation Policy

- Fields marked as deprecated in a minor version MUST be supported for at least one major version cycle.
- Deprecation is announced via the `openeval.deprecated` metadata key in the schema and in release notes.

---

## Examples

### Example 1: Simple Q&A Eval Suite

```json
{
  "$schema": "https://evalport.org/schema/suite.json",
  "version": "1.0.0",
  "id": "suite_qa_basic",
  "name": "Basic Q&A Evaluation",
  "graders": [
    {
      "id": "gr_exact",
      "type": "exact_match",
      "params": { "ignore_case": true }
    }
  ],
  "test_cases": [
    {
      "id": "tc_1",
      "input": "What is 2+2?",
      "expected_output": "4",
      "graders": ["gr_exact"]
    },
    {
      "id": "tc_2",
      "input": "What is the boiling point of water in Celsius?",
      "expected_output": "100",
      "graders": ["gr_exact"]
    }
  ],
  "config": {
    "provider": { "model": "gpt-4o", "temperature": 0.0 }
  }
}
```

### Example 2: RAG Evaluation Suite

```json
{
  "$schema": "https://evalport.org/schema/suite.json",
  "version": "1.0.0",
  "id": "suite_rag_001",
  "name": "RAG Pipeline Evaluation",
  "description": "Evaluates RAG pipeline with faithfulness and relevancy checks",
  "metadata": { "openeval.profile": "rag" },
  "graders": [
    {
      "id": "gr_faithfulness",
      "type": "llm_judge",
      "description": "Checks if output is faithful to retrieved context",
      "params": {
        "model": "gpt-4o",
        "prompt": "Given the context: {context}\nAnd the output: {output}\nIs the output fully supported by the context? Respond with JSON: {\"score\": 0.0-1.0, \"reason\": \"...\"}",
        "schema": {
          "type": "object",
          "properties": {
            "score": { "type": "number" },
            "reason": { "type": "string" }
          },
          "required": ["score", "reason"]
        }
      }
    },
    {
      "id": "gr_answer_relevancy",
      "type": "semantic_similarity",
      "params": { "model": "text-embedding-3-small", "threshold": 0.75 }
    }
  ],
  "test_cases": [
    {
      "id": "tc_001",
      "input": "What is Kubernetes?",
      "expected_output": "Kubernetes is an open-source container orchestration platform.",
      "context": ["Kubernetes is an open-source container orchestration system for automating deployment and scaling."],
      "retrieval_context": ["Kubernetes is an open-source container orchestration system..."],
      "graders": ["gr_faithfulness", "gr_answer_relevancy"],
      "metadata": { "category": "tech", "difficulty": "medium" }
    }
  ],
  "config": {
    "provider": { "model": "gpt-4o", "temperature": 0.0 },
    "defaults": { "timeout_ms": 30000 }
  }
}
```

### Example 3: Agent Evaluation Suite

```json
{
  "$schema": "https://evalport.org/schema/suite.json",
  "version": "1.0.0",
  "id": "suite_agent_001",
  "name": "Agent Tool Selection Evaluation",
  "metadata": { "openeval.profile": "agent" },
  "graders": [
    {
      "id": "gr_tools_correct",
      "type": "exact_match",
      "description": "Checks if the agent called the expected tools",
      "params": { "ignore_case": true }
    },
    {
      "id": "gr_json_response",
      "type": "json_schema",
      "params": {
        "schema": {
          "type": "object",
          "properties": {
            "action": { "type": "string" },
            "parameters": { "type": "object" }
          },
          "required": ["action"]
        }
      }
    }
  ],
  "test_cases": [
    {
      "id": "tc_001",
      "input": "Search for recent papers on quantum computing",
      "expected_tools": ["web_search"],
      "expected_output": "{\"action\": \"web_search\", \"parameters\": {\"query\": \"recent papers quantum computing\"}}",
      "graders": ["gr_tools_correct", "gr_json_response"],
      "metadata": { "scenario": "tool_selection" }
    }
  ]
}
```

### Example 4: Result Set

```json
{
  "$schema": "https://evalport.org/schema/resultset.json",
  "version": "1.0.0",
  "suite_id": "suite_qa_basic",
  "suite_version": "1.0.0",
  "run_id": "run_20260115_100000",
  "started_at": "2026-01-15T10:00:00Z",
  "completed_at": "2026-01-15T10:00:05Z",
  "runner": { "name": "evalport-cli", "version": "1.0.0" },
  "provider": { "model": "gpt-4o", "temperature": 0.0 },
  "results": [
    {
      "test_case_id": "tc_1",
      "actual_output": "4",
      "grader_results": [
        {
          "grader_id": "gr_exact",
          "type": "exact_match",
          "score": 1.0,
          "passed": true
        }
      ],
      "passed": true,
      "duration_ms": 450
    },
    {
      "test_case_id": "tc_2",
      "actual_output": "The boiling point of water is 100°C at sea level.",
      "grader_results": [
        {
          "grader_id": "gr_exact",
          "type": "exact_match",
          "score": 0.0,
          "passed": false,
          "reason": "Expected '100', got 'The boiling point of water is 100°C at sea level.'"
        }
      ],
      "passed": false,
      "duration_ms": 520
    }
  ],
  "summary": {
    "total": 2,
    "passed": 1,
    "failed": 1,
    "skipped": 0,
    "pass_rate": 0.5,
    "avg_score": 0.5,
    "duration_ms": 970,
    "by_grader": {
      "gr_exact": { "passed": 1, "failed": 1, "avg_score": 0.5 }
    }
  }
}
```

### Example 5: JSONL Streaming Format

For large suites, test cases can be streamed as JSONL. Each line is a complete `TestCase` document:

```jsonl
{"id":"tc_001","input":"What is 2+2?","expected_output":"4","graders":["gr_exact"]}
{"id":"tc_002","input":"What is the capital of Japan?","expected_output":"Tokyo","graders":["gr_exact"]}
{"id":"tc_003","input":"Who wrote Hamlet?","expected_output":"William Shakespeare","graders":["gr_exact"]}
```

The accompanying suite metadata file (`suite.json`) references the JSONL file:
```json
{
  "version": "1.0.0",
  "id": "suite_large_qa",
  "test_cases_file": "test_cases.jsonl",
  "graders": [
    { "id": "gr_exact", "type": "exact_match" }
  ]
}
```

---

## Reference Implementation

The EvalPort reference implementation includes:

1. **JSON Schemas** — `schema/testcase.json`, `schema/grader.json`, `schema/suite.json`, `schema/resultset.json`
2. **TypeScript SDK** — `evalport-sdk` npm package for reading, writing, and validating EvalPort documents
3. **Python SDK** — `openeval` PyPI package with the same capabilities
4. **CLI** — `openeval` command-line tool for validation, conversion, and suite initialization
5. **Example API** — A REST API for serving and running eval suites
6. **Example integrations** — Migrated eval suites from DeepEval, Promptfoo, and Inspect AI formats

See the `README.md` for installation and usage instructions.

---

## Migration Guide

### From DeepEval

| DeepEval | EvalPort |
|----------|----------|
| `LLMTestCase(input, actual_output, expected_output, context)` | `TestCase` with `input`, `expected_output`, `context` |
| `assert_test(test_case, metrics)` | `TestCase.graders` references suite-level `Grader` definitions |
| `FaithfulnessMetric(threshold=0.7)` | `Grader` with `type: "llm_judge"`, `params.threshold: 0.7` |
| `AnswerRelevancyMetric` | `Grader` with `type: "semantic_similarity"` |
| `ConversationalTestCase` | `TestCase` with `input` as array of turns |

### From Promptfoo

| Promptfoo | EvalPort |
|-----------|----------|
| `vars` object | `input` + `context` |
| `assert` array | Inline graders or suite-level grader references |
| `{ type: "equals", value: "..." }` | `{ type: "exact_match", params: {} }` with `expected_output` |
| `{ type: "contains-json" }` | `{ type: "json_schema" }` |
| `{ type: "ic", value: "..." }` | `{ type: "llm_judge", params: { prompt: "..." } }` |

### From OpenAI Evals

| OpenAI Evals | EvalPort |
|--------------|----------|
| `test` object in `test_data.jsonl` | `TestCase` |
| `grader` field | `Grader` in suite |
| `modelgraded` spec | `Grader` with `type: "llm_judge"` |
| `sampling` config | Suite `config.provider` |

### From Inspect AI

| Inspect AI | EvalPort |
|------------|----------|
| `Sample(input, target)` | `TestCase` with `input`, `expected_output` |
| Solver functions | `Grader` with `type: "code"` or `type: "custom"` |
| `score()` return | `GraderResult` with `score`, `passed` |

---

## FAQ

**Q: Why not just use JSONL with agreed-upon field names?**

A: Field names alone don't capture grader semantics. A `semantic_similarity` grader needs a threshold, an embedding model, and a comparison method. A `llm_judge` needs a prompt template, a judge model, and an output schema. Without standardizing these, "agreeing on field names" still produces incompatible eval suites.

**Q: Why not extend OpenAI Evals' format?**

A: OpenAI Evals' format is tightly coupled to OpenAI's runner and grader implementation. It doesn't support arbitrary providers, custom graders, or agent evaluation. EvalPort is provider-agnostic and extensible.

**Q: How does EvalPort relate to OpenTelemetry GenAI?**

A: They are complementary. OpenTelemetry GenAI standardizes execution traces (spans for LLM calls, tool calls). EvalPort standardizes evaluation data (test cases, graders, results). A result set can reference an OTel trace ID for execution details.

**Q: How does EvalPort relate to MCP (Model Context Protocol)?**

A: MCP standardizes how AI applications discover and invoke tools. EvalPort standardizes how to evaluate AI systems. An agent eval suite can reference MCP tool names in `expected_tools` to verify an agent calls the right tools.

**Q: Why not wait for a standards body (ISO, IEEE, W3C) to define this?**

A: Standards bodies move slowly (2-5 years). The LLM eval ecosystem is evolving monthly. EvalPort follows the IETF "rough consensus and running code" model — ship a useful spec with reference implementations, iterate based on adoption, and submit to a standards body once the format is proven.

**Q: What about non-English evaluations?**

A: EvalPort is language-agnostic. Test inputs, expected outputs, and grader prompts can be in any language. The `metadata.language` field (optional) can indicate the primary language of a suite.

**Q: Can EvalPort handle multi-turn conversational evaluation?**

A: Yes. The `input` field accepts an array of strings representing conversational turns. For structured conversation (with roles), use `metadata.conversation` with `{ "role": "user", "content": "..." }` objects.

**Q: How are costs tracked?**

A: The result `metadata` field can include `openeval.cost` with token counts and estimated cost. This is optional and runner-dependent, as pricing varies by provider.

**Q: What if my grader type isn't in the standard set?**

A: Use `type: "custom"` with a `handler` string that identifies your grader. Runners that don't recognize the handler will skip it gracefully. You can register custom grader types in the extensions registry.

**Q: Is EvalPort tied to any specific LLM provider?**

A: No. EvalPort is provider-agnostic. The `provider` field specifies which model to use, and `api_base` supports self-hosted models. Graders that use LLMs (like `llm_judge` and `semantic_similarity`) specify their own model independently of the system under test.

---

## Appendix A: Grader Type Reference

### exact_match
```json
{ "type": "exact_match", "params": { "ignore_case": false, "trim_whitespace": true } }
```
Compares `actual_output` to `expected_output` as strings. Score is 1.0 on match, 0.0 otherwise.

### contains
```json
{ "type": "contains", "params": { "substring": "Paris", "ignore_case": false } }
```
Checks if `actual_output` contains `substring`. Score is 1.0 if found, 0.0 otherwise.

### regex
```json
{ "type": "regex", "params": { "pattern": "^\\d{4}-\\d{2}-\\d{2}$", "flags": "" } }
```
Tests `actual_output` against the regex pattern. Score is 1.0 on match, 0.0 otherwise.

### semantic_similarity
```json
{ "type": "semantic_similarity", "params": { "model": "text-embedding-3-small", "threshold": 0.85 } }
```
Computes cosine similarity between embeddings of `actual_output` and `expected_output`. Score is the similarity value. Passed if score >= threshold.

### llm_judge
```json
{ "type": "llm_judge", "params": {
    "model": "gpt-4o",
    "prompt": "Evaluate if {output} answers {input} correctly. Expected: {expected}. Return JSON.",
    "temperature": 0.0,
    "schema": { "type": "object", "properties": { "score": { "type": "number" }, "reason": { "type": "string" } }, "required": ["score", "reason"] }
}}
```
Uses an LLM to evaluate the output. The prompt supports `{input}`, `{output}`, `{expected}`, `{context}` substitutions. The judge's response is parsed according to `schema`; the `score` field provides the numeric score.

### json_schema
```json
{ "type": "json_schema", "params": { "schema": { "type": "object", "properties": {} }, "strict": true } }
```
Validates `actual_output` (parsed as JSON) against the provided JSON Schema. Score is 1.0 if valid, 0.0 if invalid.

### json_path
```json
{ "type": "json_path", "params": { "path": "$.status", "expected": "success", "operator": "eq" } }
```
Parses `actual_output` as JSON, extracts the value at `path` (JSONPath), and compares it to `expected` using `operator`. Score is 1.0 if the comparison passes, 0.0 otherwise.

### code
```json
{ "type": "code", "params": {
    "language": "python",
    "source": "def grade(input, output, expected, context):\n    return 1.0 if output.strip() == expected.strip() else 0.0",
    "timeout_ms": 5000
}}
```
Executes a grading function. The function receives `input`, `output`, `expected`, and `context` as arguments and returns a numeric score. MUST be sandboxed.

### human
```json
{ "type": "human", "params": { "instructions": "Rate the helpfulness of the response from 1-5." } }
```
Defers scoring to a human reviewer. The result set records a placeholder until the human review is completed.

### custom
```json
{ "type": "custom", "params": { "handler": "com.example.my_grader", "my_param": "value" } }
```
A framework-specific grader. The `handler` identifies the implementation. Unrecognized handlers are skipped gracefully.

---

## Appendix B: Reserved Metadata Keys

| Key | Scope | Description |
|-----|-------|-------------|
| `openeval.profile` | Suite | Usage profile (`rag`, `agent`, `safety`, etc.) |
| `openeval.pii_flags` | TestCase, Result | Fields containing PII |
| `openeval.deprecated` | Any | Indicates the field is deprecated |
| `openeval.cost` | Result | Token counts and estimated cost |
| `openeval.trace_id` | Result | OpenTelemetry trace ID |
| `openeval.language` | Suite | Primary language of the suite |
| `openeval.source` | Suite, TestCase | Origin of the data |

---

## Intellectual Property

EvalPort is released under the Apache 2.0 license. The specification, schemas, and reference implementations are free to use, modify, and distribute. No patent grants are implied. Contributors retain their copyrights under the terms of the Apache 2.0 license.

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-07-28 | Initial draft for community review |