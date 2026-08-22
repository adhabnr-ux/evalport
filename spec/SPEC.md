# EvalPort — The Open Evaluation Standard

**Version:** 1.0.0-rc.3  
**Status:** Release Candidate — Adopted by Inspect AI (merged), under active review by TruLens, implemented by 30 framework adapters, governance in place with 3 of 4 open RFC topics landed and 1 (suite/result signing) still tracked for community input  
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

**Type openness (normative):** `type` is not a closed enum. Any non-empty string is a valid grader type. The 11 types listed above are "well-known" — validators and runners give them standardized `params` validation and, where applicable, built-in execution support. Any other string (e.g. `"trulens_feedback"`, `"ragas_faithfulness"`) is a valid, framework-specific type name and is validated exactly like `custom`: `params.handler` is REQUIRED. This lets a document declare a framework-native grader type without inventing a fake `custom` wrapper, while still guaranteeing every non-standard grader carries enough information (`handler`) for a runner that doesn't recognize the type to skip it gracefully rather than guess at its semantics. This rule is enforced identically by `spec/schemas/grader.json` (via a catch-all `if type not in [...11 well-known values], then require params.handler` conditional) and by both reference SDKs (`sdk/python/openeval/validate.py`, `sdk/typescript/src/validate.ts`).

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
| `completed_at` | string (ISO 8601) | Timestamp this individual result was produced. Distinct from the `ResultSet`-level `completed_at` (whole-run finish time). Used as the merge tiebreaker for resumed/partial runs — see Extension Mechanism → Resumable Runs & Partial ResultSets. |
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

- `GraderResult.score` MUST be either `null`, or a number in the closed range [0.0, 1.0]. There is no `score_range` extension — every grader normalizes its native score to [0.0, 1.0] (or `null`; see Rule 6) before it is a valid EvalPort document. This is a hard requirement, not a convention: `spec/schemas/resultset.json` declares `score` as `{"type": ["number", "null"], "minimum": 0, "maximum": 1}`, and both reference SDKs reject an out-of-range or non-numeric (including boolean) score.
- A grader whose native scoring scale is not already [0.0, 1.0] (e.g. a 1-5 Likert scale, a raw cosine-similarity value that can be negative, a framework-specific 0-100 score) MUST clamp/normalize it to [0.0, 1.0] for the `score` field. To preserve the original value for debugging or re-analysis, use the reserved `metadata.openeval.raw_score` key on the `GraderResult` (see Appendix B) rather than putting an out-of-range value in `score` itself.
- Pass/fail is determined by comparing the score to the grader's threshold (default threshold: 1.0 for exact match, specified via params for other types).

### 6. Result Consistency

- Every grader referenced by a test case MUST have a corresponding `GraderResult` in the result set, unless the grader was `skipped`.
- Skipped or not-yet-executed graders (e.g. `human` review pending, an `unsupported_grader_type`, a runner error before scoring) MUST be represented with `score: null` and `passed: false`. `score: null` means "not verified" — the grader did not produce a score, which is distinct from `passed: false` on a numeric score, which means "verified failing" (the grader ran and the output did not meet the threshold). Consumers MUST NOT treat a `null`-score result as equivalent to a scored failure when computing pass rates, aggregate statistics, or the suite-level `passed` field (see `metadata.openeval.aggregation` in Extension Mechanism) — a `null` score should either be excluded from the aggregate denominator or surfaced separately as "pending/unscored," per the aggregation strategy declared for the run.
- `GraderResult.type` is REQUIRED and MUST match the `type` of the grader it corresponds to (or, for an inline/ad-hoc grader, the type used to produce the result) — this is what lets a validator or downstream tool apply type-specific interpretation to `score`/`passed` without re-resolving the grader definition from the suite.

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

### Aggregation Extension (`metadata.openeval.aggregation`)

By default (Rule 6), a `Result.passed` is the strict logical AND of every non-skipped `GraderResult.passed` for that test case: if any grader failed, the test case failed. This default is intentionally simple and matches most frameworks' native semantics, but it does not fit every use case — some frameworks want a *weighted* combination of scores (e.g. a rubric where some criteria matter more than others), a *majority* vote across graders, or an *any-pass* semantic (at least one grader must pass, useful for "does at least one of these N acceptable answers match"). Rather than leave this as an unspecified gap (as earlier drafts of this document did — see `spec/CRITIQUE.md` item #1), the `metadata.openeval.aggregation` key formally specifies it.

`openeval.aggregation` MAY be set in a suite's top-level `metadata` (declaring the suite's default aggregation policy for every test case in it) and/or in a `Result`'s own `metadata` (overriding the policy for that one result). Its value is an object:

```json
{
  "openeval.aggregation": {
    "strategy": "weighted",
    "threshold": 0.7
  }
}
```

| `strategy` | Meaning | `threshold` |
|---|---|---|
| `all` (default) | `passed` is true iff every non-null-scored `GraderResult.passed` is true. Equivalent to omitting `openeval.aggregation` entirely. | Not used. |
| `any` | `passed` is true iff at least one non-null-scored `GraderResult.passed` is true. | Not used. |
| `majority` | `passed` is true iff more than half of the non-null-scored `GraderResult`s have `passed: true`. | Optional; overrides the 0.5 cutoff, e.g. `0.6` requires a 60% majority. |
| `weighted` | `passed` is true iff the weighted average of `GraderResult.score` (using each grader's `weight`, default 1.0, from its definition in the suite) is `>= threshold`. `GraderResult`s with `score: null` are excluded from both the numerator and the denominator, not treated as 0. | REQUIRED. A number in [0.0, 1.0]. |

In every strategy, a `GraderResult` with `score: null` (per Rule 6, "not verified" — skipped, pending, or errored) is excluded from the aggregation entirely rather than counted as a failure. A test case whose graders are *all* null-scored has no basis for a pass/fail verdict; runners MUST report such a case's `passed` as `false` and SHOULD surface it distinctly (e.g. via `metadata.openeval.aggregation_status: "unscored"`) so it is not silently conflated with a verified failure in downstream reporting.

`openeval.aggregation` changes only how `Result.passed` (and, by extension, any suite-level summary pass rate a runner computes) is derived from the individual `GraderResult`s — it never changes what an individual `GraderResult.passed`/`score` means, and it is never required: a document with no `openeval.aggregation` key uses the `all` default and is fully valid.

### Resumable Runs & Partial ResultSets (`metadata.openeval.partial`, `Result.completed_at`)

Results can already be written incrementally by any runner, but prior to this section the spec defined no way to mark a `ResultSet` as covering only part of its suite (e.g. a run interrupted by a crash, a rate limit, or a manual stop) or to merge two partial `ResultSet`s from the same interrupted run back together. Resolves [Discussion #10](https://github.com/adhabnr-ux/evalport/discussions/10), deferred from `spec/CRITIQUE.md` item #4 ("should be added in v1.1").

**Marking a `ResultSet` partial** needs no schema change — `ResultSet.metadata` already permits arbitrary keys:

```json
{ "metadata": { "openeval.partial": true } }
```

A `ResultSet` with no `openeval.partial` key, or `openeval.partial: false`, is assumed complete (covers every test case in its suite) — this is fully backward compatible with every `ResultSet` produced before this section existed.

**Merging two partial `ResultSet`s** for the same `run_id` needs a tiebreaker when both cover the same `test_case_id` with different results (e.g. a retried test case). `Result.completed_at` (optional, `date-time`, distinct from the `ResultSet`-level `completed_at` which marks when the *whole run* finished) is the field that makes this decidable:

1. For any `test_case_id` present in both partials, the `Result` with the later `completed_at` wins.
2. If either `Result` is missing `completed_at` (an older or non-conforming producer), a merge tool MUST NOT guess an ordering — it SHOULD reject the merge and require the caller to specify precedence explicitly. Silently picking a default order for untimestamped partials produces a confidently-wrong merged `ResultSet` with no way to detect it after the fact, the same failure shape Rule 6 already guards against at the individual-`GraderResult` level.
3. The merged `ResultSet` SHOULD drop `openeval.partial` (or set it `false`) only once it genuinely covers every `test_case_id` in the suite — a merge of two partials that still leaves gaps is itself still partial.

This section defines the convention; it does not mandate a specific CLI merge command or its exact interface — that's a reasonable follow-up for whichever runner or the `evalport-cli` package wants to implement it, not something this spec revision blocks on. See `spec/conformance/fixtures/partial_resultset_resumable_run.json` for a worked example.

### Judge Hardening Self-Report (`metadata.openeval.judge_hardening`)

Resolves [Discussion #11](https://github.com/adhabnr-ux/evalport/discussions/11) ("Should `llm_judge` injection mitigations be a MUST, not a SHOULD?"), deferred from `spec/CRITIQUE.md` item #3 ("partially fixed"). Structured output, delimiting untrusted content, and output-length caps remain SHOULDs (not MUSTs) for `llm_judge` graders — see Security Considerations → Prompt Injection in Graders — because a spec-level MUST would need to either standardize prompt assembly itself (out of scope: every framework's judge prompt is different) or promote one reference implementation's behavior to the required one before any alternate implementation has been confirmed to match it.

Instead, a runner executing an `llm_judge` grader MAY self-report which mitigations it actually applied on the corresponding `GraderResult.metadata`:

```json
{ "metadata": { "openeval.judge_hardening": "structured_output+delimited+length_capped" } }
```

The value is a free-text, `+`-joined set of mitigation names — not a schema-enforced enum, since the mitigations worth naming will grow over time and standardizing the name set itself is a separate, smaller question from whether self-reporting is useful at all. This needs no schema change (`GraderResult.metadata` already permits arbitrary keys) and mirrors a pattern that already independently emerged across several shipped adapters for the analogous problem of an opaque judge internals: `giskard-openeval-adapter` and `llamaindex-openeval-adapter` both document, rather than fabricate, a judge's actual prompt/model when the source framework doesn't expose one directly. `openeval.judge_hardening` is the same "state honestly what you know, don't assert what you don't" shape, applied specifically to injection-hardening claims. A runner that claims a mitigation without applying it is simply lying in its own metadata — self-report is not a substitute for a runner actually being hardened, only a way to make that fact inspectable after the run. `spec/conformance/fixtures/judge_hardening_self_report.json` confirms the *convention itself* validates cleanly (a `GraderResult` carrying this key is spec-valid); it does not and cannot verify that a runner's claimed mitigation actually held under a real injection attempt, since that's runtime grading behavior, not document structure — see `spec/conformance/README.md`'s "What this doesn't cover (yet)" for that gap.

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

These remain SHOULDs rather than MUSTs — see [Discussion #11](https://github.com/adhabnr-ux/evalport/discussions/11) for why promoting them to MUST isn't straightforward (it would require either standardizing prompt assembly itself or promoting one reference implementation's behavior ahead of independent confirmation). A runner MAY self-report which of these it actually applied via the `metadata.openeval.judge_hardening` key on the `GraderResult` — see Extension Mechanism → Judge Hardening Self-Report — making the claim inspectable downstream even though the spec doesn't mandate the mitigations themselves.

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
7. **Conformance test suite** — `spec/conformance/` (resolves [Discussion #9](https://github.com/adhabnr-ux/evalport/discussions/9)): portable JSON fixtures, each a `(document, expected valid/invalid)` pair independently checked against both the JSON Schema files and the Python SDK's hand-rolled validator, so a conformance implementation in any language — not just the two reference SDKs — can test against the same fixtures without depending on this repo's code.

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
| `openeval.raw_score` | GraderResult | The grader's native, pre-normalization score (e.g. a 1-5 Likert value, a raw cosine similarity that may be negative, a framework's native 0-100 score) preserved for debugging/re-analysis when `score` had to be clamped or rescaled into [0.0, 1.0] to satisfy Validation Rule 5. Type and range are grader-specific and unconstrained by this spec. Adapter authors: emit this whenever your source framework's score isn't already [0.0, 1.0] — several shipped EvalPort adapters (e.g. for frameworks whose graders return confidence scores or Likert ratings) already follow this convention. |
| `openeval.aggregation` | EvalSuite (default), Result (override) | Declares the pass/fail aggregation strategy across a test case's `GraderResult`s when the default strict-AND-of-all-graders semantic doesn't fit. See Extension Mechanism → Aggregation Extension for the full `{"strategy": ..., "threshold": ...}` schema. |
| `openeval.aggregation_status` | Result | Set by a runner to `"unscored"` when every `GraderResult` for a test case has `score: null`, so downstream reporting doesn't conflate "no grader produced a verdict" with "a grader ran and failed." See Validation Rule 6. |
| `openeval.partial` | ResultSet | `true` if this `ResultSet` covers only part of its suite (e.g. an interrupted run). See Extension Mechanism → Resumable Runs & Partial ResultSets. |
| `openeval.judge_hardening` | GraderResult | Free-text, `+`-joined self-report of which prompt-injection mitigations a runner actually applied to an `llm_judge` grader (e.g. `"structured_output+delimited"`). See Extension Mechanism → Judge Hardening Self-Report and Security Considerations → Prompt Injection in Graders. |

---

## Governance

EvalPort is currently stewarded by its original author ([@adhabnr-ux](https://github.com/adhabnr-ux)) as spec lead, with direct write access extended to contributors who've shipped real, tested work against the spec — see [`CONTRIBUTORS.md`](https://github.com/adhabnr-ux/evalport/blob/main/CONTRIBUTORS.md) for who that is today. This is a pre-1.0 project; governance is deliberately lightweight right now and will formalize (a named working group, documented voting, a defined path from "collaborator" to "maintainer") as the contributor base grows past what one spec lead can review directly. If you think that formalization is overdue, [open a Discussion](https://github.com/adhabnr-ux/evalport/discussions) and say so — this section is itself subject to the RFC process below.

**How a spec change happens**, restated here in full rather than only in `.github/CONTRIBUTING.md`, since a document calling itself an RFC should describe its own process:

1. Open a GitHub Discussion in the **Ideas** category, titled `[Spec Change] <short description>`. State the problem, the proposed change, and its impact on backward compatibility.
2. A two-week comment period. This is where reach matters more than authority — a well-reasoned objection from a first-time contributor carries the same weight as one from a collaborator.
3. If rough consensus emerges, the change is implemented in a PR against `spec/SPEC.md` (and mirrored to the root `SPEC.md`), the JSON Schemas, and both reference SDKs together — a spec change that doesn't touch the SDKs' validators isn't actually specified, it's aspirational.
4. Changes that break backward compatibility (see the Versioning and Backward Compatibility sections above) require sign-off from the spec lead regardless of comment-period consensus, since a breaking change affects every adapter listed in the README, not just the proposer.

**How to become a collaborator:** there's no application process. In practice it has gone: ship a real adapter or converter (tested against the actual `validate_suite()`/`validate_result_set()`, not a mock), engage substantively on an issue or PR, and get invited. `CONTRIBUTORS.md` is the record of who's done that so far — it's a low bar in the sense that anyone can clear it, and a real one in the sense that a merged, tested PR is what clears it, not a comment.

---

## Open Design Questions — RFC Topics We Need Help With

The self-critique in [`spec/CRITIQUE.md`](https://github.com/adhabnr-ux/evalport/blob/main/spec/CRITIQUE.md) flags several items as deliberately deferred rather than resolved. Rather than let those sit as prose nobody acts on, each has an open Discussion where the actual design work happens. These are good entry points if you want to shape the spec itself rather than build a framework adapter — no prior EvalPort contribution required, just a considered opinion and, ideally, prior art from a comparable problem you've seen solved (or badly solved) elsewhere. Three of the four below have since landed a concrete spec change plus a reference implementation, exactly the way [Discussion #13](https://github.com/adhabnr-ux/evalport/discussions/13) (adapter packaging convention) previously did — the Discussion threads stay open for anyone who wants to refine or push back on the shipped design, they're just no longer *unimplemented*.

| Topic | Status | Discuss |
|---|---|---|
| Suite/result signing for integrity verification | Still open. No way to detect a publicly-hosted suite (e.g. anything in `benchmarks/`) was tampered with after publication. Explicitly deferred to v1.1/v2.0 in `CRITIQUE.md` #9. | [Discussion #8](https://github.com/adhabnr-ux/evalport/discussions/8) |
| Formal conformance test suite for runners | **Landed in 1.0.0-rc.3.** `spec/conformance/` — 8 portable JSON fixtures, each independently checked against the JSON Schemas and the Python SDK's hand-rolled validator, wired into CI. See Reference Implementation above. `CRITIQUE.md` #14 status updated from "Partial" to "Addressed." | [Discussion #9](https://github.com/adhabnr-ux/evalport/discussions/9) |
| Resuming interrupted runs and merging partial ResultSets | **Landed in 1.0.0-rc.3.** `Result.completed_at` (schema addition) plus `metadata.openeval.partial` (metadata convention, no schema change) — see Extension Mechanism → Resumable Runs & Partial ResultSets. `CRITIQUE.md` #4 status updated from "should be added in v1.1" to "Addressed." | [Discussion #10](https://github.com/adhabnr-ux/evalport/discussions/10) |
| `llm_judge` prompt-injection mitigations: MUST or SHOULD? | **Landed in 1.0.0-rc.3.** Mitigations stay SHOULDs (a spec-level MUST isn't tractable without standardizing prompt assembly), but a runner can now self-report which it applied via `metadata.openeval.judge_hardening` — see Extension Mechanism → Judge Hardening Self-Report. `CRITIQUE.md` #3 status updated from "Partially fixed" to "Addressed (self-report), MUST question itself resolved as won't-fix — see Discussion for reasoning." | [Discussion #11](https://github.com/adhabnr-ux/evalport/discussions/11) |

If you've got a fifth topic that belongs on this list — something the spec should address but doesn't yet — open a `[Spec Change]` Discussion for it directly; this table gets updated to reflect whatever's actually open, not maintained as a fixed roadmap.

---

## Intellectual Property

EvalPort is released under the Apache 2.0 license. The specification, schemas, and reference implementations are free to use, modify, and distribute. No patent grants are implied. Contributors retain their copyrights under the terms of the Apache 2.0 license.

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-rc.3 | 2026-08-22 | Landed 3 of the 4 Open Design Questions RFC topics as concrete, tested implementations (following the precedent set by Discussion #13's resolution), all verified against both reference SDKs' test suites in this revision. **Discussion #9 (conformance suite):** added `spec/conformance/` — a portable, language-agnostic fixture format (`fixtures/*.json`, each an `{expect, document}` pair) covering 8 real edge cases pulled from building the 30 shipped adapters plus the two RFC conventions below; a reference runner (`run.py`) verified against both the hand-rolled Python validator and the raw JSON Schema; a README; and a new `conformance-suite` CI job. **Discussion #10 (resumable runs / partial ResultSets):** added optional `Result.completed_at` (ISO 8601 timestamp) to `spec/schemas/resultset.json` and both SDKs' `Result` types, plus the schema-change-free `metadata.openeval.partial` convention for marking an in-progress ResultSet — see Extension Mechanism → Resumable Runs & Partial ResultSets. **Discussion #11 (judge hardening):** documented `metadata.openeval.judge_hardening`, a self-report convention (grounded in the pattern independently used by the giskard and llamaindex adapters) letting a runner declare which prompt-injection mitigations it applied to an `llm_judge` grader, since a spec-level MUST isn't tractable without standardizing prompt assembly — see Extension Mechanism → Judge Hardening Self-Report. Also fixed a drift bug: both SDKs' `OPENEVAL_VERSION` constant had been left at `"1.0.0-rc.1"` after the spec's own Version header advanced to `1.0.0-rc.2`, silently stamping every generated document with a stale spec version; now correctly `"1.0.0-rc.3"` in both. Added 3 new regression tests per SDK (`test_schema_consistency.py`, `schema-consistency.test.ts`) covering the new field and convention in both validation paths. Discussion #8 (suite/result signing) remains open. |
| 1.0.0-rc.2 | 2026-08-16 | Added a **Governance** section (spec lead, the RFC process restated in full inside the spec itself rather than only in CONTRIBUTING.md, and the actual path to becoming a collaborator) and an **Open Design Questions** table. The table links each item `spec/CRITIQUE.md` explicitly deferred to v1.1/v2.0 -- suite/result signing, a formal conformance test suite, resumable runs and partial-ResultSet merging, and whether `llm_judge` injection mitigations should be MUST instead of SHOULD -- to a live GitHub Discussion (#8-#11) where the actual design work happens, instead of leaving them as unlinked prose nobody could act on. |
| 1.0.0-rc.1 | 2026-08-16 | Promoted from draft to release candidate, reflecting real-world adoption: 20 shipped framework adapters, one merged third-party integration (Inspect AI, PR #4797), and one third-party integration under active maintainer review (TruLens, PR #2697). Substantive changes, all verified against the reference SDKs' test suites in this revision: (1) `version` fields now accept full semver 2.0.0 (prerelease + build metadata, e.g. `1.0.0-rc.1`) instead of only `X.Y.Z` or `X.Y.Z-draft` — fixed in `spec/schemas/suite.json`, `spec/schemas/resultset.json`, and both reference SDKs, which previously rejected this document's own version string. (2) Grader `type` is now formally documented and schema-enforced as open rather than a closed 11-value enum: any non-empty type string is valid and is validated like `custom` (`params.handler` required) unless it's one of the 11 well-known types, matching what the Custom Grader Types section already promised but the schema and SDKs didn't actually implement. (3) `spec/schemas/grader.json`'s per-type `allOf` conditionals now correctly require `params` to be present (previously a grader like `{"id": "g1", "type": "custom"}` with no `params` at all passed the JSON Schema despite being rejected by both SDKs — the conditionals constrained `params`'s shape but never required its presence). (4) Removed the never-defined `score_range` extension from Validation Rule 5; added the `openeval.raw_score` reserved metadata key so a grader's native (possibly out-of-[0,1]) score can be preserved when it must be clamped/normalized. (5) Formally specified the `openeval.aggregation` extension (`all`/`any`/`majority`/`weighted` strategies), resolving the gap `spec/CRITIQUE.md` had flagged as fixed while leaving the actual mechanism undefined. (6) Clarified Rule 6 to distinguish `score: null` ("not verified") from a scored failure ("verified failing"), and required `GraderResult.type`. (7) Added `sdk/python/tests/test_schema_consistency.py` and `sdk/typescript/tests/schema-consistency.test.ts`, which cross-validate every JSON Schema file against its corresponding hand-rolled SDK validator on every CI run, as a permanent regression guard against these two validation paths drifting apart again. |
| 1.0.0-draft | 2026-07-28 | Initial draft for community review |
