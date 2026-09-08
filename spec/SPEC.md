# EvalPort — The Open Evaluation Standard

**Version:** 1.0.0-rc.5  
**Status:** Release Candidate — listed in Inspect AI's community extensions directory (PR merged, a docs listing rather than a native code integration); a standalone `to_openeval()`/`from_openeval()` module for TruLens has been reviewed and approved by a TruLens maintainer but is not yet merged (blocked on CI/rename, see spec/ADOPTION.md); 46 standalone adapter packages exist in this repo's own `adapters/` directory, built by the EvalPort maintainer against each framework's public data shapes — these are standalone packages in this repo, not adoptions by the upstream frameworks; governance in place with all 5 of 5 open RFC topics landed as concrete spec changes with reference implementations  
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
| `isolation` | string | Trial isolation mode for repeated attempts in this `ResultSet`'s `results` — an open string (`"fresh"`/`"shared"` conventional, not exhaustive). Declared once per `ResultSet`, not per `Result` — a producer that genuinely mixes isolation modes SHOULD emit separate `ResultSet`s instead. See Extension Mechanism → Repetition & Attempt Tracking. |
| `group` | object | **PROPOSED, [Discussion #45](https://github.com/adhabnr-ux/evalport/discussions/45), not yet finalized.** Membership in a named group of sibling `ResultSet`s (a sweep, a mutation-testing run, a multi-model comparison). `group.group_id` (string) is required when `group` is present; `group.role`/`group.label` (strings) and `group.sequence` (integer ≥ 0) are optional. See Extension Mechanism → Grouped/Sibling ResultSets. |
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
| `attempt` | integer (>= 1) | 1-indexed repetition number for this `test_case_id` within this `run_id`; ascending = observation order. Absent means single-attempt. Forms the `(test_case_id, run_id, attempt)` join key for repeated trials — see Extension Mechanism → Repetition & Attempt Tracking. |
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
- `(test_case_id, run_id, attempt)` MUST be unique across a `ResultSet`'s `results[]` whenever `Result.attempt` is present — see Extension Mechanism → Repetition & Attempt Tracking. This check is a no-op for any `ResultSet` that doesn't use `attempt`.

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

### Suite/ResultSet Signing (`spec/tools/verify_signature.py`)

Resolves [Discussion #8](https://github.com/adhabnr-ux/evalport/discussions/8) ("Suite/result signing for integrity verification"), deferred from `spec/CRITIQUE.md` item #9 ("out of scope for v1, ... a v1.1 or v2.0 feature"). The problem: nothing about the EvalPort document format itself lets a consumer detect that a publicly-hosted suite (the running example throughout that discussion, and throughout this section, is `benchmarks/`) was silently modified after publication — `metadata.source` is an unverified string, and Git history provides an audit trail only for someone who trusts the specific clone they're looking at.

**This is a signing *convention*, not a schema change.** No document field carries a signature, and no validator — `validate_suite()`, `validate_result_set()`, the JSON Schemas, the conformance suite — checks for one. Signing is optional, stays optional indefinitely (the same treatment as `metadata.openeval.cost`), and is scoped specifically to the "was this tampered with in transit or at rest" threat model, which mostly matters for suites redistributed outside a direct clone of their publisher's repo. A `ResultSet` you generate and consume entirely within your own CI has no need for this.

**The mechanism, following the reasoning laid out in Discussion #8's own comment thread:**

- **Detached, not embedded.** A signed artifact is the published file (e.g. `benchmarks/gsm8k/gsm8k.json`) plus a separate [Sigstore bundle](https://docs.sigstore.dev/about/bundle/) file alongside it, named `<filename>.sigstore.json` (e.g. `benchmarks/gsm8k/gsm8k.json.sigstore.json`). Embedding a signature inside the document it covers creates a chicken-and-egg problem — the field holding the signature would have to be excluded from what gets hashed, which is exactly the kind of subtlety a detached bundle avoids by construction.
- **Raw published bytes, not a canonical form.** The bundle signs the artifact's exact bytes as published — no JSON canonicalization step (e.g. JCS/RFC 8785). This means a byte-identical re-serialization with different whitespace needs a fresh signature, but it also means no canonicalizer implementation is required in any EvalPort-consuming language, and there is no room for two implementations' canonicalizers to quietly disagree about what was "really" signed — a subtly-wrong canonicalizer is a uniquely bad place for a bug, since it could either reject a validly-signed suite or, worse, be tricked into accepting a tampered one.
- **Sigstore keyless signing, not a project-managed key.** Signing uses [Sigstore](https://www.sigstore.dev/)'s keyless flow: this repo's GitHub Actions release workflow (`.github/workflows/ci.yml`'s `sign-benchmarks` job) exchanges its OIDC token for a short-lived Fulcio-issued certificate — no long-lived private key exists anywhere for anyone to generate, protect, leak, or rotate. This is the same OIDC trust root this repo already uses for PyPI/npm Trusted Publishing (`publish-pypi` / `publish-npm`, also in `ci.yml`) — extending that same trust to benchmark-suite signing is a small conceptual step, not new infrastructure, and needs no key-management or revocation story. It's also the same approach npm provenance attestations and most modern SBOM/SLSA tooling converged on, for the same reason: a long-lived signing key is a standing liability a keyless flow eliminates entirely.
- **Identity, not just validity, is what's checked.** A valid signature alone only proves *someone* with *some* Sigstore-recognized identity signed the artifact — anyone can obtain a Fulcio certificate for their own identity and sign anything. The reference verifier requires the caller to state an expected identity (an exact GitHub Actions workflow reference, or a regex matching any release of this repo's `ci.yml`) and OIDC issuer; it refuses to treat "the signature checks out" as sufficient on its own.

**Reference verifier:** [`spec/tools/verify_signature.py`](https://github.com/adhabnr-ux/evalport/blob/main/spec/tools/verify_signature.py) — a standalone CLI/library (depends only on the `sigstore` PyPI package) that checks signature validity, Fulcio certificate chain validity against Sigstore's public root of trust, [Rekor](https://docs.sigstore.dev/logging/overview/) transparency-log inclusion, and the caller-specified identity policy, in one call:

```bash
python3 spec/tools/verify_signature.py verify benchmarks/gsm8k/gsm8k.json \
  --cert-identity-regex '^https://github\.com/adhabnr-ux/evalport/\.github/workflows/ci\.yml@refs/tags/.*$' \
  --cert-oidc-issuer https://token.actions.githubusercontent.com
```

See `spec/tools/README.md` for full usage, including why an identity check is mandatory (`--unsafe-skip-identity-check` exists only for debugging a bundle in isolation and is never the recommended path) and where a real release's signature bundles are published (attached to the GitHub Release as an archive, not committed to `main` — a benchmark file changes between releases, and its signature should reflect exactly the release it shipped in, not drift against an evolving `main`).

**Honest scope note on what's been verified so far.** `spec/tools/verify_signature.py`'s own test suite (`spec/tools/tests/test_verify_signature.py`) is real, not a mock — it verifies the script against genuine, independently-published Sigstore bundles vendored from another project's own test fixtures (see `spec/tools/tests/fixtures/NOTICE.md`), including a real GitHub Actions OIDC-signed artifact, checked against Sigstore's production infrastructure over the network. What this revision has *not* done is exercise `ci.yml`'s `sign-benchmarks` job against a real EvalPort release: minting a fresh Sigstore signature requires an interactive OIDC login (a browser-based identity-provider flow) that no sandboxed or headless development environment can perform non-interactively, so that job can only be genuinely exercised by this repo's own release CI, the first time a real release runs it. The verifier being independently proven correct against real Sigstore infrastructure is what makes it trustworthy to run in that job in the first place — but "the verifier works" and "this specific CI job is correctly wired" are two different claims, and only the first one is backed by a test run as of this revision.

### Repetition & Attempt Tracking (`Result.attempt`, `ResultSet.isolation`)

Resolves [Discussion #22](https://github.com/adhabnr-ux/evalport/discussions/22) ("RFC: repetition/attempt tracking in ResultSet"), raised externally via [issue #20](https://github.com/adhabnr-ux/evalport/issues/20) by AgentVerity's maintainer, deferred from `spec/CRITIQUE.md` item #15. Prior to this section, `ResultSet.results[]` documented "one result per test case" as a convention only — neither `resultset.json` nor `validate_result_set()` enforced it, so a producer emitting several `Result`s for the same `test_case_id` (LangSmith's `num_repetitions`, Promptfoo's per-test repeats, Inspect AI's `epochs` all do this upstream) was silently unaddressed: no field said *why* there were several, nothing joined them, and `summary` computation had no defined behavior when it happened.

**`Result.attempt`** (optional integer, `minimum: 1`) is the join key. It is a 1-indexed repetition number for a `test_case_id` within a `run_id`; ascending values are observation order (attempt 2 was observed after attempt 1), so consumers get a documented ordering guarantee instead of inferring one from `completed_at` or array position. Absent, or `1` with no sibling attempts, means a single-attempt result — every `ResultSet` produced before this section needs no change. `(test_case_id, run_id, attempt)` MUST be unique across `results[]` whenever `attempt` is present (Validation Rules → Uniqueness); a runner encountering a duplicate MUST reject the `ResultSet`, the same treatment `DUPLICATE_ID` already gets for a suite's test case/grader IDs.

**`ResultSet.isolation`** (optional string) records whether the repeats represented in this `ResultSet` ran in fresh sessions/model instances or a shared session/context — a fact that changes what a stability or flip-rate computation over those repeats can support statistically (independent trials vs. correlated ones). It is an **open string, not a closed enum**: `"fresh"` and `"shared"` are documented as conventional values, not an exhaustive list, so a new isolation strategy some other framework invents later never needs a spec change just to be nameable — the same reasoning [Discussion #22](https://github.com/adhabnr-ux/evalport/discussions/22) applied to keeping grader `type` open (see Custom Grader Types).

**Design decision: `isolation` lives on `ResultSet`, once, not on each `Result`.** This was the one genuinely open question after the rest of the design converged. Discussion #22 had converged on an open string, plus a refinement that — if `isolation` stayed per-`Result` — would require "one isolation value per repetition group" (every `Result` sharing `(test_case_id, run_id)` must agree). In the parallel issue #20 thread, weighing per-`Result` against `ResultSet`-level directly, AgentVerity's maintainer settled the placement question: *"AgentVerity's own evidence model declares isolation once per collection set, not per decision, because a window that mixes fresh and shared trials changes what the flip-rate interval supports at all. A producer who genuinely mixes isolation modes should emit two ResultSets rather than annotate per result, which keeps the summary statistics well-defined too. If a real per-result case appears later, an optional override can be added without breaking anything."* This spec adopts that placement as the implementation: a single `ResultSet`-level field matches AgentVerity's evidence model exactly, and it's simpler for a real reason beyond fewer bytes — a per-`Result` field needs its own consistency rule (do all `Result`s sharing `(test_case_id, run_id)` agree on `isolation`?) that a `ResultSet`-level value makes trivially true by construction, since there is only one value to check. Consistent with "if a real per-result case appears later" above, no per-`Result` override is added speculatively here; that stays a future, additive extension for if and when a genuine need for it shows up.

```json
{
  "version": "1.0.0",
  "suite_id": "suite_stability_eval",
  "run_id": "run_20260830_repeated",
  "started_at": "2026-08-30T09:00:00Z",
  "isolation": "fresh",
  "results": [
    { "test_case_id": "tc_001", "attempt": 1, "passed": true, "grader_results": [ { "grader_id": "gr1", "type": "exact_match", "score": 1.0, "passed": true } ] },
    { "test_case_id": "tc_001", "attempt": 2, "passed": true, "grader_results": [ { "grader_id": "gr1", "type": "exact_match", "score": 1.0, "passed": true } ] },
    { "test_case_id": "tc_001", "attempt": 3, "passed": false, "grader_results": [ { "grader_id": "gr1", "type": "exact_match", "score": 0.0, "passed": false } ] }
  ]
}
```

See `spec/conformance/fixtures/multi_attempt_resultset_valid.json` (a valid multi-attempt `ResultSet`) and `spec/conformance/fixtures/duplicate_attempt_collision_rejected.json` (a same-`(test_case_id, run_id, attempt)` collision correctly rejected) — contributed against these exact field names per AgentVerity's offer in Discussion #22.

### Grouped/Sibling ResultSets (`ResultSet.group`) — PROPOSED, not yet finalized

> **Status:** this section documents [Discussion #45](https://github.com/adhabnr-ux/evalport/discussions/45), open for comment. The schema addition, SDK validators, and conformance fixtures described here exist on a reference-implementation branch/PR referenced from that discussion — **not on `main`** — so they can be reviewed and tested without being mistaken for a landed spec change. This subsection will be rewritten in the past tense (matching Repetition & Attempt Tracking above) if and when the RFC concludes and actually merges, the same path [Discussion #22](https://github.com/adhabnr-ux/evalport/discussions/22) took to become the section above.

Grew out of [issue #36](https://github.com/adhabnr-ux/evalport/issues/36) ("No representation for grouped/sweep ResultSets with rollup semantics"), itself raised from a cross-project conversation in [AshwinUgale/muteval#44](https://github.com/AshwinUgale/muteval/issues/44). The gap: nothing in the schema relates one `ResultSet` to a set of sibling `ResultSet`s — `Repetition & Attempt Tracking` above fixed repeated trials *within* one `run_id`, but did nothing for grouping *across* several `run_id`s (a mutation-testing sweep's one-`ResultSet`-per-mutant, a hyperparameter grid search's one-`ResultSet`-per-trial, a multi-model comparison's one-`ResultSet`-per-model).

**`ResultSet.group`** (optional object, required sub-field `group_id`) is the proposed join key, modeled directly on precedent from six real systems that already solve part of this problem — W&B Sweeps (`Run.sweep_id`), MLflow nested runs (`mlflow.get_parent_run`), Stryker's `mutation-testing-report-schema` (per-mutant `status`, no rollup field), Optuna's `Study`/`FrozenTrial` (see below), promptfoo's `EvaluateResult.provider`/`CompletedPrompt.metrics` (see below), and Google Cloud's Vertex AI Vizier `Study`/`Trial` resource hierarchy, confirmed directly in the open-source `google/vizier` implementation behind it (see below) — all of which independently converged on the same shape: **the join key lives on the member and points at the group; the group-level rollup is computed by the consumer, not stored as a schema-mandated document.** A seventh system checked for the same reason, AWS SageMaker's hyperparameter tuning API, does **not** converge on this shape — it's discussed on its own terms below rather than left out for disagreeing. This spec deliberately follows the six-system precedent rather than also standardizing a separate rollup/manifest document — see Discussion #45 for the full reasoning, including why the informal alternative (relying on `suite_id` conventions) was rejected the same way an equivalent informal option was rejected for `isolation`.

- `group.group_id` (string, required when `group` is present) — identifier shared by every `ResultSet` in the group. An open, producer-chosen string (UUID, slug, timestamp-based, ...), the same design as `suite_id`/`run_id`.
- `group.role` (optional string) — this member's role or outcome within the group, e.g. `"mutant"`, `"seed"`, `"baseline"`, `"candidate"`. An **open string, not a closed enum**, for the same reason `isolation` isn't one (see above).
- `group.label` (optional string) — a human-readable name for this member, for display only, not a join key.
- `group.sequence` (optional integer, `minimum: 0`) — this member's 0-indexed position within the group, for producers that know the group's total size at emission time.

```json
{
  "version": "1.1.0",
  "suite_id": "billing-suite",
  "run_id": "mutant-017-run",
  "started_at": "2026-09-01T10:00:00Z",
  "isolation": "fresh",
  "group": {
    "group_id": "mutation-sweep-2026-09-01",
    "role": "mutant",
    "label": "mutant_017 (relational-operator-swap in billing.py:42)",
    "sequence": 17
  },
  "results": [
    { "test_case_id": "case_1", "attempt": 1, "passed": true, "grader_results": [ { "grader_id": "gr1", "type": "exact_match", "score": 1.0, "passed": true } ] },
    { "test_case_id": "case_2", "attempt": 1, "passed": false, "grader_results": [ { "grader_id": "gr1", "type": "exact_match", "score": 0.0, "passed": false } ] }
  ]
}
```

**Whether `sequence` is meaningful depends on the producer knowing the group's total size upfront — verified, not assumed, for the motivating case.** Reading `AshwinUgale/muteval`'s actual current `src/muteval/runner.py`: `run_mutation_testing()` calls `select_mutants()`, which fully materializes the mutant list via `generate_mutants()` — synchronously, before any per-mutant evaluation begins (whether run serially or via `ThreadPoolExecutor.map`, which preserves input order). So for mutation testing specifically, the total mutant count and each mutant's position genuinely are known before its `ResultSet` would be emitted; `sequence` is not dead weight there. A producer whose grouping strategy discovers members incrementally (e.g. a search that doesn't know its own trial count upfront) can simply omit `sequence` — it's optional for exactly that reason.

**On a standard `role` vocabulary for mutation testing specifically:** muteval's real per-mutant outcome (`MutantOutcome` in `runner.py`) is not a single flat status string the way Stryker's `Killed`/`Survived`/`NoCoverage`/... enum is — it's several orthogonal signals (`killed: bool`, `errored: bool`, `output_changed: Optional[bool]` distinguishing a real coverage gap from an inert/equivalent mutant, and a separate `severity: "high"|"medium"|"low"` ranking). Collapsing all of that into one closed `role` enum would either lose information or invent a combinatorial vocabulary nobody asked for. The module docstring's own framing — a mutant is *"killed"* (suite failed, caught the injected regression) or *"survives"* (suite still passed) or, on a harness error, *"errored"* — is the one dimension that maps cleanly to a single `role` value, so `"killed"` / `"survived"` / `"errored"` are documented here as a **conventional, non-enforced** starting vocabulary for mutation-testing producers; severity and inert-vs-real status stay better expressed via `group.label` or domain metadata, not folded into `role`. This was offered as a considered default pending confirmation from an actual muteval-side integration — confirmed, with a refinement, below.

**Confirmed by muteval's maintainer, with a refinement that sharpens the metadata split above.** [Replying directly on Discussion #45](https://github.com/adhabnr-ux/evalport/discussions/45#discussioncomment-18308435), AshwinUgale confirmed both the `sequence` and `role` reasoning above, then went further: muteval's "survived" state isn't one thing internally — a survivor is either a real coverage gap or an inert/equivalent mutant the tool excludes from its *effective* mutation score, so a consumer reconstructing that rollup from `role` alone would get muteval's raw score, "not the effective one" (AshwinUgale, Discussion #45). His conclusion tracks the split already proposed above rather than requiring a schema change: "'survived' needs to distinguish real-gap vs inert" (AshwinUgale, Discussion #45) if `role` is meant to support rollup reconstruction — precisely the job already assigned to domain `metadata`, not `role`, in the paragraph above.

Concretely, that means carrying the real-gap-vs-inert bit as free-form `metadata` on each `ResultSet`, reusing muteval's own field for it (`output_changed`, from `MutantOutcome` in `runner.py`) rather than inventing new spec vocabulary for something this schema already lets a producer express without any schema change:

```json
[
  {
    "version": "1.1.0",
