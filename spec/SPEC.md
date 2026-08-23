# EvalPort — The Open Evaluation Standard

**Version:** 1.0.0-rc.4  
**Status:** Release Candidate — Adopted by Inspect AI (merged), under active review by TruLens, implemented by 30 framework adapters, governance in place with all 4 of 4 open RFC topics landed as concrete spec changes with reference implementations  
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
┌───────────────────────────────────────────────────────┐
│                    Eval Suite                         │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Test Case │  │ Test Case │  │ Test Case │  ...    │
│  │  #1       │  │  #2       │  │  #3       │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                 │
│       └─────────────┼──────────────┘                 │
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
| `json_path` | Extracts a value via JSONPath and compares it | `path` (string), `expecte