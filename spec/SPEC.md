# EvalPort — The Open Evaluation Standard

**Version:** 1.0.0-rc.3  
**Status:** Release Candidate — Adopted by Inspect AI (merged), under active review by TruLens, implemented by 30 framework adapters, governance in place with 3 of 4 open RFC topics landed and 1 (suite/result signing) still tracked for community input  
**License:** Apache 2.0  
**Specification Lead:** EvalPort Working Group

---

## Abstract

EvalPort is an open, language-agnostic specification for representing LLM evaluation test cases, scoring criteria (graders), evaluation suites, and result sets. It defines a portable data format that enables evaluation datasets and results to be shared across evaluation frameworks (DeepEval, Promptfoo, Ragas, Inspect AI, LangSmith, Braintrust, OpenAI Evals, MLflow, and others) without loss of semantic fidelity.

The specification consists of four JSON document types — **TestCase**, **Grader,**, **EvalSuite**, and **ResultSet** — each defined by a JSON Schema, together with a grader type system, validation rules, versioning policy, and extension mechanism. Reference implementations are provided as TypeScript and Python SDKs, a CLI tool, and example integrations.

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

The specification defines four document types: Eval Suite (containing Test Cases and Graders), and Result Set (containing Results and Summary Statistics). A Suite Configuration section holds provider and default settings. Running a suite produces a result set.
