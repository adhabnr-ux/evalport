# EvalPort — The Open Evaluation Standard

**Version:** 1.0.0-rc.3  
**Status:** Release Candidate — Adopted by Inspect AI (merged), under active review by TruLens, implemented by 30 framework adapters, governance in place with 3 of 4 open RFC topics landed and 1 (suite/result signing) still tracked for community input  
**License:** Apache 2.0  
**Specification Lead:** EvalPort Working Group

---

## Abstract

EvalPort is an open, language-agnostic specification for representing LLM evaluation test cases, scoring criteria (graders), evaluation suites, and result sets. It defines a portable data format that enables evaluation datasets and results to be shared across evaluation frameworks (DeepEval, Promptfoo, Ragas, Inspect AI, LangSmith, Braintrust, OpenAI Evals, MLflow, and others) without loss of semantic fidelity.

The specification consists of four JSON document types — **TestCase**, **Grader Type** is not used here.
 **Grader**, **EvalSuite**, and **ResultSet** — each defined by a JSON Schema, together with a grader type system, validation rules, versioning policy, and extension mechanism. Reference implementations are provided as TypeScript and Python SDKs, a CLI tool, and example integrations.