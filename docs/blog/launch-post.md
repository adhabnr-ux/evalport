# EvalPort: Why LLM Evaluation Needs a Standard Format

*Published July 2026*

## The Problem

If you've worked with LLMs for more than a month, you've hit this wall: you build an evaluation suite in DeepEval, your colleague uses Promptfoo, and your enterprise team standardized on LangSmith. Your 500-case RAG evaluation dataset is locked in one framework's format. You can't share it. You can't compare results. You can't switch tools without rewriting everything.

This isn't a minor inconvenience. It's a structural problem that's holding back the entire AI evaluation ecosystem:

1. **Eval datasets are not portable.** A test suite built in DeepEval cannot be run in Promptfoo without manual conversion.
2. **Graders are not interoperable.** Each framework represents scoring criteria differently — DeepEval uses metric classes, Promptfoo uses assertion objects, Inspect uses solver functions.
3. **Results cannot be compared.** Results from LangSmith, Braintrust, and Arize use incompatible schemas.
4. **Vendor lock-in.** Teams that invest in building eval datasets face switching costs that lock them in.
5. **No shared benchmarks.** The community can't publish reproducible benchmark datasets that work across frameworks.

## The Solution

**EvalPort** is an open, language-agnostic specification for representing LLM evaluation test cases, scoring criteria (graders), evaluation suites, and result sets.

It defines four standard document types:

- **TestCase** — a single eval input with expected output and grader references
- **Grader** — a scoring criterion (exact_match, semantic_similarity, llm_judge, etc.)
- **EvalSuite** — a collection of test cases and shared graders
- **ResultSet** — results from running a suite, with summary statistics

The key insight: **graders carry their own semantics.** A `semantic_similarity` grader specifies its threshold and model. An `llm_judge` grader specifies its prompt template and output schema. This means an eval suite isn't just a list of questions — it's a complete, self-describing evaluation that any framework can execute.

## Why Now?

The LLM eval ecosystem has matured to the point where fragmentation is actively harmful:

- **10+ major eval frameworks** exist (DeepEval, Promptfoo, Ragas, Inspect AI, LangSmith, Braintrust, OpenAI Evals, MLflow, and more)
- **MCP** (Model Context Protocol) standardized tool discovery — eval is the next layer
- **OpenTelemetry GenAI** is standardizing traces — eval data is the missing piece
- **Enterprise adoption** requires reproducibility and cross-tool comparison

The ecosystem is ready for a standard. EvalPort fills the gap.

## What's Included

The v1.0.0-rc.1 release includes:

- **Full specification** (SPEC.md) — 1000+ lines covering all aspects of the format
- **4 JSON Schemas** — machine-readable validation for all document types
- **TypeScript SDK** — `@evalport/sdk` on npm
- **Python SDK** — `openeval` on PyPI
- **CLI tool** — validate, convert, init, summary commands
- **REST API** — reference server for storing and serving eval suites
- **Converters** — import from Promptfoo, DeepEval, Inspect AI, and OpenAI Evals
- **11 grader types** — covering 90%+ of real-world evaluation needs
- **Examples** — basic Q&A, RAG, agent, multi-turn, and safety eval suites

## How It Works

Here's a simple EvalPort suite:

```json
{
  "version": "1.0.0",
  "id": "my_qa_eval",
  "graders": [
    {"id": "gr_exact", "type": "exact_match", "params": {"ignore_case": true}},
    {"id": "gr_semantic", "type": "semantic_similarity", "params": {"threshold": 0.85}}
  ],
  "test_cases": [
    {"id": "tc_1", "input": "What is the capital of France?", "expected_output": "Paris", "graders": ["gr_exact", "gr_semantic"]}
  ],
  "config": {"provider": {"model": "gpt-4o", "temperature": 0.0}}
}
```

Any compliant eval runner can import this suite, execute it, and produce a ResultSet — regardless of the framework's native format.

## Not a Replacement — an Interchange Format

EvalPort is not trying to replace DeepEval, Promptfoo, or any other framework. It's an **interchange format** — like JSON for data, or OpenAPI for APIs. Frameworks keep their native formats; EvalPort is the bridge that lets them talk to each other.

The value proposition for framework maintainers: **your users can now import eval datasets from any other tool.** That's a feature, not a concession.

## Get Involved

- **Try it**: `npm install -g @evalport/cli && openeval init my-suite`
- **Read the spec**: [SPEC.md](https://github.com/openeval/openeval/blob/main/spec/SPEC.md)
- **Convert your existing evals**: `openeval convert promptfoo openeval config.json output.json`
- **Give feedback**: Open a GitHub Discussion
- **Adopt it**: Add `to_openeval()` and `from_openeval()` to your framework

EvalPort is Apache 2.0, community-driven, and ready for your feedback.

---

*EvalPort is a community project. We welcome contributions, feedback, and framework integrations. Visit [GitHub](https://github.com/openeval/openeval) to get started.*
