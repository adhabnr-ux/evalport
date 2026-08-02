# EvalPort's Grader Type System: How It Works

*Published July 2026*

## The Core Design Decision

When designing EvalPort, the hardest question was: **how do we represent scoring criteria (graders) in a way that's both portable and expressive?**

Too simple, and the format can't capture real evaluation logic. Too complex, and no framework will implement it.

We landed on 11 grader types that cover 90%+ of real-world LLM evaluation needs, plus a `custom` escape hatch for the rest.

## The 11 Standard Types

### Simple Matchers (4 types)

These are deterministic, fast, and framework-independent:

- **`exact_match`** — string equality (with case/whitespace options)
- **`contains`** — substring check
- **`regex`** — regex pattern match
- **`json_schema`** — validates output against a JSON Schema

These cover most factual Q&A and structured output evaluation. No LLM calls needed — just string/JSON operations.

### Semantic Matchers (2 types)

- **`semantic_similarity`** — cosine similarity between embeddings of actual and expected output
- **`json_path`** — extract a value via JSONPath and compare it

`semantic_similarity` is the workhorse for "is this answer close enough?" evaluations. It needs an embedding model, but the threshold and model are specified in the grader params, so any runner can reproduce the score.

### LLM-Based Graders (2 types)

- **`llm_judge`** — an LLM evaluates the output against a rubric
- **`model graded`** — alias for `llm_judge` (OpenAI Evals compatibility)

`llm_judge` is the most powerful and most dangerous grader type. It uses an LLM to score outputs, which means:

1. **Prompt injection risk** — a malicious test input could manipulate the judge
2. **Non-determinism** — the same output may get different scores on different runs
3. **Cost** — each test case requires an LLM call

The spec mitigates these by recommending structured output (JSON schema) for judge responses and delimited prompt templates.

### Specialized Types (3 types)

- **`code`** — execute a custom grading function (sandboxed, with timeout)
- **`human`** — defer to human review
- **`custom`** — framework-specific handler

## The Design Principles

### 1. Graders Are Self-Describing

A grader isn't just a type name — it carries its own parameters:

```json
{
  "type": "semantic_similarity",
  "params": {"model": "text-embedding-3-small", "threshold": 0.85}
}
```

This means any runner can execute the grader without external configuration. The threshold, model, and comparison method are all in the data.

### 2. Unknown Types Don't Break Execution

If a runner encounters a grader type it doesn't support, it MUST skip it gracefully:

```json
{"grader_id": "gr_1", "type": "custom", "score": null, "passed": false, "metadata": {"skip_reason": "unsupported_grader_type"}}
```

This is critical for interoperability. A suite with a `code` grader can still be imported by a runner that doesn't support code execution — the other graders still run.

### 3. Scores Are Normalized to [0.0, 1.0]

All grader scores MUST be in [0.0, 1.0]. This enables:
- Cross-grader comparison
- Weighted aggregation
- Universal pass/fail thresholds

### 4. Pass/Fail Is Strict AND Logic

A test case passes only if ALL graders pass. This matches the convention from traditional software testing — a test suite fails if any test fails.

## How `llm_judge` Works in Practice

The `llm_judge` grader is the most complex type. Here's a real-world example:

```json
{
  "id": "gr_faithfulness",
  "type": "llm_judge",
  "params": {
    "model": "gpt-4o",
    "prompt": "Given the context: {context}\nAnd the output: {output}\nIs the output fully supported by the context? Return JSON: {\"score\": 0.0-1.0, \"reason\": \"...\"}",
    "temperature": 0.0,
    "schema": {
      "type": "object",
      "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"}
      },
      "required": ["score", "reason"]
    }
  }
}
```

Key features:
- **`{context}`, `{output}`, `{input}`, `{expected}` substitutions** — the runner replaces these tokens with the test case's values
- **`schema`** — constrains the judge's response to a JSON object with a `score` field
- **`temperature: 0.0`** — maximizes determinism

The judge's response is parsed according to `schema`, and the `score` field provides the numeric score. The `reason` field is stored in the grader result's `metadata` for debugging.

## The `custom` Escape Hatch

For framework-specific graders that don't fit any standard type:

```json
{"type": "custom", "params": {"handler": "deepeval:FaithfulnessMetric", "threshold": 0.7}}
```

Runners that recognize the handler execute it. Runners that don't skip it. This lets frameworks preserve their native grading logic while still participating in the EvalPort ecosystem.

## What's Next for Graders

v1.1 will add:
- **`multi_judge`** — run multiple judge prompts and aggregate
- **`human_in_loop`** — LLM judge with human override for borderline cases
- **`cost_aware`** — score that factors in token cost

But v1.0 is intentionally minimal. We add grader types only when 2+ frameworks implement them. Standards should emerge from practice, not precede it.

---

*Read the full spec at [SPEC.md](https://github.com/openeval/openeval/blob/main/spec/SPEC.md). Try the SDKs at [npm](https://www.npmjs.com/package/@evalport/sdk) and [PyPI](https://pypi.org/project/openeval/).*
