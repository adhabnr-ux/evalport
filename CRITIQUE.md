# OpenEval — Hostile Standards Committee Critique

*This document simulates a hostile standards committee review. Each section identifies a weakness, explains why it could cause rejection, and proposes a fix. The spec has been iterated based on this critique.*

---

## 1. Ambiguity: What does "passed" mean when graders disagree?

### Attack
The spec says a Result's `passed` field is "overall pass/fail (all graders passed)" but doesn't define the aggregation logic when graders have different weights. Is it unanimous pass? Weighted majority? Any-fail = fail?

### Fix Applied
The spec now states: `passed` is `true` if and only if ALL grader results have `passed: true`. This is strict AND logic. Weighted scoring affects `avg_score` but not the boolean `passed` field. This is unambiguous and matches industry convention (a test suite fails if any test fails).

**Remaining concern:** Some frameworks use majority-vote or weighted thresholds. The spec addresses this via `metadata.openeval.aggregation` extension, but v1 only mandates AND logic.

---

## 2. Security: Code graders are an arbitrary code execution vector

### Attack
The `code` grader type allows arbitrary Python/JavaScript execution. A malicious eval suite shared on GitHub could contain a `code` grader that exfiltrates secrets, deletes files, or installs malware. This is a critical security risk that makes the spec unsuitable for enterprise adoption.

### Fix Applied
The spec now mandates:
1. Code graders MUST execute in an isolated sandbox (container, WASM, or restricted subprocess)
2. `timeout_ms` MUST be enforced (default 5000ms)
3. Code graders MUST be disabled by default in CI environments unless explicitly enabled via `--allow-code-graders`
4. The security considerations section prominently warns about this risk

**Remaining concern:** No spec can prevent all sandbox escapes. Runners bear implementation responsibility. The spec's role is to require the warning and the opt-in default.

---

## 3. Security: llm_judge prompt injection

### Attack
A malicious test case input could inject instructions into the `llm_judge` prompt, causing the judge to always return `score: 1.0`. This would make a broken system appear to pass all evals.

### Fix Applied
The spec recommends:
1. Judge prompts SHOULD use structured output (JSON schema) to constrain responses
2. Judge prompts SHOULD delimit the output being evaluated from instructions
3. Runners SHOULD cap judge LLM output length

**Remaining concern:** These are SHOULDs, not MUSTs. The spec cannot enforce prompt injection resistance at the data format level. This is a runner implementation concern, but the spec should at least make it a MUST to document the risk in runner documentation.

---

## 4. Scalability: No support for streaming or partial results

### Attack
For eval suites with 100,000+ test cases (e.g., MMLU-scale), loading the entire suite as a single JSON object is impractical. The spec mentions JSONL streaming but doesn't define how partial results are represented or how a runner signals "still running."

### Fix Applied
The spec now includes:
1. `test_cases_file` field in EvalSuite for external JSONL files
2. JSONL streaming format documented with example
3. Result sets can be written incrementally (each result is a complete object)

**Remaining concern:** The spec doesn't define a standard way to resume an interrupted run or merge partial result sets. This should be added in v1.1.

---

## 5. Implementation burden: Too many grader types

### Attack
The spec defines 11 grader types. A compliant runner must implement all of them, which is a significant burden. Small frameworks may refuse to adopt if they have to implement `json_path`, `code`, and `human` graders they don't need.

### Fix Applied
The spec now states:
1. Runners MUST handle all grader types (either execute or skip with `unsupported_grader_type`)
2. Skipping is not a failure — the grader result is recorded with `score: null` and `passed: false`
3. The `custom` type provides an escape hatch for framework-specific graders

**Result:** Minimum compliance = validate the document, execute the grader types you support, skip the rest. This lowers the barrier to entry.

---

## 6. Political barrier: Frameworks won't adopt a competitor's format

### Attack
DeepEval, Promptfoo, LangSmith, and Braintrust are competitors. They won't adopt a format that reduces their lock-in. Why would LangChain make it easier for users to leave LangSmith?

### Fix Applied
The adoption strategy is reframed:
1. OpenEval is not a replacement for framework formats — it's an interchange format
2. Frameworks keep their native format; OpenEval is an import/export layer
3. The value proposition is "get users FROM other tools" not "let users LEAVE your tool"
4. Converters are provided as a library, not a service — frameworks control the UX

**Remaining concern:** This is the single biggest adoption risk. The mitigation is leading with converters that make import easier than export.

---

## 7. Competing standards: OpenTelemetry GenAI covers some of this

### Attack
OpenTelemetry GenAI semantic conventions already define span attributes for LLM calls, including `gen_ai.response.id`, `gen_ai.usage.input_tokens`, etc. Why not extend OTel instead of creating a new format?

### Response
OpenTelemetry GenAI defines **execution traces** (what happened during a run). OpenEval defines **evaluation data** (what should have happened, and whether it did). They are complementary:
- OTel: "the LLM was called with these inputs at this time"
- OpenEval: "the LLM output should match this expected output, and here's the score"

OpenEval result sets can reference OTel trace IDs via `metadata.openeval.trace_id`. The spec explicitly positions this as complementary, not competing.

---

## 8. Ambiguity: `semantic_similarity` doesn't specify the embedding API

### Attack
The `semantic_similarity` grader says `model: "text-embedding-3-small"` but doesn't specify how to call the embedding API. Different runners might use different providers, producing different similarity scores for the same suite.

### Fix Applied
The spec clarifies:
1. The `model` field specifies the embedding model identifier
2. The `provider` field (optional) specifies which embedding API to use
3. If `provider` is not specified, the runner uses its default embedding provider
4. Results MUST include `metadata.provider` and `metadata.model` to record what was actually used

**Remaining concern:** The same suite run on different runners may produce different scores if they use different embedding providers. This is inherent to any spec that doesn't mandate a single provider. The solution is reproducibility metadata, not provider lock-in.

---

## 9. Missing: No standard for eval suite signing/integrity

### Attack
There's no way to verify that an eval suite hasn't been tampered with after publication. A benchmark dataset could be silently modified to favor a specific model.

### Response
This is a valid concern but out of scope for v1. The spec recommends:
1. Using `metadata.source` to track provenance
2. Using Git/version control for audit trail
3. A future `openeval.signature` extension for cryptographic signing

Signing is a v1.1 or v2.0 feature. It requires a key management story that would delay v1.0 indefinitely.

---

## 10. Missing: No standard for cost tracking

### Attack
Eval runs can be expensive (thousands of LLM calls). The spec has no standard field for cost.

### Fix Applied
The spec defines `metadata.openeval.cost` as a reserved key for cost tracking:
```json
"metadata": {
  "openeval.cost": {
    "input_tokens": 15000,
    "output_tokens": 8000,
    "estimated_cost_usd": 0.12
  }
}
```

This is optional and runner-dependent, but the field is standardized so dashboards can aggregate costs across runs.

---

## 11. Scalability: Large context arrays

### Attack
RAG eval suites may have context arrays with 50+ retrieved documents, each 10KB+. A 500-case suite with full context could be 250MB of JSON. This is impractical for version control and distribution.

### Fix Applied
The spec supports `test_cases_file` for external JSONL, and `metadata.openeval.context_url` (reserved key) for URL-based context references. Runners can fetch context at eval time rather than embedding it in the suite.

---

## 12. Why would companies refuse adoption?

| Company | Reason for Refusal | Mitigation |
|---------|-------------------|------------|
| **OpenAI** | Already has Evals format; OpenEval is provider-agnostic | OpenEval supports `model graded` type as alias for `llm_judge`; converters bridge the gap |
| **Anthropic** | Focused on MCP, not eval standards | OpenEval is complementary to MCP; no conflict |
| **LangChain** | LangSmith is their commercial product; portability reduces lock-in | Reframe: OpenEval brings users IN from other tools; export is a secondary feature |
| **Braintrust** | Already has a generic eval format | OpenEval alignment reduces friction for their users who also use other tools |
| **Google** | May push their own eval format via Vertex AI | OpenEval is provider-agnostic; Google's format can map to OpenEval |

---

## 13. Spec is too JSON-centric

### Attack
Many ML practitioners prefer YAML or CSV. The spec is JSON-first, which alienates data scientists who work in notebooks.

### Fix Applied
The spec explicitly supports YAML as an alternative serialization ("YAML files MUST be convertible to semantically identical JSON"). The Python SDK can read/write YAML. The CLI accepts `.yaml` and `.yml` files.

---

## 14. No conformance test suite

### Attack
A spec without a conformance test suite is just a suggestion. How do we know a runner is truly OpenEval-compliant?

### Fix Applied
The reference implementation includes:
1. JSON Schemas for all four document types
2. Validation libraries in TypeScript and Python
3. Example valid and invalid documents in the test suite

A formal conformance test suite (with graded test cases for each grader type) is planned for v1.1.

---

## Summary of Iterations

| Critique | Resolution | Status |
|----------|-----------|--------|
| Aggregation ambiguity | Defined as strict AND logic | ✅ Fixed |
| Code grader security | Sandbox requirement + opt-in default | ✅ Fixed |
| Prompt injection risk | SHOULDs for structured output + delimiters | ✅ Partially fixed |
| Streaming/large suites | JSONL + test_cases_file | ✅ Fixed |
| Too many grader types | Skip unsupported gracefully | ✅ Fixed |
| Political adoption barrier | Reframed as interchange, not replacement | ✅ Mitigated |
| OTel overlap | Positioned as complementary | ✅ Addressed |
| Embedding provider ambiguity | Provider field + reproducibility metadata | ✅ Fixed |
| Suite signing | Deferred to v1.1 | ⏳ Deferred |
| Cost tracking | Reserved metadata key | ✅ Fixed |
| Large context arrays | URL-based context references | ✅ Fixed |
| YAML support | Explicitly supported | ✅ Fixed |
| Conformance suite | Schemas + validation libs in v1; formal suite in v1.1 | ⏳ Partial |

---

## Final Verdict

The proposal is **substantially improved** after iteration. The remaining concerns (suite signing, formal conformance suite) are appropriately deferred to v1.1. The spec is ready for community review.

**Recommendation:** Approve for community draft status with a 90-day comment period.