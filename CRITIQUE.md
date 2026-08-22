# EvalPort — Hostile Standards Committee Critique

*This document simulates a hostile standards committee review. Each section identifies a weakness, explains why it could cause rejection, and proposes a fix. The spec has been iterated based on this critique.*

---

## 1. Ambiguity: What does "passed" mean when graders disagree?

### Attack
The spec says a Result's `passed` field is "overall pass/fail (all graders passed)" but doesn't define the aggregation logic when graders have different weights. Is it unanimous pass? Weighted majority? Any-fail = fail?

### Fix Applied
The spec now states: `passed` is `true` if and only if ALL grader results have `passed: true`. This is strict AND logic by default (`openeval.aggregation` unset, or explicitly `{"strategy": "all"}`). This is unambiguous and matches industry convention (a test suite fails if any test fails).

**Resolved in 1.0.0-rc.1:** The `metadata.openeval.aggregation` extension referenced above is now fully specified (see SPEC.md, Extension Mechanism → Aggregation Extension), not just named. It defines four strategies — `all` (default), `any`, `majority` (with an optional custom cutoff), and `weighted` (using each grader's `weight` against a required `threshold`) — and specifies how `score: null` results are excluded from the aggregation rather than counted as failures. Frameworks that need majority-vote or weighted-threshold semantics can now express that declaratively in `metadata` instead of only via `avg_score`, which was previously an informational field with no normative effect on `passed`.

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

**Update (1.0.0-rc.3, [Discussion #11](https://github.com/adhabnr-ux/evalport/discussions/11)):** the MUST-vs-SHOULD question itself is resolved as won't-fix — a spec-level MUST isn't tractable without also standardizing prompt assembly, which is out of scope for a data-interchange format. What did land: `metadata.openeval.judge_hardening`, a reserved key a runner can set to self-report which of the three SHOULD mitigations it actually applied to a given `llm_judge` grader (see SPEC.md's Extension Mechanism → Judge Hardening Self-Report, and Security Considerations → Prompt Injection in Graders). This makes the "document the risk" ask above concrete and machine-checkable rather than left to runner docs, without claiming to solve prompt injection resistance itself — that remains a runner implementation concern, honestly out of this suite's structural-validation scope (see `spec/conformance/README.md`'s "What this doesn't cover" section).

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

**Update (1.0.0-rc.3, [Discussion #10](https://github.com/adhabnr-ux/evalport/discussions/10)):** addressed. `Result.completed_at` (new optional field, `spec/schemas/resultset.json`) timestamps each individual result as it's produced, and `metadata.openeval.partial` (a metadata convention, no schema change needed) marks a ResultSet as in-progress. Together they define a concrete merge algorithm for combining two partial ResultSets from the same `run_id` — last-write-wins per `test_case_id` keyed by `completed_at`, falling back to rejecting the merge if either side is missing a timestamp — documented in SPEC.md's Extension Mechanism → Resumable Runs & Partial ResultSets, with a conformance fixture (`spec/conformance/fixtures/partial_resultset_resumable_run.json`) and cross-SDK regression tests in both `test_schema_consistency.py` and `schema-consistency.test.ts`.

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
1. EvalPort is not a replacement for framework formats — it's an interchange format
2. Frameworks keep their native format; EvalPort is an import/export layer
3. The value proposition is "get users FROM other tools" not "let users LEAVE your tool"
4. Converters are provided as a library, not a service — frameworks control the UX

**Remaining concern:** This is the single biggest adoption risk. The mitigation is leading with converters that make import easier than export.

---

## 7. Competing standards: OpenTelemetry GenAI covers some of this

### Attack
OpenTelemetry GenAI semantic conventions already define span attributes for LLM calls, including `gen_ai.response.id`, `gen_ai.usage.input_tokens`, etc. Why not extend OTel instead of creating a new format?

### Response
OpenTelemetry GenAI defines **execution traces** (what happened during a run). EvalPort defines **evaluation data** (what should have happened, and whether it did). They are complementary:
- OTel: "the LLM was called with these inputs at this time"
- EvalPort: "the LLM output should match this expected output, and here's the score"

EvalPort result sets can reference OTel trace IDs via `metadata.openeval.trace_id`. The spec explicitly positions this as complementary, not competing.

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
| **OpenAI** | Already has Evals format; EvalPort is provider-agnostic | EvalPort supports `model graded` type as alias for `llm_judge`; converters bridge the gap |
| **Anthropic** | Focused on MCP, not eval standards | EvalPort is complementary to MCP; no conflict |
| **LangChain** | LangSmith is their commercial product; portability reduces lock-in | Reframe: EvalPort brings users IN from other tools; export is a secondary feature |
| **Braintrust** | Already has a generic eval format | EvalPort alignment reduces friction for their users who also use other tools |
| **Google** | May push their own eval format via Vertex AI | EvalPort is provider-agnostic; Google's format can map to EvalPort |

---

## 13. Spec is too JSON-centric

### Attack
Many ML practitioners prefer YAML or CSV. The spec is JSON-first, which alienates data scientists who work in notebooks.

### Fix Applied
The spec explicitly supports YAML as an alternative serialization ("YAML files MUST be convertible to semantically identical JSON"). The Python SDK can read/write YAML. The CLI accepts `.yaml` and `.yml` files.

---

## 14. No conformance test suite

### Attack
A spec without a conformance test suite is just a suggestion. How do we know a runner is truly EvalPort-compliant?

### Fix Applied
The reference implementation includes:
1. JSON Schemas for all four document types
2. Validation libraries in TypeScript and Python
3. Example valid and invalid documents in the test suite

A formal conformance test suite (with graded test cases for each grader type) is planned for v1.1.

**Update (1.0.0-rc.3, [Discussion #9](https://github.com/adhabnr-ux/evalport/discussions/9)):** addressed. `spec/conformance/` ships 8 portable, language-agnostic fixtures (`fixtures/*.json`, each a self-contained `{type, expect, document}` triple) covering real edge cases pulled from building the 30 shipped adapters — the null-vs-scored-failure distinction (Rule 6), score-range enforcement (Rule 5), the boolean-is-not-a-valid-score cross-language gotcha, custom/non-standard grader `params.handler` requirements — plus fixtures for the two conventions that landed alongside it (Discussions #10 and #11). A reference runner (`run.py`) checks every fixture against this repo's own hand-rolled Python validator and is wired into CI as a dedicated `conformance-suite` job; every fixture has also been independently verified against the raw JSON Schema files (`Draft202012Validator`), not just the hand-rolled path, so `expect.valid` reflects genuine agreement between both validation paths this project maintains. Any third-party implementation in another language can load these same fixture files and check its own validator's answer against `expect.valid` without depending on this repo's Python or TypeScript code — see `spec/conformance/README.md`, which also states plainly what this suite does *not* yet cover (CLI runtime behavior, cross-document referential rules beyond the existing checks).

---

## Summary of Iterations

| Critique | Resolution | Status |
|----------|-----------|--------|
| Aggregation ambiguity | Strict AND logic by default; `metadata.openeval.aggregation` now formally specifies `all`/`any`/`majority`/`weighted` as of 1.0.0-rc.1 | ✅ Fixed |
| Code grader security | Sandbox requirement + opt-in default | ✅ Fixed |
| Prompt injection risk | SHOULDs for structured output + delimiters; MUST-vs-SHOULD resolved won't-fix, `metadata.openeval.judge_hardening` self-report convention added as of 1.0.0-rc.3 ([Discussion #11](https://github.com/adhabnr-ux/evalport/discussions/11)) | ✅ Addressed (self-report); underlying SHOULDs stand |
| Streaming/large suites | JSONL + test_cases_file | ✅ Fixed |
| Too many grader types | Skip unsupported gracefully; type is an open string (not a closed enum) as of 1.0.0-rc.1, schema- and SDK-enforced identically | ✅ Fixed |
| Political adoption barrier | Reframed as interchange, not replacement; validated in practice by 30 shipped adapters and a merged Inspect AI PR | ✅ Mitigated |
| OTel overlap | Positioned as complementary | ✅ Addressed |
| Embedding provider ambiguity | Provider field + reproducibility metadata | ✅ Fixed |
| Suite signing | Deferred to v1.1; still open as of 1.0.0-rc.3 ([Discussion #8](https://github.com/adhabnr-ux/evalport/discussions/8)) | ⏳ Deferred |
| Cost tracking | Reserved metadata key | ✅ Fixed |
| Large context arrays | URL-based context references | ✅ Fixed |
| YAML support | Explicitly supported | ✅ Fixed |
| Resumable/partial runs | `Result.completed_at` + `metadata.openeval.partial` as of 1.0.0-rc.3 ([Discussion #10](https://github.com/adhabnr-ux/evalport/discussions/10)) | ✅ Addressed |
| Conformance suite | Schemas + validation libs in v1, cross-checked against each other in CI since 1.0.0-rc.1; formal portable conformance suite (`spec/conformance/`, 8 fixtures, CI-wired) shipped in 1.0.0-rc.3 ([Discussion #9](https://github.com/adhabnr-ux/evalport/discussions/9)) | ✅ Addressed |

---

## Final Verdict

The proposal is **substantially improved** after iteration. As of 1.0.0-rc.3, three of the four items this document originally flagged as deferred — the conformance suite, resumable/partial runs, and the judge-hardening self-report convention — have shipped as concrete, tested spec changes with reference implementations, following the same Discussion-then-PR process used for every other change in this log. The one item still genuinely open is suite/result signing (Discussion #8) — it needs either community input on a signing scheme or a spec-lead decision, and unlike the other three it can't be fully exercised in every environment (a genuine Sigstore/Fulcio signature requires real CI OIDC identity), so it's being scoped carefully rather than rushed. As of 1.0.0-rc.3 the spec is implemented by 30 shipped framework adapters, merged into Inspect AI's official community extensions (PR #4797), and under active review by TruLens (PR #2697).

**Recommendation:** Approve for release-candidate status. Promote to 1.0.0 final once the TruLens review concludes and no further breaking feedback surfaces from the community comment period.