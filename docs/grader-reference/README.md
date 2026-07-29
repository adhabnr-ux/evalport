# Grader Type Reference

OpenEval defines 11 standard grader types. Runners MUST handle all types (execute or skip gracefully).

---

## exact_match

String equality comparison between `actual_output` and `expected_output`.

```json
{"type": "exact_match", "params": {"ignore_case": false, "trim_whitespace": true}}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ignore_case` | boolean | false | Case-insensitive comparison |
| `trim_whitespace` | boolean | true | Trim whitespace before comparison |

**Score**: 1.0 on match, 0.0 otherwise.

---

## contains

Checks if `actual_output` contains a substring.

```json
{"type": "contains", "params": {"substring": "Paris", "ignore_case": false}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `substring` | string | yes | The substring to search for |
| `ignore_case` | boolean | no | Case-insensitive search (default: false) |

**Score**: 1.0 if found, 0.0 otherwise.

---

## regex

Matches `actual_output` against a regex pattern.

```json
{"type": "regex", "params": {"pattern": "^\\d{4}-\\d{2}-\\d{2}$", "flags": ""}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `pattern` | string | yes | RE2 regex pattern |
| `flags` | string | no | Regex flags |

**Score**: 1.0 on match, 0.0 otherwise.

---

## semantic_similarity

Cosine similarity between embeddings of `actual_output` and `expected_output`.

```json
{"type": "semantic_similarity", "params": {"model": "text-embedding-3-small", "threshold": 0.85}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `threshold` | number | yes | Pass threshold (0.0-1.0) |
| `model` | string | no | Embedding model identifier |
| `provider` | string | no | Embedding API provider |

**Score**: The actual similarity value. **Passed**: score >= threshold.

---

## llm_judge

An LLM evaluates the output against a rubric. The most flexible grader type.

```json
{"type": "llm_judge", "params": {
  "model": "gpt-4o",
  "prompt": "Evaluate if {output} correctly answers {input}. Expected: {expected}. Return JSON.",
  "temperature": 0.0,
  "schema": {
    "type": "object",
    "properties": {
      "score": {"type": "number"},
      "reason": {"type": "string"}
    },
    "required": ["score", "reason"]
  }
}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | yes | Judge LLM model |
| `prompt` | string | yes | Prompt template (must contain `{output}`, `{input}`, or `{expected}`) |
| `provider` | string | no | LLM API provider |
| `temperature` | number | no | Sampling temperature (default: 0.0) |
| `schema` | object | no | JSON Schema for structured judge output |

**Prompt substitutions**: `{input}`, `{output}`, `{expected}`, `{context}`

**Score**: Parsed from judge response `score` field. **Passed**: score >= threshold (default 1.0, set via params).

---

## json_schema

Validates `actual_output` (parsed as JSON) against a JSON Schema.

```json
{"type": "json_schema", "params": {"schema": {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}, "strict": true}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `schema` | object | yes | JSON Schema to validate against |
| `strict` | boolean | no | Strict mode (default: false) |

**Score**: 1.0 if valid, 0.0 if invalid.

---

## json_path

Extracts a value via JSONPath and compares it.

```json
{"type": "json_path", "params": {"path": "$.status", "expected": "success", "operator": "eq"}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | yes | JSONPath expression |
| `expected` | string | yes | Expected value |
| `operator` | string | no | Comparison: eq, ne, gt, lt, gte, lte, contains (default: eq) |

**Score**: 1.0 if comparison passes, 0.0 otherwise.

---

## code

Executes a custom grading function in a sandbox.

```json
{"type": "code", "params": {
  "language": "python",
  "source": "def grade(input, output, expected, context):\n    return 1.0 if output.strip() == expected.strip() else 0.0",
  "timeout_ms": 5000
}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `language` | string | yes | "python" or "javascript" |
| `source` | string | yes | Grading function source code |
| `timeout_ms` | integer | no | Timeout (default: 5000) |

**Security**: MUST be sandboxed. Disabled by default in CI unless `--allow-code-graders` is passed.

**Score**: Return value of the grading function.

---

## human

Defers scoring to a human reviewer.

```json
{"type": "human", "params": {"instructions": "Rate helpfulness 1-5."}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `instructions` | string | no | Instructions for the human reviewer |

**Score**: Filled in by the human reviewer. Placeholder until reviewed.

---

## model graded

Alias for `llm_judge` (OpenAI Evals compatibility). Same params and behavior.

---

## custom

Framework-specific grader not in the standard set.

```json
{"type": "custom", "params": {"handler": "com.example.my_grader", "my_param": "value"}}
```

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `handler` | string | yes | Handler identifier for the grader implementation |

**Behavior**: Runners that don't recognize the handler MUST skip the grader (score: null, passed: false, skip_reason: unsupported_grader_type).

---

## Grader Score Ranges

All grader scores MUST be in [0.0, 1.0] unless a `score_range` extension is specified. Pass/fail is determined by comparing the score to the grader's threshold.

| Grader Type | Default Threshold |
|-------------|------------------|
| exact_match | 1.0 |
| contains | 1.0 |
| regex | 1.0 |
| semantic_similarity | specified via `threshold` param |
| llm_judge | 1.0 (override via `threshold` param) |
| json_schema | 1.0 |
| json_path | 1.0 |
| code | 1.0 |
| human | N/A (reviewer sets) |
| custom | 1.0 |
