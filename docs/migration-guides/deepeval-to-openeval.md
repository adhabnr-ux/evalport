# Migrating from DeepEval to EvalPort

## Overview

DeepEval uses Python classes (`LLMTestCase`, metric objects) for evaluation. EvalPort is a JSON-based portable format. This guide helps you convert DeepEval test cases to EvalPort suites.

## Field Mapping

| DeepEval | EvalPort |
|----------|----------|
| `LLMTestCase.input` | `TestCase.input` |
| `LLMTestCase.actual_output` | (not in suite; appears in ResultSet) |
| `LLMTestCase.expected_output` | `TestCase.expected_output` |
| `LLMTestCase.context` | `TestCase.context` |
| `LLMTestCase.retrieval_context` | `TestCase.retrieval_context` |
| `ConversationalTestCase` | `TestCase.input` as array of turns |
| `FaithfulnessMetric` | `Grader.type: "llm_judge"` |
| `AnswerRelevancyMetric` | `Grader.type: "semantic_similarity"` |
| `ExactMatchMetric` | `Grader.type: "exact_match"` |
| `GEval` | `Grader.type: "llm_judge"` |
| Custom metrics | `Grader.type: "custom"` |

## SDK Conversion

### Python

```python
from openeval.convert import from_deepeval
import json

de = json.load(open("deepeval_export.json"))
suite = from_deepeval(de)
json.dump(suite, open("output.json", "w"), indent=2)
```

## Example

### Before (DeepEval export)

```json
{
  "test_cases": [
    {
      "input": "What is Kubernetes?",
      "expected_output": "Container orchestration platform",
      "context": ["K8s orchestrates containers"],
      "metrics": ["faithfulness", "answer_relevancy"]
    }
  ]
}
```

### After (EvalPort)

```json
{
  "version": "1.0.0",
  "id": "suite_deepeval_import",
  "name": "Imported from DeepEval",
  "graders": [
    {"id": "gr_0_0", "type": "llm_judge", "description": "Faithfulness (DeepEval)", "params": {"model": "gpt-4o", "prompt": "..."}},
    {"id": "gr_0_1", "type": "semantic_similarity", "params": {"threshold": 0.5}}
  ],
  "test_cases": [
    {"id": "tc_0", "input": "What is Kubernetes?", "expected_output": "Container orchestration platform", "context": ["K8s orchestrates containers"], "graders": ["gr_0_0", "gr_0_1"]}
  ]
}
```

## DeepEval Metric → EvalPort Grader Mapping

| DeepEval Metric | EvalPort Grader Type | Notes |
|----------------|---------------------|-------|
| `FaithfulnessMetric` | `llm_judge` | Prompt checks if output is faithful to context |
| `AnswerRelevancyMetric` | `semantic_similarity` | Threshold from metric params |
| `ContextualPrecisionMetric` | `llm_judge` | Checks if retrieval context is relevant |
| `ContextualRecallMetric` | `llm_judge` | Checks if expected info is in context |
| `ExactMatchMetric` | `exact_match` | Direct mapping |
| `GEval` | `llm_judge` | Custom criteria via prompt |
| `SummarizationMetric` | `llm_judge` | Checks summary quality |
| `HallucinationMetric` | `llm_judge` | Checks for hallucinated content |
| `ToxicityMetric` | `llm_judge` | Checks for toxic output |

## Limitations

- DeepEval metric thresholds and custom criteria are approximated; exact behavior may differ
- DeepEval's `generate_synthetic_data` is out of scope (data generation, not eval format)
- Custom metric classes map to `custom` graders with `handler: "deepeval:METRIC_NAME"`
