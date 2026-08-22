# parea-openeval-adapter

Convert Parea AI (https://www.parea.ai/) agent evaluation datasets, test cases, and experiment results to and from EvalPort (https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "parea-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/parea-openeval-adapter"
```

## Usage

```python
from parea_openeval_adapter import to_openeval, from_openeval, experiment_to_openeval

# 1. Convert Parea TestCaseCollection to EvalPort suite
suite = to_openeval(my_test_case_collection)

# Validate suite
from openeval.validate import validate_suite
assert validate_suite(suite).valid

# 2. Convert EvalPort suite back to Parea TestCase dicts
test_cases = from_openeval(suite)

# 3. Convert Parea Experiment or ExperimentStatsSchema to EvalPort ResultSet
result_set = experiment_to_openeval(my_experiment)

# Validate results
from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

## How It Maps

### Test Cases
- `TestCase.inputs` (dict) is mapped to EvalPort `input` (string). Standard keys like `"input"`, `"question"`, `"query"`, and `"prompt"` are checked first. If none match and only one key exists, its value is used. Otherwise, the dict is JSON-serialized.
- Original inputs are preserved under `metadata.parea.inputs` so `from_openeval()` can reconstruct the exact Parea-compatible dictionary.
- `TestCase.target` maps to `expected_output`.
- `TestCase.tags` maps to `tags`.

### Experiments & Traces
- Individual metric scores (Parea `EvaluationResult` objects) are converted to EvalPort custom grader results.
- Latency, input tokens, output tokens, total tokens, and cost are captured under each result's `metadata`.
- Pass rates and grader pass statuses are inferred using `score >= 0.5`.

## Spec

See the full EvalPort specification at https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 - see LICENSE.
