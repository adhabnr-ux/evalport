# openeval

Python SDK for EvalPort — The Open Evaluation Standard.

## Install

```bash
pip install openeval
```

## Usage

### Validate a suite

```python
from openeval.validate import validate_suite

result = validate_suite({
    "version": "1.0.0",
    "id": "my_suite",
    "graders": [{"id": "gr1", "type": "exact_match"}],
    "test_cases": [{"id": "tc1", "input": "Hello", "expected_output": "Hi", "graders": ["gr1"]}]
})
print(result.valid)  # True
```

### Convert from Promptfoo

```python
from openeval.convert import from_promptfoo

suite = from_promptfoo(promptfoo_config)
```

### Convert from DeepEval

```python
from openeval.converters_deepeval import from_deepeval

suite = from_deepeval(deepeval_export)
```

### Convert from Inspect AI

```python
from openeval.converters_inspect import from_inspect

suite = from_inspect(inspect_data)
```

### Convert from OpenAI Evals

```python
from openeval.converters_openai import from_openai_evals

suite = from_openai_evals(evals_data)
```

### Convert from / to CrewAI

```python
from openeval.converters_crewai import from_crewai, crewai_result_to_result_set

suite = from_crewai({"tasks": crewai_task_defs})
result_set = crewai_result_to_result_set(crewai_run_result, suite, run_id="run_001")
```

### Compute summary

```python
from openeval.convert import compute_summary, create_result_set

summary = compute_summary(results)
result_set = create_result_set(suite, results, "run_001")
```

## API

### Validation
- `validate_suite(doc)` → `ValidationResult`
- `validate_test_case(doc)` → `ValidationResult`
- `validate_grader(doc)` → `ValidationResult`
- `validate_result_set(doc)` → `ValidationResult`
- `validate_document(doc, type)` → `ValidationResult`

### Conversion
- `from_promptfoo(config)` → `dict`
- `from_deepeval(data)` → `dict` (from `converters_deepeval`)
- `from_inspect(data)` → `dict` (from `converters_inspect`)
- `from_openai_evals(data)` → `dict` (from `converters_openai`)
- `from_crewai(data)` → `dict` (from `converters_crewai`)
- `crewai_result_to_result_set(crew_result, suite, run_id)` → `dict` (from `converters_crewai`)
- `compute_summary(results)` → `dict`
- `create_result_set(suite, results, run_id)` → `dict`

## License

Apache 2.0
