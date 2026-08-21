# dspy-openeval-adapter

Convert [DSPy](https://github.com/stanfordnlp/dspy) devsets and `dspy.Evaluate` results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "dspy-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/dspy-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Export a devset to an EvalPort suite

```python
import dspy
from dspy_openeval_adapter import to_openeval

devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
]

suite = to_openeval(devset, input_keys=["question"], expected_key="answer")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

`input_keys` names which `Example` field(s) become `TestCase.input` (EvalPort has no named-field concept, so multiple keys are flattened into `["key: value", ...]` — one string per key). `expected_key` names the one field that becomes `expected_output`. On a round trip through this adapter, nothing is lost: every original field is additionally preserved under `test_case.metadata.dspy.fields`.

### Import an EvalPort suite as a devset, ready to run

```python
from dspy_openeval_adapter import from_openeval

devset = from_openeval(suite)  # -> list[dspy.Example], input keys already marked

evaluator = dspy.Evaluate(devset=devset, metric=my_metric, display_progress=True)
result = evaluator(my_program)
```

A suite built by this adapter round-trips its exact original `Example` fields. A hand-authored suite (or one from a different EvalPort-speaking tool) instead gets positionally-named fields (`input_1`, `input_2`, ...) unless you pass `input_keys=[...]` explicitly to name them yourself.

### Export evaluation results to an EvalPort ResultSet

```python
from dspy_openeval_adapter import evaluation_result_to_openeval

result = evaluator(my_program)  # a dspy.EvaluationResult
result_set = evaluation_result_to_openeval(result, suite_id=suite["id"], metric=my_metric)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

A DSPy metric is arbitrary Python — it may return `bool` (the common case), a plain number (rarely outside `[0, 1]`), or a `dspy.Prediction(score=..., feedback=...)` for GEPA-style feedback-augmented metrics. All three are handled: bools map directly, `Prediction.score`/`feedback` map onto the grader result's `score`/`reason`, and any other numeric score is clamped into EvalPort's required `[0, 1]` range with the original value preserved in `grader_result.metadata.dspy.raw_score` whenever clamping changed it.

### The full loop

```python
suite = to_openeval(devset, input_keys=["question"], expected_key="answer", ids=["fr", "jp"])
devset2 = from_openeval(suite)
result = dspy.Evaluate(devset=devset2, metric=my_metric, display_progress=False)(my_program)
result_set = evaluation_result_to_openeval(result, suite_id=suite["id"])
# result_set["results"][i]["test_case_id"] == suite["test_cases"][i]["id"], preserved end to end
```

## Why the metric itself isn't a real grader

EvalPort has no way to serialize an arbitrary Python callable, and a DSPy metric function is exactly that — there's no portable, re-executable representation to give it. `to_openeval()` therefore emits one placeholder `custom` grader per suite that documents "the caller must supply a metric at run time," rather than fabricating a fake grader implementation. This is the same honest limitation every adapter in this ecosystem takes for framework-specific logic with no native EvalPort shape (see [vertexai-openeval-adapter](../vertexai-openeval-adapter)'s `CustomMetric` handling, or [crewai-openeval-adapter](../crewai-openeval-adapter)'s tool-selection grader, for the same pattern).

## What round-trips losslessly, and what doesn't

DSPy → EvalPort → DSPy (via this adapter both ways): lossless — every `Example` field and its input/label marking survives exactly, restored from `metadata.dspy.fields`.

DSPy → EvalPort → some other tool: the flattened `"key: value"` input strings and the single `expected_output` value are readable by any EvalPort consumer, but a different tool has no way to know which field was a DSPy signature's actual input versus free-form context — the same tradeoff every adapter here takes for structure that doesn't have a native EvalPort field.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
