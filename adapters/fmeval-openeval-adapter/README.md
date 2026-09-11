# fmeval-openeval-adapter

Convert [AWS fmeval](https://github.com/aws/fmeval) (Foundation Model Evaluations Library) datasets and evaluation output to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "fmeval-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/fmeval-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Datasets → EvalPort suite, and back

```python
from fmeval_openeval_adapter import to_openeval, from_openeval

rows = [
    {"model_input": "What is the capital of France?", "target_output": "Paris", "category": "geography"},
    {"model_input": "Who wrote Hamlet?", "target_output": "Shakespeare", "category": "literature"},
]

suite = to_openeval(rows, suite_id="my_eval_suite")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and back: load an EvalPort suite as fmeval-loadable dataset rows
fmeval_rows = from_openeval(suite)
with open("fmeval_dataset.jsonl", "w") as f:
    for row in fmeval_rows:
        f.write(json.dumps(row) + "\n")

# Then point a DataConfig at it:
# from fmeval.data_loaders.data_config import DataConfig
# from fmeval.constants import MIME_TYPE_JSONLINES
# config = DataConfig(
#     dataset_name="from_evalport", dataset_uri="fmeval_dataset.jsonl",
#     dataset_mime_type=MIME_TYPE_JSONLINES,
#     model_input_location="model_input", target_output_location="target_output",
# )
```

`rows` is any iterable of plain dicts — the same rows you'd write to a JSON Lines file and point an fmeval `DataConfig` at. They can already use fmeval's own canonical column names (`model_input`, `target_output`, `category`, `context`) or your own raw field names (`question`/`answer`, etc.), which are auto-detected the same way every schema-less adapter in this repository handles its source format — you can always override the guess:

```python
suite = to_openeval(rows, input_key="question", expected_output_key="answer")
```

fmeval's `context` column (used by RAG-oriented algorithms) maps directly onto EvalPort's own `TestCase.context` field. Every other field on a row is preserved under the test case's `metadata["fmeval"]`, so nothing is ever silently dropped even when the input/expected-output/context/category guess doesn't pick the field you'd expect.

### Evaluation output → EvalPort ResultSet

```python
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.factual_knowledge import FactualKnowledge, FactualKnowledgeConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval_openeval_adapter import eval_output_to_openeval

config = DataConfig(
    dataset_name="capitals", dataset_uri="dataset.jsonl", dataset_mime_type=MIME_TYPE_JSONLINES,
    model_input_location="question", target_output_location="answer", model_output_location="model_output",
)
outputs = FactualKnowledge(FactualKnowledgeConfig()).evaluate(model=None, dataset_config=config, save=True)

# outputs[0].output_path is the real per-record JSON Lines file fmeval wrote.
result_set = eval_output_to_openeval(output_path=outputs[0].output_path, suite_id="my_eval_suite", run_id="run-1")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Each fmeval `EvalScore` (read from the real `EvalOutputRecord`-shaped JSON Lines fmeval writes to `output_path`) becomes one EvalPort `GraderResult`, keyed by the score's own name (`"factual_knowledge"`, `"rouge"`, `"bertscore"`, `"toxicity"`, ...). A score with an `error` instead of a `value` — fmeval's own convention for a per-record computation failure — becomes a `GraderResult` with `score=None`, `passed=False`, and the error text as `reason`.

**Score direction.** Every fmeval score is "higher is better" *except* two verified exceptions: `toxicity` (0..1, where 1.0 means fully toxic) and `word_error_rate` (0..1+, where 0 means a perfect match). `eval_output_to_openeval()` accounts for this automatically — pass `lower_is_better={...}` to extend the set for any other fmeval score with this convention.

**Score range.** EvalPort requires every `GraderResult.score` to be `null` or within `[0, 1]`. Most fmeval scores already are, but `log_probability_difference` (prompt-stereotyping bias) is a signed, unbounded log-probability delta and `word_error_rate` can exceed `1.0`. Rather than reject those, this adapter clamps the EvalPort `score` field into range while always preserving the true value under `grader_results[].metadata.raw_value` — nothing from the real fmeval output is lost.

**Matching test case ids.** fmeval evaluates via a distributed Ray Dataset internally, which does not guarantee output row order matches input row order under parallel execution. Pass `test_cases=suite["test_cases"]` (the same suite you built with `to_openeval()`) to have `eval_output_to_openeval()` recover the correct `test_case_id` for each record by matching on `(input, expected_output)`, rather than relying on emission order.

## What round-trips losslessly, and what doesn't

Exporting a dataset to EvalPort and back into an fmeval-loadable JSON Lines file (`to_openeval` → `from_openeval`) is lossless for `model_input`/`target_output`/`context` and any other field carried in `metadata["fmeval"]`. What doesn't survive a round trip through a *different* tool is that fmeval-specific field shape — a receiving tool that isn't fmeval only sees the generic `input`/`expected_output`/`context`/`metadata` fields, the same tradeoff every adapter in this repository makes (see the [Opik adapter](../opik-openeval-adapter) for the same pattern).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
