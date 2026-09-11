# unitxt-openeval-adapter

Convert [IBM unitxt](https://github.com/IBM/unitxt) datasets and evaluation results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "unitxt-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/unitxt-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Task rows → EvalPort suite, and back

```python
from unitxt_openeval_adapter import to_openeval, from_openeval

rows = [
    {"question": "What is the capital of France?", "answers": ["Paris"]},
    {"question": "Who wrote Hamlet?", "answers": ["Shakespeare"]},
]

suite = to_openeval(rows, suite_id="my_eval_suite")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and back: reconstruct unitxt-loadable test_set rows
restored = from_openeval(suite)
```

`rows` is any iterable of plain dicts in the same shape you'd pass as `test_set=` to `unitxt.api.create_dataset(task=..., test_set=rows)` — a unitxt task's own raw input schema. The input/expected-output field is auto-detected from several real built-in tasks' conventions (`question`/`answers` for `tasks.qa.open`, `text`/`class` for classification, `document`/`summary` for summarization, ...), or named explicitly:

```python
suite = to_openeval(rows, input_key="document", expected_output_key="summary")
```

unitxt commonly represents the expected output as a list of acceptable answers (`answers: ["Paris"]`) rather than a single string. The first element is used for EvalPort's single-string `TestCase.expected_output`, and the full original list is always additionally preserved under `metadata["unitxt"]["expected_output_full"]` — a multi-reference task never silently loses its other valid answers. Every other field on a row is preserved under `metadata["unitxt"]`, so nothing from the original row is ever dropped.

### Evaluation results → EvalPort ResultSet

```python
from unitxt.api import create_dataset, evaluate
from unitxt_openeval_adapter import evaluation_to_openeval

dataset = create_dataset(task="tasks.qa.open", test_set=rows, split="test", format="formats.chat_api")
predictions = ["Paris", "William Shakespeare"]
results = evaluate(predictions=predictions, data=dataset)

result_set = evaluation_to_openeval(results, suite_id="my_eval_suite", run_id="run-1")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every key in an instance's `score["instance"]` dict — except the `score_name` string identifier and the `score` field, which is always a numeric duplicate of whichever metric `score_name` names, verified against a real unitxt run — becomes its own EvalPort `GraderResult`. A task with multiple attached metrics (e.g. `metrics.rouge`, which computes `rouge1`/`rouge2`/`rougeL`/`rougeLsum` together) gets one `GraderResult` per metric, not just the task's single primary score.

**Matching test case ids.** Unlike a distributed evaluation framework, `unitxt.evaluate()` is a synchronous, in-process, order-preserving operation over the exact `predictions` list you pass it — verified in this adapter's own test suite against a real run. `test_case_id`s are assigned by position (`"tc_0"`, `"tc_1"`, ...), which always matches the order `to_openeval()` assigns for the same rows, so no separate id-matching step is needed (contrast with the [fmeval adapter](../fmeval-openeval-adapter), whose distributed Ray Dataset execution does need one).

**Global scores.** The run-level `score["global"]` dict (identical across every instance in a single `evaluate()` call, and including confidence-interval fields not meaningful per-instance) is preserved once under `result_set["metadata"]["unitxt"]["global_scores"]` rather than repeated on every result.

**Score range.** EvalPort requires every `GraderResult.score` to be `null` or within `[0, 1]`. Most unitxt metrics (accuracy, F1, rouge, BLEU-derived scores) already are, but this adapter clamps defensively for any metric that isn't, always preserving the true value under `grader_results[].metadata.raw_value` — the same convention used by every numeric-score adapter in this repository (see the [fmeval adapter](../fmeval-openeval-adapter)).

## What round-trips losslessly, and what doesn't

Exporting a dataset to EvalPort and back (`to_openeval` → `from_openeval`) is lossless for `input`/`expected_output` (including the full multi-reference `answers` list) and every other field carried in `metadata["unitxt"]`. What doesn't survive a round trip through a *different* tool is that unitxt task-specific field shape (e.g. the original `question`/`answers` key names for `tasks.qa.open`) — a receiving tool that isn't unitxt only sees the generic `input`/`expected_output`/`metadata` fields, the same tradeoff every adapter in this repository makes (see the [Opik adapter](../opik-openeval-adapter) for the same pattern).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
