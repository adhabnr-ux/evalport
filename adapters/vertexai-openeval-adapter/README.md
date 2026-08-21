# vertexai-openeval-adapter

Convert [Vertex AI's Gen AI Evaluation Service](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview) (`vertexai.evaluation`) metrics, evaluation instances, and results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "vertexai-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/vertexai-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

This installs `evalport-sdk` as its only hard dependency. `google-cloud-aiplatform[evaluation]` itself is **not** installed automatically -- add it separately.

## Usage

### Evaluation instances + metrics → EvalPort suite, and back

```python
from vertexai.evaluation import PointwiseMetric, PointwiseMetricPromptTemplate, CustomMetric
from vertexai_openeval_adapter import to_openeval, from_openeval

fluency = PointwiseMetric(
    metric="fluency",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria={"fluency": "The response is grammatically correct and clear."},
        rating_rubric={"1": "fluent", "0": "not fluent"},
    ),
)
exact_match = CustomMetric(
    name="exact_match_custom",
    metric_function=lambda instance: {
        "exact_match_custom": 1.0 if instance["response"].strip() == instance["reference"].strip() else 0.0
    },
)

suite = to_openeval(
    instances=[{"prompt": "What is the capital of France?", "reference": "Paris"}],
    metrics=[fluency, exact_match],
    suite_id="geo_facts",
)

from openeval.validate import validate_suite
assert validate_suite(suite).valid

# ...and back: rebuild everything needed to run an EvalTask for real.
rebuilt = from_openeval(suite)  # rebuilt["metrics"] only contains fluency -- see Grader mapping
import pandas as pd
from vertexai.evaluation import EvalTask

dataset = pd.DataFrame(rebuilt["instances"])
dataset["response"] = [my_app(p) for p in dataset["prompt"]]  # run your own system under test

eval_result = EvalTask(dataset=dataset, metrics=[fluency, exact_match]).evaluate()
```

`to_openeval()` takes exactly the shape needed to *define* an evaluation run -- Vertex-style instance dicts (`{"prompt": ..., "reference": ...}`) plus a list of metric objects -- deliberately excluding the `"response"` field even if present, since that's the output being graded and doesn't exist yet at suite-definition time. Every metric becomes one EvalPort grader, applied to every test case, matching how `EvalTask(metrics=[...])` runs every metric against every instance uniformly.

### EvalResult → EvalPort ResultSet

```python
from vertexai_openeval_adapter import batch_eval_result_to_openeval

result_set = batch_eval_result_to_openeval(
    eval_result.metrics_table,       # the real pandas.DataFrame EvalTask.evaluate() returns
    test_case_ids=rebuilt["ids"],
    metrics=[fluency, exact_match],
    suite_id="geo_facts",
    run_id="run-2026-08-15",
    started_at="2026-08-15T00:00:00Z",
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

`metrics_table` is read using Vertex's own real column-naming convention (`f"{metric_name}/score"`, `f"{metric_name}/explanation"`), verified directly from `vertexai/evaluation/_evaluation.py`, not guessed.

## Grader mapping

| Vertex metric | EvalPort grader `type` | Direction |
|---|---|---|
| `PointwiseMetric` | `llm_judge` (real, literal prompt template preserved in `params.prompt`) | both |
| `CustomMetric` | `custom` (`params.handler` names the metric) | export only |
| `PairwiseMetric` | `custom` (`params.handler` names the metric) | export only |
| raw string metric names (`"rouge_1"`, `"bleu"`, `"exact_match"`, `"tool_call_valid"`, etc.) | -- | **not supported**, raises `TypeError` |

`PointwiseMetric` is genuinely different from most other adapters' `llm_judge` mapping in this ecosystem: `metric.metric_prompt_template` renders to the actual, literal instruction/criteria/rubric text Vertex sends its judge model -- not a synthesized description. `CustomMetric` is, per Vertex's own docstring, *"computed on the client-side using the user-defined metric function in SDK only, not by the Vertex Gen AI Evaluation Service"* -- genuinely local code with no fixed shape EvalPort can generically interpret, so it maps onto `custom` and is **export only**. `PairwiseMetric` judges a candidate response against a *baseline* response rather than one response against a query/context/reference -- there's no EvalPort grader shape for a two-response comparison, so (matching the [LlamaIndex adapter](../llamaindex-openeval-adapter#grader-mapping)'s handling of its own `PairwiseComparisonEvaluator`) it's exported as `custom` and never reconstructed on import.

Raw string metric names (Vertex also accepts `metrics=["rouge_1", "bleu", "exact_match", ...]` directly, without wrapping them in a `Metric` object) are deliberately **not accepted** by this adapter -- whether each one computes client-side or via a live Vertex API call could not be verified offline within this adapter's own test suite (no GCP credentials available), so rather than guess and risk misrepresenting a grader's execution model, `to_openeval()` raises `TypeError` naming the unsupported metric. Wrap the underlying computation in a `CustomMetric` if you need one of these today.

## What round-trips losslessly, and what doesn't

A `PointwiseMetric`'s identity (`metric_name`, the full rendered prompt template text) round-trips exactly -- `from_openeval()` reconstructs the exact same `PointwiseMetric` whenever the grader carries this adapter's own `metadata.vertexai` (`openeval.*` is the only reserved metadata prefix; everything else preserved this way follows the same convention every adapter in this repo uses). The original *structured* `PointwiseMetricPromptTemplate` fields (separate `criteria`/`rating_rubric`/`instruction` dicts) are not reconstructed as structured data -- only the already-rendered string, since Vertex's own `PointwiseMetric(metric_prompt_template=...)` accepts a plain string just as validly and the rendered text is everything the metric actually needs to run.

`CustomMetric`/`PairwiseMetric` cannot be reconstructed on import (see above) -- their graders are exported so their presence and config are never silently dropped, but running them again requires the caller to keep their own reference to the original metric object and supply it themselves to `batch_eval_result_to_openeval()`'s `metrics` argument, exactly the same "you own execution, we own the data shape" boundary every other EvalPort adapter draws.

Instance fields beyond `prompt`/`reference` are honestly **not** force-mapped onto EvalPort's `context` field -- Vertex's per-metric instance schema varies (some need `context`, others `instruction`, `baseline_response`, tool-call fields, and so on), and guessing which extra key means what would misrepresent the data. Every extra instance field is instead preserved verbatim under `test_case.metadata.vertexai.extra_instance_fields`, so nothing is lost, just not force-mapped onto a field it may not semantically match.

## Running the tests

```bash
cd adapters/vertexai-openeval-adapter
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest tests/ -v
```

28 tests, all running against the real `vertexai.evaluation` classes (`PointwiseMetric`, `PairwiseMetric`, `CustomMetric`, `PointwiseMetricPromptTemplate`) and the real `openeval.validate.validate_suite()`/`validate_result_set()` -- not mocks. `PointwiseMetric`/`PairwiseMetric` are constructed (real objects, real rendered prompt text) but never run through `EvalTask.evaluate()`, since that calls the live Vertex AI Evaluation Service and needs GCP credentials. `CustomMetric` is different -- per Vertex's own docstring it computes client-side, so `TestEndToEndWithRealCustomMetric` actually calls a real `CustomMetric.metric_function()` directly, the exact same way `vertexai/evaluation/_evaluation.py` itself invokes it internally (confirmed by reading that source file): `to_openeval()` → `from_openeval()` → real `metric.metric_function(row)` calls → `batch_eval_result_to_openeval()` → `validate_result_set()`.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
