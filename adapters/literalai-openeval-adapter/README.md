# literalai-openeval-adapter

Convert [Literal AI](https://docs.literalai.com/) `Dataset` / `DatasetItem` /
`DatasetExperiment` data to and from [EvalPort](https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM evaluation datasets.

## Install

```
pip install "literalai-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/literalai-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+` / `#subdirectory=` support (verified working).

To also install the real Literal AI SDK (needed to work with actual Literal AI
objects rather than plain dicts):

```
pip install "literalai-openeval-adapter[literalai] @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/literalai-openeval-adapter"
```

Verified minimum: `literalai==0.1.300`.

## Usage

### Datasets → EvalPort suite

```python
import literalai
from literalai_openeval_adapter import to_openeval

client = literalai.LiteralClient(api_key="...")
dataset = client.api.get_dataset(name="my-dataset")

suite = to_openeval(
    dataset.items,
    suite_id=dataset.id,
    suite_name=dataset.name,
)

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

### EvalPort suite → Literal AI DatasetItem dicts

```python
from literalai_openeval_adapter import from_openeval

items = from_openeval(suite)
# Each item is a dict ready for Dataset.create_item()
for item in items:
    dataset.create_item(
        input=item["input"],
        expected_output=item["expected_output"],
        metadata=item["metadata"],
    )
```

### DatasetExperiment results → EvalPort ResultSet

```python
from literalai_openeval_adapter import results_to_openeval

experiment = client.api.get_dataset_experiment(id="my-exp-id")
result_set = results_to_openeval(
    experiment.items,
    result_set_id=experiment.id,
    experiment_name=experiment.name,
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

## Design notes

Three translation challenges are handled explicitly:

### 1. `input` / `expected_output` flattening
Literal AI datasets are schema-free key-value rows
(e.g. `{"question": "What is 2+2?", "context": "..."}`).
EvalPort requires a plain string. This adapter picks the value of the first
key whose name looks like an input (`question`, `query`, `input`, `prompt`,
`text`, …) and falls back to JSON-serialising the whole dict so nothing is
silently dropped. The original raw dict is always preserved in
`metadata["_literalai_raw"]` so you can reconstruct it with `from_openeval()`.

### 2. Score clamping
Literal AI's `Score.value` is an unbounded `float` (e.g. `8.5` or `100`).
EvalPort's `GraderResult.score` must be in `[0, 1]` or `null`. This adapter
**clamps** the value and stores the original in `metadata["_raw_score"]` so
no information is lost.

### 3. Score type mapping
Literal AI's `Score.type` is a 3-way tag: `"HUMAN"` / `"CODE"` / `"AI"`.
EvalPort grader types are `"human"` / `"code"` / `"llm_judge"`.

| Literal AI | EvalPort    |
|------------|-------------|
| `AI`       | `llm_judge` |
| `CODE`     | `code`      |
| `HUMAN`    | `human`     |

## Spec

See the full EvalPort specification at
<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>

## License

Apache 2.0 — see [LICENSE](LICENSE).
