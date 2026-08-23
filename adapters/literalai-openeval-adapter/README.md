# literalai-openeval-adapter

Convert [Literal AI](https://docs.literalai.com/) `Dataset` / `DatasetItem` /
`DatasetExperiment` data to and from [EvalPort](https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM evaluation datasets.

## Install

```
pip install "literalai-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/literalai-openeval-adapter"
```

Not yet published to PyPI — installs directly from source via pip's
`git+` / `#subdirectory=` support (verified working).

## Usage

### Datasets → EvalPort suite

Pass a Literal AI `Dataset` object, or a dict with `"name"`, optional `"id"`, and `"items"` (a list of
DatasetItem-shaped dicts):

```python
import literalai
from literalai_openeval_adapter import to_openeval

client = literalai.LiteralClient(api_key="...")
dataset = client.api.get_dataset(name="my-dataset")

suite = to_openeval(dataset)           # llm_judge grader (default)
# suite = to_openeval(dataset, grader_type="exact_match")  # for clean-string expected answers

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
# items is a list of dicts ready for Dataset.create_item()
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
    experiment,
    suite_id=dataset.id,
    run_id=experiment.id,
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

## Design notes

Three translation challenges are handled explicitly:

### 1. `input` / `expected_output` flattening

Literal AI datasets are schema-free key-value rows
(e.g. `{"question": "What is 2+2?", "context": "..."}`).
EvalPort requires a plain string. `flatten_dict_field()` picks the value of the
first key matching a preferred name (`question`, `query`, `input`, `prompt`,
`text`), then falls back to the first string-valued field in the dict.
If the dict contains **no string-valued fields at all**, it falls back to JSON-serializing the whole dict (instead of raising) — an empty dict still raises `ValueError`, since there's nothing to represent.

The original dict is always preserved under `metadata.literalai.original_input` so
`from_openeval()` can restore it losslessly.

### 2. Score clamping

Literal AI's `Score.value` is an unbounded `float` (e.g. `8.5` or `100`).
EvalPort's `GraderResult.score` must be in `[0, 1]` or `null`. This adapter
**clamps** the value and preserves the original under the spec's own reserved
metadata key, `metadata.openeval.raw_score` so no information is lost.

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
