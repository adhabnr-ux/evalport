# phoenix-openeval-adapter

Convert [Arize Phoenix](https://github.com/Arize-ai/phoenix) datasets and experiment results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "phoenix-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/phoenix-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Datasets → EvalPort suite, and back

```python
from phoenix.client import Client
from phoenix_openeval_adapter import to_openeval, from_openeval

client = Client()
dataset = client.datasets.get_dataset(dataset="my-dataset")

suite = to_openeval(dataset.examples, suite_id="my_dataset_eval")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and back: load an EvalPort suite as Phoenix-uploadable dataset examples
examples = from_openeval(suite)
client.datasets.create_dataset(name="from-evalport", examples=examples)
```

Phoenix dataset examples carry `input` and `output` as arbitrary JSON mappings (e.g. `{"question": "..."}` / `{"answer": "..."}`), not flat strings, since Phoenix examples are schema-less by design. `to_openeval()` auto-detects which key inside each mapping is the model input and which is the expected output by checking common naming conventions (`input`/`question`/`query`/`prompt`/`user_input`, and `expected_output`/`output`/`answer`/`reference`/`ground_truth`), and you can always override the guess:

```python
suite = to_openeval(dataset.examples, input_key="my_custom_input_field", expected_output_key="my_custom_target_field")
```

If a mapping has exactly one key and none of the known names match, that key's value is used directly. If it has more than one unrecognized key, the whole mapping is preserved as a JSON string rather than guessing wrong — either way, the full raw `input`/`output` mappings are always kept under the test case's `metadata["phoenix"]`, so nothing from the original example is ever silently dropped even when the heuristic doesn't pick the field you'd expect.

### Experiment results → EvalPort ResultSet

```python
from phoenix.client.experiments import run_experiment
from phoenix_openeval_adapter import experiment_to_openeval

ran_experiment = run_experiment(dataset, task=my_task, evaluators=[my_evaluator])

result_set = experiment_to_openeval(ran_experiment, suite_id="my_dataset_eval")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Each Phoenix evaluator result (`score`, `label`, `explanation`) becomes one EvalPort `GraderResult`. Phoenix evaluators can return a numeric score, a label, or both — a grader result with a score passes when `score >= pass_threshold` (default `0.5`); a label-only result (common for pass/fail-style code evaluators) passes when the label reads as an affirmative value (`"pass"`, `"correct"`, `"yes"`, etc.). A result's overall `passed` follows the same convention every other EvalPort adapter uses: every one of its grader results must individually pass.

## What round-trips losslessly, and what doesn't

Exporting a dataset to EvalPort and back into Phoenix (`to_openeval` → `from_openeval` → `create_dataset()`) is lossless for the `input`/`expected_output` text and any explicit `metadata` on the test case. What doesn't survive a round trip through a *different* tool is Phoenix-specific bookkeeping — the original example id, the full raw `input`/`output` mappings if they had more fields than just the detected one — which live under `metadata["phoenix"]` rather than being interpretable by a receiving tool that isn't Phoenix, the same tradeoff every adapter in this repository makes (see the [Opik adapter](../opik-openeval-adapter) for the same pattern with Opik-specific fields).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
