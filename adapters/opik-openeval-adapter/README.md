# opik-openeval-adapter

Convert [Comet Opik](https://github.com/comet-ml/opik) datasets and experiment results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

When [opik#7798](https://github.com/comet-ml/opik/issues/7798) was filed proposing EvalPort import/export, the repository's own triage agent ("Scout") did a full investigation of the SDK and recommended exactly this: a standalone `opik-openeval-adapter` package built against Opik's existing public surface (`Dataset.get_items()`/`insert()`, `ExperimentItemContent`, `FeedbackScoreDict`) with zero changes to Opik core — the same playbook that already shipped for [AutoGen](../autogen-openeval-adapter) and [CrewAI](../crewai-openeval-adapter). If Opik decides later to bundle this as a first-party `integrations/evalport/` module (the "Option B" Scout also outlined), this package's mapping is the reference implementation to start from.

## Install

```bash
pip install opik-openeval-adapter
```

## Usage

### Datasets → EvalPort suite, and back

```python
from opik_openeval_adapter import to_openeval, from_openeval
import opik

client = opik.Opik()
dataset = client.get_dataset("my-dataset")
items = dataset.get_items()  # List[Dict[str, Any]]

suite = to_openeval(items, suite_id="my_dataset_eval")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and back: load an EvalPort suite as Opik-insertable dataset items
restored_items = from_openeval(suite)
dataset.insert(restored_items)
```

Opik dataset items are schema-less — any JSON-serializable fields per item. `to_openeval()` auto-detects which field is the model input and which is the expected output by checking common naming conventions (`input`/`question`/`user_input`/`prompt`/`query`, and `expected_output`/`expected_answer`/`answer`/`reference`/`ground_truth`), and you can always override the guess:

```python
suite = to_openeval(items, input_key="my_custom_input_field", expected_output_key="my_custom_target_field")
```

Every other field on the item — whatever the guess picks or misses — is preserved under the test case's `metadata`, so nothing from the original dataset item is ever silently dropped, even when the auto-detected input/expected-output guess is wrong for an unusual dataset shape.

### Experiment results → EvalPort ResultSet

```python
from opik_openeval_adapter import experiment_to_openeval

experiment = client.get_experiment_by_name("my-experiment")
items = experiment.get_items()  # ExperimentItemContent objects

result_set = experiment_to_openeval(items, suite_id="my_dataset_eval", run_id="run_001")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Each Opik `FeedbackScoreDict` (`name`, `value`, `category_name`, `reason`) becomes one EvalPort `GraderResult`. Opik feedback scores are a bare numeric value with no built-in pass/fail — `pass_threshold` (default `0.5`) is where that boolean gets decided; pass a different threshold if your scoring scale doesn't center on 0.5. A result's overall `passed` follows the same convention `evalport run`'s own runner uses: every one of its grader results must individually pass.

## What round-trips losslessly, and what doesn't

Exporting a dataset to EvalPort and back into Opik (`to_openeval` → `from_openeval` → `dataset.insert()`) is lossless for the data itself — every field survives via `metadata`. What doesn't survive a round-trip through a *different* tool is Opik-specific semantics: a receiving tool that isn't Opik has no way to interpret the `opik.dataset_item_id`/`opik.evaluators` bookkeeping this adapter writes into `metadata`, the same tradeoff every adapter in this repository makes (see the [AutoGen adapter](../autogen-openeval-adapter) for the same pattern with AutoGen-specific fields).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
