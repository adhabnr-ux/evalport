# weave-openeval-adapter

Convert [Weights & Biases Weave](https://github.com/wandb/weave) datasets and evaluation results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "weave-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/weave-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Datasets → EvalPort suite, and back

```python
import weave
from weave_openeval_adapter import to_openeval, from_openeval

dataset = weave.Dataset(name="grammar", rows=[
    {"id": "0", "question": "What is the capital of France?", "expected": "Paris"},
    {"id": "1", "question": "Who wrote 'To Kill a Mockingbird'?", "expected": "Harper Lee"},
])

suite = to_openeval(dataset, suite_id="grammar_suite")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("grammar_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and back: load an EvalPort suite as Weave dataset rows
rows = from_openeval(suite)
ds = weave.Dataset(name="from-evalport", rows=rows)
weave.publish(ds)
```

Weave dataset rows are flat, schema-less dicts — there's no fixed `input`/`output` column, just whatever keys you gave `weave.Dataset(rows=...)`. `to_openeval()` auto-detects which column is the model input and which is the expected output by checking common naming conventions (`input`/`question`/`query`/`prompt`/`user_input`/`text`, and `expected_output`/`expected`/`output`/`answer`/`reference`/`ground_truth`/`label`/`correction`), the same heuristic the [Phoenix](../phoenix-openeval-adapter) and [Opik](../opik-openeval-adapter) adapters use for their own schema-less formats. If a row has exactly one remaining candidate key after the known names are checked, that key's value is used; if more than one unrecognized key remains, the whole row is preserved as a JSON string rather than guessing wrong. You can always override the guess:

```python
suite = to_openeval(dataset, input_key="my_custom_prompt_field", expected_output_key="my_custom_target_field")
```

Either way, the full original row is always kept under the test case's `metadata["weave"]["row"]`, so nothing from the source dataset is ever silently dropped — and `from_openeval()` reconstructs that exact original row when it's present, so a Weave → EvalPort → Weave round trip through this adapter is lossless.

### Evaluation results → EvalPort ResultSet

```python
from weave_openeval_adapter import evaluation_to_openeval

# `rows` is the same list you built the dataset from; `eval_results` is one
# dict per row in the same order, shaped like Weave's own
# `Evaluation.predict_and_score()` output: {"output": ..., "scores": {...},
# "model_latency": ...}. You get this shape back from `EvaluationLogger`,
# or by iterating `EvaluationResults.rows` after
# `evaluation.get_eval_results(model)`.
result_set = evaluation_to_openeval(rows, eval_results, suite_id="grammar_suite", run_id="run-1")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Each Weave scorer result becomes one EvalPort `GraderResult`. Weave scorers can return a bare bool, a bare number, or a dict of sub-fields (the same three shapes Weave's own `auto_summarize()` special-cases when it summarizes a completed evaluation) — a bool result maps directly to `passed`, a numeric result is clamped into EvalPort's required `[0, 1]` range and passes at `>= 0.5`, and a dict result is searched for a conventional `passed`/`correct`/`score`/`value` field before falling back to the first bool or number found inside it. The full raw scorer result is always preserved under `metadata.raw` so nothing is lost even when the heuristic doesn't pick the field you'd expect. A result's overall `passed` follows the same convention every other EvalPort adapter uses: every one of its grader results must individually pass.

## What round-trips losslessly, and what doesn't

Exporting a dataset to EvalPort and back into Weave (`to_openeval` → `from_openeval` → `weave.Dataset(rows=...)`) is lossless: the full original row is preserved verbatim under `metadata["weave"]["row"]` and reconstructed exactly on the way back. What doesn't survive a round trip through a *different* tool is that same Weave-specific row shape and any custom column names it used — a receiving tool that isn't Weave only sees the generic `input`/`expected_output`/`metadata` fields, the same tradeoff every adapter in this repository makes (see the [Phoenix adapter](../phoenix-openeval-adapter) for the same pattern with Phoenix-specific fields).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
