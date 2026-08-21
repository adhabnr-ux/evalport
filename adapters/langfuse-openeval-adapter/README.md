# langfuse-openeval-adapter

Convert [Langfuse](https://github.com/langfuse/langfuse) datasets and experiment results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "langfuse-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/langfuse-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Dataset → EvalPort suite, and back

```python
from langfuse import Langfuse
from langfuse_openeval_adapter import to_openeval, from_openeval

lf = Langfuse()
dataset = lf.get_dataset("geo_science_eval")

suite = to_openeval(dataset.items, suite_id="geo_science_eval")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

# ...and back: load an EvalPort suite as Langfuse-shaped rows, usable
# directly as run_experiment()'s `data` argument
rows = from_openeval(suite)

result = lf.run_experiment(
    name="geo_science_eval",
    data=rows,
    task=lambda item, **_: my_app(item["input"]),
)
```

You don't need a stored Langfuse dataset to use `to_openeval()` — it also accepts plain `{"input": ..., "expected_output": ..., "metadata": ...}` dicts, the same shape Langfuse's own `run_experiment(data=...)` accepts for ad-hoc data (`LocalExperimentItem`).

Langfuse's `input`/`expected_output`/`metadata` map directly onto EvalPort's `TestCase.input`/`expected_output`/`metadata`. The full original dataset item — including Langfuse-only fields with no home on an EvalPort `TestCase` (`id`, `status`, `dataset_id`, `dataset_name`, `source_trace_id`, …) — is preserved under the test case's `metadata["langfuse"]["item"]`, and `from_openeval()` reconstructs it verbatim when present, so a round trip through this adapter never drops a field Langfuse itself would need (e.g. to call `create_dataset_item(dataset_name=..., **item)` again).

### Experiment results → EvalPort ResultSet

```python
from langfuse_openeval_adapter import experiment_result_to_openeval

result = lf.run_experiment(
    name="geo_science_eval",
    data=rows,
    task=lambda item, **_: my_app(item["input"]),
    evaluators=[correctness_evaluator, conciseness_evaluator],
)

result_set = experiment_result_to_openeval(result, suite_id="geo_science_eval", run_id=result.run_name)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every `Evaluation` Langfuse scores against an item becomes one EvalPort `GraderResult`. Langfuse's `Evaluation.value` has three possible shapes and each is handled on its own terms rather than force-fit into one rule:

- **`BOOLEAN`** (or a Python `bool` value) → `score` is `1.0`/`0.0`, `passed` matches the boolean directly.
- **`NUMERIC`** (or a raw `int`/`float`) → clamped into EvalPort's required `[0, 1]` range; if the original value was outside that range (Langfuse numeric scores have no guaranteed scale — a 1–5 rubric is just as valid as a 0–1 probability), the un-clamped original is preserved under `metadata["raw_value"]` so nothing is silently lost. Passes at `>= 0.5` after clamping, the convention every adapter in this repository uses when a tool doesn't supply its own pass/fail flag.
- **`CATEGORICAL`** (an arbitrary string label) → EvalPort has no numeric score to report, so `score` is `null`; `passed` is true only for a small, explicit set of affirmative labels (`"true"`, `"pass"`, `"correct"`, `"good"`, `"yes"`, `"1"`), and the raw label always survives under `metadata["value"]` even when that heuristic guesses wrong.

`Evaluation.comment` becomes the grader result's `reason`. `ExperimentItemResult.trace_id`/`dataset_run_id` are preserved under each result's own `metadata`. Evaluations scored against the whole run rather than one item (`ExperimentResult.run_evaluations` — e.g. a run-level regression check) have no single test case to attach to, so they're preserved under the ResultSet's top-level `metadata["langfuse"]["run_evaluations"]` instead of being dropped.

`experiment_result_to_openeval()` also accepts a bare list of `ExperimentItemResult` objects directly (skipping the `ExperimentResult` wrapper), for callers who already have the per-item results in hand.

## What round-trips losslessly, and what doesn't

Exporting a Langfuse dataset to EvalPort and back (`to_openeval` → `from_openeval` → `create_dataset_item`/`run_experiment`) is lossless: the full original dataset item — `id`, `status`, `dataset_id`, `dataset_name`, `source_trace_id`, and any custom `metadata` — is preserved verbatim under `metadata["langfuse"]["item"]` and reconstructed exactly on the way back. What doesn't survive a round trip through a *different* tool is that Langfuse-specific item shape — a receiving tool that isn't Langfuse only sees the generic `input`/`expected_output`/`metadata` fields, the same tradeoff every adapter in this repository makes (see the [UpTrain adapter](../uptrain-openeval-adapter) and [Weave adapter](../weave-openeval-adapter) for the same pattern).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
