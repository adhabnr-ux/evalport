# mlflow-openeval-adapter

Convert [MLflow](https://mlflow.org) `mlflow.evaluate()` results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

This follows the same playbook that already worked for [autogen-openeval-adapter](../autogen-openeval-adapter), [crewai-openeval-adapter](../crewai-openeval-adapter), [ragas-openeval-adapter](../ragas-openeval-adapter), [langsmith-openeval-adapter](../langsmith-openeval-adapter), and [braintrust-openeval-adapter](../braintrust-openeval-adapter): it works against MLflow's public `EvaluationResult` shape (`.metrics` and `.tables["eval_results_table"]`) from the outside, so you get EvalPort import/export today without needing anything merged into the `mlflow` package.

## Install

```bash
pip install "mlflow-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/mlflow-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

```python
import mlflow
from mlflow_openeval_adapter import to_openeval, from_openeval

result = mlflow.evaluate(
    model=my_model,
    data=eval_data,
    targets="targets",
    extra_metrics=[mlflow.metrics.exact_match()],
)

suite = to_openeval(result, run_id="my-eval-run")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and the other direction: load an EvalPort suite as a fresh eval dataframe
import pandas as pd
rows = from_openeval(suite)
mlflow.evaluate(model=my_model, data=pd.DataFrame(rows), targets="targets")
```

Every per-row metric-score column MLflow produces (e.g. `"exact_match/v1/score"`, `"toxicity/v1/score"`) becomes its own EvalPort grader (`gr_<metric>`, type `custom`, handler `mlflow:<column>`), and the scores MLflow already computed are preserved per test case under `metadata.mlflow_scores`. The run's aggregate metrics (`result.metrics`) are preserved at the suite level under `metadata.mlflow_metrics` — an `evaluate()` run is already-scored data, not just a task definition, so nothing is thrown away on the way in.

## Credit

Tracked as [evalport#4](https://github.com/adhabnr-ux/evalport/issues/4).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
