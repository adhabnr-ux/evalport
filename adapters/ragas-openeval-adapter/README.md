# ragas-openeval-adapter

Convert [Ragas](https://github.com/explodinggradients/ragas) evaluation results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

This follows the same playbook that already worked for [autogen-openeval-adapter](../autogen-openeval-adapter) and [crewai-openeval-adapter](../crewai-openeval-adapter): it works against Ragas's public `EvaluationResult` shape (the object returned by `ragas.evaluate()`, via its documented `.to_pandas()` method) from the outside, so you get EvalPort import/export today without needing anything merged into Ragas's core. If native EvalPort support ever lands in Ragas, this package still works — it just becomes optional.

## Install

```bash
pip install ragas-openeval-adapter
```

## Usage

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas_openeval_adapter import to_openeval, from_openeval

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

suite = to_openeval(result)

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and the other direction: load an EvalPort suite as a fresh Ragas dataset
from datasets import Dataset
samples = from_openeval(suite)
dataset = Dataset.from_list(samples)
```

Each Ragas metric present on a sample (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `answer_correctness`, and others) becomes its own EvalPort grader (`gr_<metric>`, type `custom`, handler `ragas:<metric>`), and the scores Ragas already computed are preserved per test case under `metadata.ragas_scores` — `evaluate()` output is scored data, not just a task definition, so nothing is thrown away on the way in.

## Credit

Tracked as [evalport#1](https://github.com/adhabnr-ux/evalport/issues/1).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
