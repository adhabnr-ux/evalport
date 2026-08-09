# langsmith-openeval-adapter

Convert [LangSmith](https://smith.langchain.com) experiment results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

This follows the same playbook that already worked for [autogen-openeval-adapter](../autogen-openeval-adapter), [crewai-openeval-adapter](../crewai-openeval-adapter), and [ragas-openeval-adapter](../ragas-openeval-adapter): it works against LangSmith's public `Run`/`Feedback` shapes (objects or dicts) from the outside, so you get EvalPort import/export today without needing anything merged into the `langsmith` SDK.

## Install

```bash
pip install langsmith-openeval-adapter
```

## Usage

```python
from langsmith import Client
from langsmith_openeval_adapter import to_openeval, from_openeval

client = Client()
runs = list(client.list_runs(project_name="my-project", is_root=True))
run_ids = [r.id for r in runs]
feedback = list(client.list_feedback(run_ids=run_ids))

suite = to_openeval(runs, feedback=feedback, run_id="my-project")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and the other direction: load an EvalPort suite as a fresh LangSmith dataset
examples = from_openeval(suite)
dataset = client.create_dataset("from-evalport")
client.create_examples(dataset_id=dataset.id, examples=examples)
```

Every distinct feedback key found across your runs (e.g. `"correctness"`, `"helpfulness"`) becomes its own EvalPort grader (`gr_<key>`, type `custom`, handler `langsmith:<key>`), and the scores LangSmith already recorded are preserved per test case under `metadata.langsmith_feedback` — an experiment run is already-scored data, not just a task definition, so nothing is thrown away on the way in. Run `inputs`/`outputs` dicts are flattened to a single string using common field names (`input`/`question`/`query`/`prompt`, `output`/`answer`/`response`) when present, and JSON-serialized otherwise so multi-field chains don't lose data.

## Credit

Tracked as [evalport#2](https://github.com/adhabnr-ux/evalport/issues/2).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
