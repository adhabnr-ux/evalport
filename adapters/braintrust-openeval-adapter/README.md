# braintrust-openeval-adapter

Convert [Braintrust](https://www.braintrust.dev) `Eval()` results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

This follows the same playbook that already worked for [autogen-openeval-adapter](../autogen-openeval-adapter), [crewai-openeval-adapter](../crewai-openeval-adapter), [ragas-openeval-adapter](../ragas-openeval-adapter), and [langsmith-openeval-adapter](../langsmith-openeval-adapter): it works against Braintrust's public `Eval()` result shape (each case exposing `input`/`expected`/`output`/`scores`) from the outside, so you get EvalPort import/export today without needing anything merged into the `braintrust` SDK.

## Install

```bash
pip install "braintrust-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/braintrust-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

```python
from braintrust import Eval
from autoevals import Factuality
from braintrust_openeval_adapter import to_openeval, from_openeval

result = Eval(
    "my-project",
    data=lambda: [{"input": "What is 2+2?", "expected": "4"}],
    task=my_task,
    scores=[Factuality],
)

suite = to_openeval(result)

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and the other direction: load an EvalPort suite as Braintrust eval cases
cases = from_openeval(suite)
Eval("my-project", data=cases, task=my_task, scores=[Factuality])
```

Every scorer present on a case (`Factuality`, `ExactMatch`, or a custom scorer function's name) becomes its own EvalPort grader (`gr_<name>`, type `custom`, handler `braintrust:<name>`), and the scores Braintrust already computed are preserved per test case under `metadata.braintrust_scores` — an `Eval()` run is already-scored data, not just a task definition, so nothing is thrown away on the way in.

## Credit

Tracked as [evalport#3](https://github.com/adhabnr-ux/evalport/issues/3).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
