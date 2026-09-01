# crewai-openeval-adapter

Convert [CrewAI](https://github.com/crewAIInc/crewAI) crew/task evaluation results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

A proposal for native EvalPort support has been open on [crewAIInc/crewAI#6711](https://github.com/crewAIInc/crewAI/issues/6711) since July 2026 with no maintainer engagement yet. Rather than block on that review landing, this package follows the same playbook that already worked for AutoGen ([autogen-openeval-adapter](../autogen-openeval-adapter)): it works against CrewAI's public `Task` / `TaskOutput` / `Crew` shapes (objects or dicts) from the outside, so you get EvalPort import/export today without needing anything merged into CrewAI's core. If native support lands in CrewAI later, this package still works — it just becomes optional.

## Install

```bash
pip install "crewai-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/crewai-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

```python
from crewai_openeval_adapter import to_openeval, from_openeval

# crew_result is whatever your Crew.kickoff() returned (CrewOutput),
# or a plain dict with the same shape.
suite = to_openeval(crew_result)

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# ...and the other direction: load an EvalPort suite as CrewAI task dicts
tasks = from_openeval(suite)
```

By default, `to_openeval()` generates an `llm_judge` grader for output quality — CrewAI's `expected_output` is conventionally a natural-language description ("a 3-bullet summary"), not a literal string, so exact matching would almost always fail. Pass `grader_type="exact_match"` if your tasks really do expect literal output:

```python
suite = to_openeval(crew_result, grader_type="exact_match")
```

When a task has `tools` (read directly off the task, or off `task.agent.tools`), an additional `gr_tool_selection` grader is attached and `expected_tools` is populated, so tool-selection accuracy can be checked independently of output text by whichever EvalPort runner executes the suite.

## Credit

This adapter follows the same standalone-package approach as [autogen-openeval-adapter](../autogen-openeval-adapter). It's informed by the real proposal thread at [crewAIInc/crewAI#6711](https://github.com/crewAIInc/crewAI/issues/6711) (opened July 2026 — the thread got a "would a contributor like to pick this up?" comment, but no contributor ever did) and tracked as [evalport#5](https://github.com/adhabnr-ux/evalport/issues/5). No CrewAI maintainer or contributor reviewed or shaped this specific mapping — it was designed and built independently against CrewAI's public `Task`/`TaskOutput`/`Crew` shapes, the same way the rest of this repo's standalone adapters are.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
