# beeai-openeval-adapter

Convert [BeeAI Framework](https://github.com/i-am-bee/beeai-framework)
evaluation datasets and agent run outputs (`AgentOutput`, and
`RequirementAgentOutput` from `RequirementAgent`) to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

Filed and built following
[i-am-bee/beeai-framework#1677](https://github.com/i-am-bee/beeai-framework/issues/1677),
where the BeeAI maintainer said "the standalone package in EvalPort's own
`adapters/` directory is the stronger option" over an in-tree exporter, and
asked that it be tested against `RequirementAgent` specifically.

## Why a standalone package?

BeeAI's in-tree `beeai_framework.evaluation` package only ships judge-LLM
adapters (`DeepEvalLLM`, `InstructorRagasLLM`) that let DeepEval and Ragas
metrics call a BeeAI `ChatModel` -- the framework itself has no dataset,
test-case or evaluation-result model to hang an export hook on. The shapes
BeeAI users actually produce live in its examples: the dataset items in
`python/examples/evaluation/dataset.json` and the `AgentOutput` /
`RequirementAgentOutput` that `agent.run()` returns, which
`examples/evaluation/{deepeval,ragas}/experiment.py` then hand to external
metrics. This package converts exactly those shapes (read from the framework's
source at `beeai-framework==0.1.84`, not guessed), duck-typed so it works on
the real pydantic objects or on plain dicts with the same attribute names,
and it never imports `beeai_framework` at module import time -- so it adds
no dependency to BeeAI and BeeAI adds none to it.

## Install

```
pip install "beeai-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/beeai-openeval-adapter"
```

Not yet published to PyPI -- this installs directly from source via pip's
`git+`/`#subdirectory=` support. Add `[beeai]` to also pull in
`beeai-framework` itself (pinned `>=0.1.84,<1`, the version this was tested
against; it requires Python 3.11+).

## Usage

### Dataset items -> EvalPort Suite

```python
from beeai_openeval_adapter import to_openeval, from_openeval

# The shape of python/examples/evaluation/dataset.json, as returned by
# examples/evaluation/dataset.py:load_items()
items = [
    {
        "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "expected_answer": "Delhi",
        "supporting_sentences": [
            "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
            "The Oberoi Group is a hotel company with its head office in Delhi.",
        ],
        "expected_tool_calls": 2,
        "supporting_titles": ["Oberoi family", "The Oberoi Group"],
    },
]

# expected_tool= is optional: the dataset only records a call *count*, and
# both BeeAI experiments hardcode the tool as "Wikipedia".
suite = to_openeval(items, suite_id="beeai_rag_multi_hop", expected_tool="Wikipedia")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

items_again = from_openeval(suite)  # back to the dataset.json item shape
```

`from_openeval()` converts every test case in a suite, not only ones this
package produced, so any EvalPort suite can be fed to
`examples/evaluation/agent.py`'s `RequirementAgent` through BeeAI's own
dataset loader unchanged.

### Agent runs -> EvalPort ResultSet

This is the direction
[#1677](https://github.com/i-am-bee/beeai-framework/issues/1677) is about:
exporting what `await agent.run(question)` returned. One `Result` per run;
`results_to_openeval()` batches them and computes `summary`.

```python
import asyncio
from datetime import datetime, timezone

from beeai_openeval_adapter import result_to_openeval, results_to_openeval
from examples.evaluation.agent import create_agent  # RequirementAgent

async def main() -> None:
    agent = create_agent()
    runs = []
    for tc in suite["test_cases"]:
        try:
            output = await agent.run(tc["input"])           # RequirementAgentOutput
            runs.append(result_to_openeval(tc["id"], output, expected_output=tc.get("expected_output")))
        except Exception as exc:                            # a genuine run failure
            runs.append(result_to_openeval(tc["id"], None, expected_output=tc.get("expected_output"), error=exc))

    result_set = results_to_openeval(
        runs,
        suite_id=suite["id"],
        run_id="run_2026_09_27",
        started_at=datetime.now(timezone.utc).isoformat(),
        model=agent._llm.model_id if hasattr(agent, "_llm") else None,
    )
    from openeval.validate import validate_result_set
    assert validate_result_set(result_set).valid

asyncio.run(main())
```

When you have already scored the run with DeepEval or Ragas (as
`examples/evaluation/deepeval/experiment.py` does), pass the metric results
instead of `expected_output` and they are carried through verbatim:

```python
result = result_to_openeval(
    tc["id"],
    output,
    grader_results=test_result.metrics_data,  # DeepEval MetricData objects, or any
                                              # dicts/objects with name/score/passed
)
```

With neither `grader_results` nor `expected_output`, the result is emitted as
explicitly unscored (`score: null`, `passed: false`, grader
`gr_beeai_unscored`) rather than silently passing.

## Mapping

Dataset item (`examples/evaluation/dataset.json`) -> `TestCase`:

| BeeAI | EvalPort | Notes |
|---|---|---|
| `id` (or `test_case_id`, `name`) | `id` | `tc_<index>` when absent |
| `question` (or `input`) | `input` | required |
| `expected_answer` (or `expected_output`, `answer`) | `expected_output` | a list keeps `[0]` here and the full list in `metadata.beeai.acceptable_answers` |
| `supporting_sentences` (or `context`) | `context` | |
| `expected_tool_calls` (int) | `metadata.beeai.expected_tool_calls`; `expected_tools = [expected_tool] * n` only when `expected_tool=` is passed | the dataset never names the tool, so none is invented |
| `supporting_titles` | `metadata.beeai.supporting_titles` | |
| `expected_tools` (Golden-style list) | `expected_tools` | names taken from `.name`/`.tool_name`/str |
| `tags` | `tags` | |
| any other key | `metadata.beeai.extra.<key>` | lossless; restored by `from_openeval()` |
| -- | `graders: ["gr_beeai_exact_match"]` | the `ExactMatch` metric both BeeAI experiments run first |
| -- | `metadata.openeval.source = "beeai-framework"` | suite-level tag |

`AgentOutput` / `RequirementAgentOutput` -> `Result`:

| BeeAI | EvalPort | Notes |
|---|---|---|
| `last_message.text` | `actual_output` | falls back to JSON of `output_structured` when the final message has no text; omitted when `output` is `None` (run raised) |
| `output` (messages) | `metadata.beeai.output` | via `Message.to_plain()` |
| `context` | `metadata.beeai.context` | |
| `output_structured` | `metadata.beeai.output_structured` | pydantic models via `model_dump(mode="json")` |
| `state.iteration`, `state.usage`, `state.cost` | `metadata.beeai.state.*` | `RequirementAgentOutput` only |
| `state.steps` | `metadata.beeai.state.steps[]` | `id`, `iteration`, `tool.name`, `input`, `error`; the live `Tool`/`ToolOutput` instances are not serialized |
| `state.memory.messages` | `metadata.beeai.state.trajectory` | full trajectory via `to_plain()` |
| tool-call content parts in the trajectory (or `output`) | `metadata.beeai.tool_calls[]`, `metadata.beeai.tools_called[]` | same extraction as `experiment.py::extract_real_tool_calls`; `tools_called` skips `RequirementAgent`'s internal `final_answer` tool like BeeAI's own `count_tool_usage()`, `tool_calls` keeps it |
| caller's `expected_output` | `grader_results[0]` (`gr_beeai_exact_match`, `score` 1.0/0.0) | strict `strip()` equality |
| caller's `grader_results` (DeepEval `MetricData`, Ragas `MetricResult`, dicts) | `grader_results[]` | `grader_id` from `grader_id` or `gr_beeai_<slug(name)>`; `type` defaults to `custom`; `passed` from `passed`/`success`, else `score >= threshold`; out-of-range scores clamped with `metadata.raw_score` |
| neither | `grader_results[0]` = `gr_beeai_unscored`, `score: null`, `passed: false` | |
| exception from `agent.run()` | `error.type` + `error.message` | `ChatModelError`/`BackendError` -> `provider_error`, `*Timeout*` -> `timeout`, else `runner_error`; class name and `FrameworkError.explain()` under `metadata.beeai.error` |
| caller's `duration_ms`, `completed_at`, `attempt` | same | `duration_ms` rounded to int |

`results_to_openeval()` adds `summary` (`total`, `passed`, `failed`,
`pass_rate`, `avg_score` over non-null scores, `duration_ms` when every result
has one, `by_grader`), `runner = {"name": "beeai-framework", "version":
<installed>}` (version via `importlib.metadata`, omitted if not installed),
`provider.model` from `model=`, and optional `isolation`/`completed_at`.

Lossy fields: `RequirementAgentRunState.memory` (the `BaseMemory` instance
itself) and `steps[].tool`/`steps[].output` (live `Tool`/`ToolOutput`
objects) are reduced to their messages / tool name / string form. Nothing
else is dropped.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 - see LICENSE.
