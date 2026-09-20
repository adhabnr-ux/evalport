# agency-swarm-openeval-adapter

Convert the results of [agency-swarm](https://github.com/VRSEN/agency-swarm)'s
`Agency.get_response()` calls to
[EvalPort](https://github.com/adhabnr-ux/evalport) `Result`/`ResultSet`
documents, so a regression suite of `(message, recipient_agent) ->
expected_output` test cases can be tracked as portable, structured eval data
instead of ad hoc assertions scattered across a test file.

## Why this is a standalone package, not an agency-swarm core change

Maintainer [@nicko-ai](https://github.com/nicko-ai) closed the proposal
([VRSEN/agency-swarm#772](https://github.com/VRSEN/agency-swarm/issues/772))
with:

> Thanks for the proposal. Third-party integrations live outside this
> repository — at most we ship a docs recipe. This adapter is pure glue on
> top of `Agency.get_response` and needs no core change, so it works best as
> an external package. Closing; we can link it from the docs if you publish
> it.

## No runtime dependency on agency-swarm or the OpenAI Agents SDK

`Agency.get_response()` returns a `RunResult` dataclass from the `agents`
package (the OpenAI Agents SDK agency-swarm is built on -- grounded against
`src/agency_swarm/agency/responses.py`), but this module never imports
`agents` or `agency_swarm` at runtime. Every attribute read
(`final_output`, `context_wrapper.usage`, `last_agent`, the four guardrail
result lists) goes through `getattr()` with a safe default, so this adapter
works against a real `RunResult`, a hand-built test double, or a future SDK
version that renames or removes a field, without pinning to either
package's release cadence.

Tests DO depend on the real `agents` package (see the `test` extra) to
construct genuine `RunResult`/`Agent`/`Usage`/guardrail-result instances for
fidelity, rather than hand-rolled stand-ins — but that is a test-only
dependency, never a runtime one.

## What this adapter does NOT do

It does not call `Agency.get_response()` itself, and it does not touch
`communication_flows` or agent routing. The caller runs their own suite —
awaiting `get_response()` calls however they like: sequentially,
concurrently, from pytest, from a script — and hands each
`(test_case, RunResult)` pair (or `(test_case, exception)` on failure) to
this adapter. This is deliberately "pure glue on top of
`Agency.get_response`", exactly as the maintainer scoped it: agency
construction, `communication_flows`, and how a suite is actually run stay
entirely the caller's responsibility.

## Install

```bash
pip install -e .
```

Requires `evalport-sdk>=1.0.0`. No agency-swarm or openai-agents dependency
at runtime.

## Usage

```python
import asyncio
from datetime import datetime, timezone

from agency_swarm import Agency, Agent
from agency_swarm_openeval_adapter import build_result_set
from openeval.validate import validate_result_set

ceo = Agent(name="CEO", instructions="Route requests to the right specialist.")
developer = Agent(name="Developer", instructions="Write code.")
agency = Agency(ceo, communication_flows=[(ceo, developer)])

test_cases = [
    {"id": "tc1", "input": "What's 2+2?", "recipient_agent": "CEO", "expected_output": "4"},
    {"id": "tc2", "input": "Write a hello world function", "recipient_agent": "Developer"},
]

async def run_suite():
    runs = []
    for tc in test_cases:
        try:
            result = await agency.get_response(tc["input"], tc["recipient_agent"])
            runs.append({"test_case": tc, "result": result})
        except Exception as exc:
            runs.append({"test_case": tc, "exception": exc})
    return runs

runs = asyncio.run(run_suite())
result_set = build_result_set(
    runs,
    suite_id="my-agency-suite",
    run_id="run-1",
    started_at=datetime.now(timezone.utc).isoformat(),
)
assert validate_result_set(result_set).valid
```

`build_test_suite()` builds the corresponding EvalPort suite document
(the *definition* of the test cases, independent of any particular run):

```python
from agency_swarm_openeval_adapter import build_test_suite
from openeval.validate import validate_suite

suite = build_test_suite(test_cases, suite_id="my-agency-suite")
assert validate_suite(suite).valid
```

## Grading

Unlike this repo's other adapters, there's no on-disk or in-SDK "evaluation
config" to translate — agency-swarm doesn't score its own runs. This
adapter defaults to what the original proposal specified: an `exact_match`
grader comparing `final_output` (coerced to a string; see
`_coerce_output_to_str()` for how structured/Pydantic outputs are handled)
against a test case's `expected_output`, when one is supplied.

A test case with **no** `expected_output` gets a placeholder `custom`
grader (`agency_swarm:no_expected_output`) rather than a fabricated
pass/fail. Callers who want real scoring — an LLM judge, a
domain-specific check, an assertion on which agent ultimately handled the
request — pass their own grader callables:

```python
def handoff_grader(test_case, result):
    if "expected_last_agent" not in test_case:
        return None
    matched = result.last_agent.name == test_case["expected_last_agent"]
    return {
        "grader_id": f"{test_case['id']}__handoff",
        "type": "custom",
        "score": 1.0 if matched else 0.0,
        "passed": matched,
        "metadata": {"handler": "agency_swarm:last_agent_check"},
    }

result_set = build_result_set(runs, suite_id="...", run_id="...", started_at="...",
                               graders=[default_exact_match_grader, handoff_grader])
```

Nothing in this module calls an LLM or invents a score.

## Guardrails and the `passed` field

`Agency.get_response()` can trip an input, output, tool-input, or
tool-output guardrail. If any guardrail's `tripwire_triggered` is true, the
Result's `passed` is forced to `False` regardless of grader scores — a
blocked run is a failed run. Otherwise `passed` follows the supplied
graders (`all()` of their individual `passed` flags); with no graders and
no tripped guardrail, `passed` defaults to `True`, meaning only "the call
completed without raising and without a guardrail block" — this is **not**
a correctness signal, and is documented as such so it's never mistaken for
one.

## What's preserved in metadata

Every `Result.metadata["agency_swarm"]` carries: `recipient_agent` (from
the test case), `last_agent` (which agent in the agency actually produced
`final_output`, useful for asserting on handoff behavior),
`usage` (`requests`/`input_tokens`/`output_tokens`/`total_tokens` from
`RunResult.context_wrapper.usage`), `new_items_count` and
`raw_responses_count` (sizes only, not the full transcript — those can be
large and contain tool call payloads), and per-guardrail `name` /
`tripwire_triggered` / `output_info` for all four guardrail categories.

## Lossiness of `suite_to_test_cases()`

The reverse direction only recovers what `build_test_suite()` itself put
into `metadata["agency_swarm"]`. It is not a general importer for suites
built by other EvalPort producers, which have no reason to carry that
block.

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v
```

Tests construct real `agents.RunResult`/`Agent`/`Usage`/guardrail-result
objects (not mocks) and validate every produced document against the real
`openeval.validate.validate_suite()` / `validate_result_set()`.
