# agenta-openeval-adapter

Convert [Agenta](https://github.com/Agenta-AI/agenta) testsets and evaluator invocation results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

Built at the explicit request of the Agenta maintainer [@mmabrouk](https://github.com/mmabrouk) on [Agenta-AI/agenta#6222](https://github.com/Agenta-AI/agenta/issues/6222): "we'd prefer the first option" — a standalone `agenta-openeval-adapter` package living in the EvalPort repo, depending on the `agenta` PyPI package as a normal dependency, rather than EvalPort support living inside Agenta's own core. This follows the same playbook that already worked for [CrewAI](../crewai-openeval-adapter) and [AutoGen](../autogen-openeval-adapter).

Unlike those two, `agenta` itself is a normal runtime dependency here (per the issue) rather than something the adapter merely works against from the outside — this package imports Agenta's real pydantic models (`agenta.sdk.models.testsets`, `agenta.sdk.models.workflows`) directly.

## Install

```bash
pip install "agenta-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/agenta-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Two Agenta object families, two conversions

Agenta splits *data* from *evaluation* into two separate object families, so this adapter exposes two focused functions rather than one:

* **`agenta_testset_to_suite()`** — an Agenta testset (`agenta.testsets.TestsetRevision`, wrapping a `TestsetRevisionData` of free-form `Testcase` rows) → an EvalPort `EvalSuite`.
* **`invocations_to_resultset()`** — a collection of Agenta evaluator invocations (`WorkflowInvokeRequest` / `WorkflowBatchResponse` pairs — Agenta invokes an evaluator once per test case per grader, there's no single "whole run" object) → an EvalPort `ResultSet`.

A dispatching `to_openeval(obj, **kwargs)` routes to whichever of the two fits the object you pass it, for parity with other EvalPort adapters' single-entry-point convention.

## Usage

### Testset → EvalPort suite

```python
from agenta_openeval_adapter import agenta_testset_to_suite
from openeval.validate import validate_suite

# testset_revision is whatever agenta.testsets.afetch()/aretrieve() returned
# (a TestsetRevision), or a plain dict of the same shape.
suite = agenta_testset_to_suite(testset_revision)
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

Each Agenta `Testcase.data` is a free-form dict — Agenta testsets have no fixed column schema. `agenta_testset_to_suite()` auto-detects conventional column names (`input`/`query`/`question`/`prompt` for the input, `expected_output`/`expected`/`ground_truth`/... for the expected output, `context`/`contexts`/`retrieval_context` for context), or you can name them explicitly:

```python
suite = agenta_testset_to_suite(testset_revision, input_key="question", expected_output_key="answer")
```

Any column not claimed by one of those three roles is preserved under `test_case.metadata.agenta_testcase` rather than dropped, so `from_openeval()` can restore it losslessly.

By default `agenta_testset_to_suite()` generates an `llm_judge` grader for output quality. Pass `grader_type="exact_match"` if your testset's expected outputs really are meant to match literally:

```python
suite = agenta_testset_to_suite(testset_revision, grader_type="exact_match")
```

### Evaluator invocations → EvalPort result set

Agenta invokes an evaluator once per (test case, grader) pair — collect the `(request, response)` pairs your own runner produces, then convert the whole batch at once:

```python
from agenta_openeval_adapter import invocations_to_resultset
from openeval.validate import validate_result_set

invocations = [
    {
        "test_case_id": "11111111-1111-1111-1111-111111111111",
        "grader_id": "gr_correctness",
        "request": invoke_request,     # the WorkflowInvokeRequest you sent
        "response": invoke_response,   # the WorkflowBatchResponse you got back
    },
    # ... one entry per (test case, grader) invocation
]

resultset = invocations_to_resultset(invocations, suite_id=suite["id"], run_id="run_001")
assert validate_result_set(resultset).valid
```

`response.data.outputs` (the evaluator's verdict — a bool, a number, or a dict like `{"score": 0.8, "passed": true, "reason": "..."}`, since Agenta evaluators are user-defined code) becomes the EvalPort `GraderResult`. When `request` is given, `request.data.outputs` (what Agenta handed the evaluator to grade) becomes `Result.actual_output`, and `request.data.testcase["id"]` is used as a fallback `test_case_id`. Multiple invocations sharing a `test_case_id` are grouped into one `Result` with one `GraderResult` per grader — mirroring `ResultSet.results[].grader_results[]` in the spec. A `status.code >= 400` on the response (the evaluator call itself erroring, as opposed to running fine and grading the test case as failing) is surfaced as `Result.error`.

### Reverse: EvalPort suite → Agenta testset

```python
from agenta_openeval_adapter import from_openeval

testset_data = from_openeval(suite)
# {"testcase_ids": None, "testcases": [{"id": "...", "data": {"input": ..., "expected_output": ...}}, ...]}
```

Returns plain dicts (not committed Agenta objects), since minting an actual Agenta testset also requires calling `agenta.testsets.acreate()`/`aedit()` against a live project — feed the dicts into `Testcase(**tc)` / `TestsetRevisionData(**data)` yourself, or straight into `acreate()`.

### One-call dispatcher

```python
from agenta_openeval_adapter import to_openeval

suite = to_openeval(testset_revision)                              # -> EvalSuite
resultset = to_openeval(invocations, suite_id=suite["id"])          # -> ResultSet
```

## A note on the real Agenta API

This package was built against the actually-installed `agenta` PyPI package (0.113.0), not just the issue's description of it. The issue's summary is accurate on the object *shapes* (`TestsetRevision`/`TestsetRevisionData`, `WorkflowInvokeRequest`, `WorkflowBatchResponse`/`WorkflowStreamingResponse`) but describes them as reachable via `agenta.evaluator.invoke()`. In the installed package, `agenta.evaluator` is a **decorator** for defining your own evaluator function, not an object with a callable `.invoke()`; the actual invocation entry point is a free function (`agenta.sdk.decorators.running.invoke_evaluator`), and the request/response types live under `agenta.sdk.models.workflows` rather than the top-level `agenta` namespace. See this package's module docstring for the full detail.

## Credit

Built at the explicit request of [@mmabrouk](https://github.com/mmabrouk) on [Agenta-AI/agenta#6222](https://github.com/Agenta-AI/agenta/issues/6222).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
