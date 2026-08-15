# patronus-openeval-adapter

Convert [Patronus AI](https://github.com/patronus-ai/patronus-python) (`patronus.evals`) evaluators, evaluation inputs, and evaluation results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install patronus-openeval-adapter
```

This installs `evalport-sdk` as its only hard dependency. The `patronus` package itself is **not** installed automatically -- add it separately (`pip install patronus`).

## Usage

### Evaluation inputs + evaluators → EvalPort suite, and back

```python
from patronus.evals import RemoteEvaluator
from patronus_openeval_adapter import to_openeval, from_openeval

from my_app.evaluators import ExactMatchEvaluator  # your own local Evaluator subclass

suite = to_openeval(
    inputs=["What is the capital of France?"],
    evaluators={
        "exact_match": ExactMatchEvaluator(),
        "judge": RemoteEvaluator("judge", criteria="conciseness"),
    },
    expected_outputs=["Paris"],
    suite_id="geo_facts",
)

from openeval.validate import validate_suite
assert validate_suite(suite).valid

# ...and back: rebuild everything needed to run the evaluators for real.
rebuilt = from_openeval(suite)
my_app_outputs = [my_app(q) for q in rebuilt["inputs"]]  # run your own system under test

results = {
    name: [
        ev.evaluate(
            task_input=rebuilt["inputs"][i],
            task_output=my_app_outputs[i],
            gold_answer=rebuilt["expected_outputs"][i],
            task_context=rebuilt["contexts_list"][i],
        )
        for i in range(len(rebuilt["inputs"]))
    ]
    for name, ev in rebuilt["evaluators"].items()
}
```

`to_openeval()` takes exactly the shape needed to *define* an evaluation run -- one `task_input` string per test case, plus a `{name: Evaluator}` mapping -- deliberately excluding the responses being graded, since those don't exist yet at suite-definition time (they come from whatever system you're testing, captured afterward). Every evaluator in the mapping becomes one EvalPort grader, applied to every test case, matching how a Patronus experiment runs every evaluator against every input uniformly.

### Evaluation results → EvalPort ResultSet

```python
from patronus_openeval_adapter import batch_eval_result_to_openeval

result_set = batch_eval_result_to_openeval(
    results,                           # {name: [EvaluationResult, ...]}, aligned with rebuilt["ids"]
    test_case_ids=rebuilt["ids"],
    evaluators=rebuilt["evaluators"],
    suite_id="geo_facts",
    run_id="run-2026-08-15",
    started_at="2026-08-15T00:00:00Z",
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

`evaluators` is required here, not optional: an `EvaluationResult` carries no record of which evaluator produced it, so `batch_eval_result_to_openeval()` needs the same `{name: Evaluator}` mapping the run used, in order to know whether each grader was `llm_judge` (a `RemoteEvaluator`) or `custom` (everything local), matching `to_openeval()`'s own mapping.

## Grader mapping

| Patronus evaluator | EvalPort grader `type` | Direction |
|---|---|---|
| `RemoteEvaluator` / `AsyncRemoteEvaluator` | `llm_judge` (`params.model` = `"patronus-hosted-judge"` placeholder) | both |
| `Evaluator` / `StructuredEvaluator` subclasses (local, deterministic code) | `custom` (`params.handler` names the evaluator's `canonical_name`) | export only |
| `@patronus.evaluator()`-wrapped functions | `custom` | export only |

Patronus's evaluator surface splits cleanly into two kinds, and this adapter maps each honestly rather than force-fitting both into one grader type. `RemoteEvaluator`s are Patronus's hosted, criteria-driven judges (`evaluator_id_or_alias` like `"judge"`, `"lynx"`, `"hallucination"`) -- LLM-as-judge under the hood, so they map onto `llm_judge`. Everything else is local, deterministic code with no fixed shape EvalPort can generically interpret, so it maps onto `custom` and is **export only**, per the spec's "Custom grader handling" rule -- there is no safe, generic way to reconstruct arbitrary evaluator code from a grader record on import (the same reasoning `code`/`human` graders clean-skip across the whole EvalPort ecosystem, and the same choice the [LlamaIndex adapter](../llamaindex-openeval-adapter#grader-mapping) makes for `PairwiseComparisonEvaluator`).

## What round-trips losslessly, and what doesn't

A `RemoteEvaluator`'s identity (`evaluator_id_or_alias`, `criteria`, `explain_strategy`, `tags`) round-trips exactly -- `from_openeval()` reconstructs the exact same `RemoteEvaluator` construction, not a generic stand-in, whenever the grader carries this adapter's own `metadata.patronus` (`openeval.*` is the only reserved metadata prefix; everything else preserved this way follows the same convention every adapter in this repo uses). A hand-authored `llm_judge` grader with no `metadata.patronus.evaluator_id_or_alias` -- e.g. one written directly as JSON, or produced by a different adapter -- is honestly skipped on import rather than guessed at.

Two spots are honestly approximate, both documented directly in the source:

- **`llm_judge` prompts and the `model` field.** A `RemoteEvaluator` doesn't expose a literal prompt string or model id -- the judge model and prompt template are resolved server-side by `evaluator_id_or_alias`/`criteria` at call time (calling `.get_criteria()` on an unloaded evaluator even requires a live API round trip to resolve a criteria revision, which this adapter deliberately avoids just to build a suite -- it reads the raw, pre-load `criteria` attribute instead). `to_openeval()` synthesizes an honest, human-readable rubric description naming the evaluator id and criteria, always including the `{output}`/`{input}`/`{expected}` tokens EvalPort's real validator requires `llm_judge` prompts to contain, rather than fabricating a fake extracted prompt. `params.model` is filled with the placeholder `"patronus-hosted-judge"` -- the same kind of gap documented in the [Giskard](../giskard-openeval-adapter#what-round-trips-losslessly-and-what-doesnt) and [LlamaIndex](../llamaindex-openeval-adapter#what-round-trips-losslessly-and-what-doesnt) adapters for their own server/session-resolved LLM judges.
- **`EvaluationResult.score`/`pass_`.** Patronus evaluators may set only one of the two. `batch_eval_result_to_openeval()` derives the other honestly (`score >= pass_threshold` when only `score` is set; `1.0`/`0.0` when only `pass_` is set) rather than fabricating an exact duplicate, and if *neither* is set, the grader result carries `score: null` and `passed: false` -- never a fabricated pass. `text_output`, `explanation`, `tags`, `dataset_id`, and `dataset_sample_id` -- Patronus-specific fields with no EvalPort equivalent -- are preserved verbatim under `grader_result.metadata.patronus`.

`from_openeval()` also rejects (with a clear `ValueError` naming the offending test case) any test case whose `input` is EvalPort's multi-turn array-of-strings form -- Patronus's `task_input` is a single string, and silently collapsing an array into one string would misrepresent the input rather than fail loudly.

## Running the tests

```bash
cd adapters/patronus-openeval-adapter
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest tests/ -v
```

30 tests, all running against the real `patronus.evals` classes (`Evaluator`, `RemoteEvaluator`, `EvaluationResult`) and the real `openeval.validate.validate_suite()`/`validate_result_set()` -- not mocks. `RemoteEvaluator` is instantiated (its constructor makes no network call, confirmed by reading `patronus/evals/evaluators.py` directly) but never `.evaluate()`'d, since that requires a live Patronus API key. `TestEndToEndWithRealEvaluator` drives the whole pipeline end to end with a real, local, deterministic `Evaluator` subclass actually being run: `to_openeval()` → `from_openeval()` → real `evaluator.evaluate(...)` calls → `batch_eval_result_to_openeval()` → `validate_result_set()`.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
