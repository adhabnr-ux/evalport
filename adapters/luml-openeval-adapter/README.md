# luml-openeval-adapter

Convert [luml](https://github.com/luml-ai/luml)'s `EvalItem` / `EvalResult` / `EvalResults` evaluation types to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

A proposal for a `luml-openeval-adapter` was opened on [luml-ai/luml#632](https://github.com/luml-ai/luml/issues/632) with a full field-mapping sketch. Maintainer [OKUA1](https://github.com/OKUA1) (Oleh Kostromin) replied:

> You're of course free to implement and maintain any adapter you find useful in your own repository. Since this doesn't require any changes on our side, I'll close this issue.

This package is that adapter, built exactly as promised in the follow-up comment on that issue — against the real, public `EvalItem`/`EvalResult`/`EvalResults` dataclasses (read directly from `luml-ai/luml`'s source, not the original sketch's guesses), the same standalone playbook that already worked for [mlflow-openeval-adapter](../mlflow-openeval-adapter) and the rest of this repo's adapters.

luml's real package (`luml_sdk`) requires Python ≥3.12 and is not published to PyPI, so it isn't installable as a dependency here — this adapter works against the public dataclass *shape* via duck typing instead (see the module docstring in `src/luml_openeval_adapter/__init__.py` for exactly which luml source files this was verified against).

## Install

```bash
pip install "luml-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/luml-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Two directions, matching luml's two types

luml's evaluation types split cleanly into an *unscored definition* and *already-scored evidence*, and this adapter keeps that split rather than flattening it into one lossy conversion:

| luml (`luml.experiments.evaluation.types`) | EvalPort (`openeval.types`) | This adapter |
|---|---|---|
| `EvalItem(id, inputs, expected_output, metadata)` | `TestCase` / `EvalSuite` | `eval_item_to_test_case()` / `to_openeval_suite()` |
| `EvalResult(eval_item, model_response, scores, trace_id)` + `EvalResults(results, aggregated_scores, dataset_id)` | `Result` / `GraderResult` / `ResultSet` | `to_openeval()` (primary) |

### Exporting a completed evaluation run (`EvalResults` → `ResultSet`)

This is the primary use case — turning the output of luml's own `evaluate()` into a portable, spec-conformant results file:

```python
from datetime import datetime, timezone
from luml_openeval_adapter import to_openeval
from openeval.validate import validate_result_set

# eval_results is whatever luml.experiments.evaluation.evaluate.evaluate() returned
result_set = to_openeval(
    eval_results,
    started_at=datetime.now(timezone.utc).isoformat(),  # required -- see why below
)

assert validate_result_set(result_set).valid

import json
with open("my_results.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

`started_at` has no default and must be supplied by the caller: none of `EvalResults`/`EvalResult`/`EvalItem` carries a wall-clock timestamp anywhere (`EvalResult.trace_id` is an OpenTelemetry trace id, not a time), and EvalPort's `ResultSet.started_at` is required. Rather than fabricate one, this adapter asks you for the real value — capture it yourself right before calling `evaluate()`.

Every scorer name found in `EvalResult.scores` becomes its own `GraderResult`. luml's own `evaluate()` populates `scores` with several distinct shapes depending on what happened, and this adapter handles every one of them rather than only the happy path:

* A plain numeric or boolean score (`{"relevancy": 0.82}`) → a `GraderResult` with that score, clamped into EvalPort's required `[0, 1]` range if needed (the unclamped original is preserved under `grader_result.metadata.luml_raw_score`), and `passed = score >= threshold` (default `0.5` — luml itself has no pass/fail concept, only numbers, so pick your own threshold with `to_openeval(..., threshold=0.7)`).
* An LLM-judge scorer's reasoning companion (`{"correctness": 0.9, "correctness_reasoning": "..."}` — the exact shape luml's `LLMJudgeScorer`/`SupervisedLLMJudgeScorer.parse_judgment()` produce) → one `GraderResult` (`type: "luml_llm_judge"`) with the reasoning folded into `reason`, not a second, unscoreable grader.
* A whole-item failure (`{"error": "..."}`, luml's own `evaluate.py` convention when inference or scoring raised) → a `Result.error` object with `grader_results: []`, never a graded `0.0` — "never ran" and "ran and scored zero" are different facts.
* A single scorer failing inside an otherwise-successful item (`{"__error__my_scorer": "..."}`, luml's own convention for that case) → its own `GraderResult` with `score: null` and the failure message in `reason`, alongside the item's other real scores.

`EvalResults.aggregated_scores` (luml's own per-scorer mean/min/max/count, from its `_aggregate_scores()`) is preserved verbatim under `result_set["metadata"]["luml"]["aggregated_scores"]` — nothing luml already computed is thrown away.

### Reversing a `ResultSet` back to luml's shape

```python
from luml_openeval_adapter import from_openeval

eval_results_kwargs = from_openeval(result_set)
# {"results": [...], "aggregated_scores": {...}, "dataset_id": "..."}
```

This is a **partial** reconstruction, not a lossless round trip: a `ResultSet` only links back to the original `EvalItem` via `test_case_id`, so each reconstructed item carries just that `id` (empty `inputs`/`expected_output`/`metadata`) unless you also have the original dataset and merge it in yourself. See the function's docstring for the exact per-field recovery rules.

### Exporting an unscored dataset (`EvalItem` list → `EvalSuite`)

```python
from luml_openeval_adapter import to_openeval_suite
from openeval.validate import validate_suite

# items is a list of luml EvalItem instances (or dicts with the same shape)
suite = to_openeval_suite(items, dataset_id="my_dataset")
assert validate_suite(suite).valid
```

An `EvalItem` alone carries no reference to which `Scorer`(s) it will be run through, so `to_openeval_suite()` defaults each test case's grader to a `"custom"` placeholder (`params.handler: "luml:scorer"`) rather than guessing a specific one — all five of luml's builtin scorers (completeness, correctness, prompt_alignment, relevancy, summarization) are in fact LLM-judge scorers, so pass `grader_type="llm_judge", grader_params={"model": ..., "prompt": ...}` once you know which scorer(s) actually apply to your dataset.

```python
from luml_openeval_adapter import from_openeval_suite

items_kwargs = from_openeval_suite(suite)
# [{"id": ..., "inputs": {...}, "expected_output": ..., "metadata": {...}}, ...]
```

Every function in this package returns plain dicts (constructor kwargs), never a real luml dataclass instance — build the real objects yourself, e.g. `EvalItem(**item_kwargs)`, once you have `luml_sdk` importable in your own environment.

## Credit

Built in direct response to the proposal and closure on [luml-ai/luml#632](https://github.com/luml-ai/luml/issues/632). No luml maintainer or contributor reviewed or shaped this specific mapping beyond that closing comment — it was designed and built independently against luml's public `EvalItem`/`EvalResult`/`EvalResults` shapes, the same way the rest of this repo's standalone adapters are.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
