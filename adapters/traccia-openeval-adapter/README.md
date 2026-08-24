# traccia-openeval-adapter

Convert [traccia](https://github.com/traccia-ai/traccia-py)'s
`traccia.eval.evaluate()` results to
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets and results.

Proposed and tracked at
[traccia-ai/traccia-py#35](https://github.com/traccia-ai/traccia-py/issues/35).

## Why a standalone package?

Same reasoning as this repo's other 36 adapters (see
[adapters/autogen-openeval-adapter](../autogen-openeval-adapter), the
reference implementation, and
[issue #6](https://github.com/adhabnr-ux/evalport/issues/6)): traccia-py
doesn't need to carry any EvalPort-specific code, and this package doesn't
need to wait on a traccia-py release cycle. It's a thin, independently
versioned layer on top of `EvaluateResult`'s public dataclass fields.

## Install

```
pip install "traccia-openeval-adapter[traccia] @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/traccia-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support. The `[traccia]` extra pulls in
`traccia==0.1.28`, the exact version this adapter was built and tested
against.

## Usage

```python
from traccia.eval import evaluate
from traccia_openeval_adapter import results_to_openeval
from openeval.validate import validate_result_set

result = evaluate(
    "my-experiment",
    data=[{"input": {"q": "2+2"}, "expected": "4"}],
    task=lambda row: my_agent(row["input"]),
    scorers=["exact_match"],
    persist=False,
)

result_set = results_to_openeval(result, suite_id="my-experiment")
assert validate_result_set(result_set).valid
```

## Field mapping

| traccia (`EvaluateResult`)                          | EvalPort (`ResultSet`)                          |
|------------------------------------------------------|--------------------------------------------------|
| `rows[i]["item_id"]`                                  | `results[i]["test_case_id"]`                     |
| `rows[i]["panels"][0]["output"]`                      | `results[i]["actual_output"]` (stringified)      |
| `rows[i]["panels"][0]["passed"]`                      | `results[i]["passed"]`                           |
| `rows[i]["panels"][0]["latency_ms"]`                  | `results[i]["duration_ms"]` (rounded)            |
| `rows[i]["panels"][0]["error"]`                       | `results[i]["error"]["message"]`                 |
| `rows[i]["panels"][0]["scores"][j]`                   | `results[i]["grader_results"][j]`                |
| `scores[j]["scorer_id"] or scorer_name or name`       | `grader_results[j]["grader_id"]`                 |
| `scores[j]["type"] or scorer_name or name`            | `grader_results[j]["type"]`                      |
| `scores[j]["score"]` (clamped to `[0, 1]` or `None`)  | `grader_results[j]["score"]`                     |
| `scores[j]["passed"]`                                 | `grader_results[j]["passed"]`                    |
| `scores[j]["reason"]`                                 | `grader_results[j]["reason"]`                    |
| `aggregates`                                          | `metadata["traccia_aggregates"]`                 |

`scorer_id`, `model`, `latency_ms`, `cost_usd`, `usage`, and `config` on a
score dict have no dedicated slot on EvalPort's `GraderResult`, so they're
preserved under `grader_results[j]["metadata"]` rather than dropped.

## Design notes (real wrinkles, not guesses)

Both of these were found by actually running `evaluate(..., persist=False)`
against the real 0.1.28 package — the same call shown in #35's own usage
example — not by reading the proposal sketch alone:

- **`run_id`**: EvalPort requires it; `EvaluateResult.experiment_id` is
  `None` whenever `persist=False` (`evaluate()` only allocates a uuid when
  `persist=True` — see `eval/evaluate.py`). Since `persist=False` is the
  case shown in #35's own example and is the common local/offline path,
  this adapter mints `traccia-local-<uuid12>` when no `experiment_id` (and
  no explicit `run_id=` override) is available, rather than leaving the
  required field empty or silently reusing something unrelated.
- **`started_at`**: EvalPort requires it (ISO 8601); `EvaluateResult`
  doesn't record a wall-clock run-start timestamp anywhere — only
  per-item `latency_ms`. This adapter defaults it to conversion time. Pass
  `started_at=` explicitly if you captured the real start time before
  calling `evaluate()`.
- **Single panel per row**: `evaluate()`'s `aggregates["panel_count"]` is
  hardcoded to `1` — traccia doesn't yet support multi-panel/prompt-
  comparison rows (`panels` is always a one-element list wrapping a single
  "Task" cell). `row_to_result()` raises `ValueError` if it ever sees more
  than one panel, instead of silently mapping only `panels[0]` and losing
  the rest, so a future traccia release that adds this doesn't fail
  quietly.
- **Unscored items**: when `scorers=[]`, `panel["passed"]` is `None` in
  real traccia — distinct from `False`. This adapter treats an unscored
  item as `passed: true` on the EvalPort side (nothing failed it), since
  EvalPort's `Result.passed` is a required bool and coercing `None` to
  `False` would misrepresent "not evaluated" as "evaluated and failed".

## What this does *not* do (yet)

Only the forward direction (`results_to_openeval`) is implemented. traccia
doesn't have a native "load an EvalPort suite as a traccia dataset" entry
point to target for the reverse direction, and `evaluate()`'s builtin
scorers are currently only `exact_match` / `contains` / `json_valid` — see
`eval/builtins.py` — so there's no `llm_judge`/`semantic_similarity`
mapping to design yet either. Worth revisiting as `EvaluateResult` grows.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 — see LICENSE.
