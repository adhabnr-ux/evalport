# benchflow-openeval-adapter

Export [BenchFlow](https://github.com/benchflow-ai/benchflow) rollout results (`results.jsonl`) to [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets and results.

## Why export-only, and why standalone

Filed as [benchflow#1072](https://github.com/benchflow-ai/benchflow/issues/1072). Maintainer [@ElegantLin](https://github.com/ElegantLin) confirmed BenchFlow's `result.json` / job-level `results.jsonl` are already documented external seams an adapter can build against independently — no BenchFlow core changes or dependency, same playbook as [AutoGen](../autogen-openeval-adapter) and [Opik](../opik-openeval-adapter).

There's no `from_openeval()` and no suite-side conversion. A BenchFlow task is a whole sandboxed RL environment (`task.md` + `environment/Dockerfile` + a `verifier/test.sh`), not an EvalPort `TestCase` (`{id, input, graders[]}` — a prompt plus grading criteria); forcing one into the other would lose everything that actually defines the task. This package only ever goes one direction: BenchFlow rollout results → an EvalPort `ResultSet`.

## Install

```bash
pip install "benchflow-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/benchflow-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support.

## Usage

```python
from benchflow_openeval_adapter import job_results_file_to_openeval

result_set = job_results_file_to_openeval(
    "runs/my_job/results.jsonl",
    suite_id="swe-bench-tasks",
    run_id="my_job",
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

import json
with open("resultset.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

Already have the rows parsed (e.g. read via your own pipeline, or the per-rollout file from `write_rollout_results_jsonl` rather than the aggregated job file)? Use `job_results_to_openeval(rows, ...)` directly — it takes any iterable of `results.jsonl`-shaped dicts:

```python
from benchflow_openeval_adapter import job_results_to_openeval

rows = [json.loads(line) for line in open("results.jsonl")]
result_set = job_results_to_openeval(rows, suite_id="swe-bench-tasks", run_id="my_job")
```

### Repeated trials (stability / flip-rate analysis)

Rows are grouped by `test_case_id` (BenchFlow's own `info.task_id` / `info.task_name`, matching how `write_job_results_jsonl` already dedups them). Multiple rows for the same task get ascending, 1-indexed `attempt` numbers in the order given, following EvalPort's repetition/attempt convention:

```python
result_set = job_results_to_openeval(
    rows,
    suite_id="swe-bench-tasks",
    run_id="stability_run",
    isolation="fresh",  # only if you know each trial ran in a fresh sandbox -- never guessed
)
```

A task with just one row gets no `attempt` field at all (absent means single-attempt).

### A reward scale other than [0, 1]

```python
result_set = job_results_to_openeval(
    rows,
    suite_id="my-suite",
    run_id="my_job",
    reward_range=(-1.0, 1.0),           # this task's verifier is documented to score in [-1, 1]
    metric_ranges={"style_score": (0.0, 5.0)},  # a named rubric metric on a 0-5 scale
)
```

## Field mapping

Every design choice below is grounded in BenchFlow's own source (`benchflow.trajectories.results.build_rollout_results_record`, `benchflow._utils.scoring`), not guessed — see the module docstring in [`src/benchflow_openeval_adapter/__init__.py`](src/benchflow_openeval_adapter/__init__.py) for exact citations. This directly addresses every point @ElegantLin raised on [#1072](https://github.com/benchflow-ai/benchflow/issues/1072#issuecomment-5472887549):

| Maintainer's point | How this adapter handles it |
|---|---|
| "passed only when `reward == 1.0`; a default threshold of 0.5 would turn partial credit into a pass" | `bf_reward`'s `passed` is `raw_reward == exact_pass_reward` (default `1.0`), an exact equality check — never a midpoint threshold. Verified against `benchflow._utils.scoring.classify_result`, `benchflow.eval_lift.RolloutResult.passed`, and `benchflow.review.runner`, which all use this identical convention. `Result.passed` follows the `bf_reward` grader alone (not a strict AND of every grader), so a sub-par rubric metric never demotes an otherwise-passing rollout — matching what BenchFlow itself would report for the same rollout. |
| "`metrics` already includes the top-level `reward`, so the sketch would emit `bf_reward` twice" | One `GraderResult` per `metrics` key (`_metrics_from_rewards` already flattens every reward/rubric score into `metrics`, `"reward"` included) — no separately hand-added `bf_reward` on top. |
| "Unscored and errored rollouts need `score: null`, rather than being converted to a scored `0.0` failure" | A rollout is scored iff its `metrics` dict actually contains a `"reward"` key — which `build_rollout_results_record` only adds when the verifier produced a real `rewards` dict (`RolloutResult.rewards` is `None` when "verification was skipped or failed"). Unscored rollouts get `score: null, passed: false` on every grader (spec Validation Rule 6), plus `metadata.openeval.aggregation_status: "unscored"`, never a fabricated `0.0`. A rollout whose verifier *did* run but whose later export step failed (`stop_condition == "export_error"`) keeps its real, already-computed reward — only agent/verifier failures null the score. |
| "Tasks may declare widened reward ranges such as `[-1, 1]`; clamping those values loses meaningful distinctions" | `reward_range` / `metric_ranges` declare each score's true native range so it's linearly *normalized*, not truncated, into EvalPort's required `[0.0, 1.0]` (spec Validation Rule 5 requires one or the other; normalizing preserves the distinctions a hard clamp destroys). The untouched original is always preserved at `GraderResult.metadata.openeval.raw_score` regardless. |
| "A BenchFlow job may contain multiple trials for the same task. Mapping every trial to the same `test_case_id` needs explicit attempt/isolation semantics" | See "Repeated trials" above — ascending `attempt` per `test_case_id`, `isolation` only ever set when the caller passes it explicitly (never inferred, since a `results.jsonl` row doesn't record whether trials shared sandbox state). |
| "The current sketch places `params` on `GraderResult`, which is not accepted by the current EvalPort ResultSet schema" | Not present — `GraderResult` in the current schema/SDK has no `params` field (only the suite-side `Grader` does); every `GraderResult` here uses only `grader_id`/`type`/`score`/`passed`/`metadata`. |

### Response level (job `results.jsonl` row → EvalPort `Result`)

| BenchFlow field | EvalPort field | Notes |
|---|---|---|
| `info.task_id` or `info.task_name` | `test_case_id` | Same key `write_job_results_jsonl` itself dedups rollouts by. |
| — | `attempt` | Set only when more than one row shares a `test_case_id`; ascending, 1-indexed, in input order. |
| `metrics["reward"]` via `bf_reward.passed` | `passed` | `reward == exact_pass_reward` (default `1.0`), not an AND of every grader — see table above. |
| `completion` (assistant chat messages) | `actual_output` | Text content flattened to a string; omitted if there's nothing text-shaped. |
| `timing.total` (seconds) | `duration_ms` | Omitted if `timing` is absent. |
| `error` (`{error, error_chain_str, error_chain_repr}`) | `error` (`{type: "runner_error", message}`) | The schema's `error.type` is closed to `timeout`/`provider_error`/`runner_error` — BenchFlow's specific category (`agent_error`/`verifier_error`/`export_error`) is preserved verbatim at `metadata.benchflow.error_category` instead. |
| `info.agent`, `info.agent_name`, `info.model`, `info.rollout_name`, `is_truncated`, `stop_condition`, `info.training_ready(_reason)`, `total_tool_calls`, `token_usage`, `info.reward_details` | `metadata.benchflow.*` | Passed through when present; unset keys are dropped rather than emitted as `null`. |

### Per-metric (`metrics` entry → EvalPort `GraderResult`)

Every `metrics` key except the two bookkeeping counters `n_tool_calls`/`n_prompts` becomes one `GraderResult` with `grader_id = f"bf_{key}"`, `type: "custom"` (BenchFlow's reward/rubric names are project-defined, not one of EvalPort's well-known grader types). `score` is `metrics[key]` normalized into `[0, 1]` via `reward_range` (for the `"reward"` key) or `metric_ranges.get(key, (0.0, 1.0))` (everything else); `passed` is an exact-equality check against that same convention (`exact_pass_reward` for `bf_reward`, each metric's own declared range maximum — or `metric_pass_reward` if you override it — for the rest), never an invented 0.5 cutoff. The original, pre-normalization value is always preserved at `metadata.openeval.raw_score`.

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v --cov=benchflow_openeval_adapter
```

`benchflow` requires Python ≥3.12 (this adapter itself only needs ≥3.10 — it has no import-time dependency on the `benchflow` package, only on the documented shape of the rows it writes), so the `test` extra needs a 3.12+ interpreter. Verified 2026-09-02: fresh Python 3.12 venv, `pip install -e ".[test]"` resolved `benchflow==0.7.6.dev0` from source (PyPI's published `benchflow` package is an unrelated, older project under the same name — not this one) and `evalport-sdk` from this repo's `sdk/python`, then **32/32 tests passed, 100% line coverage**. Every `results.jsonl` row exercised is built via BenchFlow's own real `benchflow.trajectories.results.build_rollout_results_record` (not a hand-typed fixture), and every produced `ResultSet` is checked against both the real `openeval.validate.validate_result_set` *and*, separately, the raw `spec/schemas/resultset.json` under `jsonschema.Draft202012Validator` — confirming compliance with the schema's stricter `additionalProperties: false` / closed-enum constraints that the hand-rolled validator doesn't itself check (e.g. `summary`'s fixed property set, `error.type`'s three-value enum).

## Standalone Python only

This isn't a Python module in `benchflow` itself — a project without a Python environment (or one using BenchFlow purely as a CLI producing `results.jsonl`) can still consume the format directly: it's one function over a small, fully-documented JSON row shape.
