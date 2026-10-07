# frontieror-openeval-adapter

Convert [FrontierOR](https://github.com/Minw913/FrontierOR) evaluation results
and public task instances to [EvalPort](https://github.com/adhabnr-ux/evalport),
the open interchange format for portable evaluation suites and results.
FrontierOR benchmarks LLM-designed algorithms on large-scale optimization
problems from OR papers.

Filed and built following
[Minw913/FrontierOR#5](https://github.com/Minw913/FrontierOR/issues/5) at the
maintainer's request. The maintainer asked for a minimal standalone prototype
built on the latest dataset and evaluation protocol, with an example export and
an explanation of how scoring is handled. This package is that prototype. It
lives in the EvalPort repo and does not change FrontierOR. FrontierOR has not
adopted it.

It was written against FrontierOR commit
[`bc1bdde`](https://github.com/Minw913/FrontierOR/commit/bc1bddebc4bbd97f246ceddf0056296a4ac87fcc)
and the Hugging Face dataset `SmartOR/FrontierOR` at revision `9f5cd54`
(the v1 update of 2026-09-27).

## Install

Not published to PyPI. Install from source:

```
pip install "frontieror-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/frontieror-openeval-adapter"
```

The only dependency is `evalport-sdk`. FrontierOR is not pip-installable, so
the adapter reads FrontierOR's output files with the standard library. If you
also have a FrontierOR checkout, pass `frontieror_root=` (or put it on
`PYTHONPATH`) and the adapter scores with FrontierOR's own `StagedQteScorer`.

## Usage

### One-shot results CSV to a ResultSet

```python
import frontieror_openeval_adapter as fo
from openeval.validate import validate_result_set

data = "frontier-or"  # local copy of the Hugging Face dataset
refs = fo.load_references(f"{data}/metadata/gurobi_references.csv.gz")  # public table
meta = fo.load_paper_meta(f"{data}/metadata/paper_meta_info.json")      # directions

rows = fo.read_results_csv("eval/eval_results.csv")  # one_shot_eval.py output
for model, model_rows in fo.split_by_model(rows).items():
    rs = fo.results_to_openeval(
        model_rows,
        references=refs,
        paper_meta=meta,
        run_id=f"my-run/{model}",
        started_at="2026-09-29T18:24:54Z",
        candidate_time_limits=3600,           # the budget the programs ran under
        frontieror_root="../FrontierOR",      # optional: use FrontierOR's own scorer
    )
    assert validate_result_set(rs).valid
```

### Self-evolution results to a ResultSet

`augment_results_with_staged_qte()` in `test_time_self_evolution/eval_modes.py`
already computes staged_qte. The adapter carries its fields over as they are
and does not re-score:

```python
rows = fo.augmented_results_to_rows(final_results, paper_id="bierwirth2017", model="gpt-5.4")
rs = fo.results_to_openeval(rows, references=refs, paper_meta=meta,
                            run_id="coral/bierwirth2017", started_at="...")
```

### Public tasks to a Suite

```python
items = fo.load_dataset_instances(data, ["bierwirth2017", "walteros2020"])
suite = fo.to_openeval(items, paper_meta=meta, suite_id="frontieror")
fo.from_openeval(suite) == items   # round trip
```

Each (paper, instance) pair becomes one `TestCase` with id
`"<paper_id>:<instance>"`, for example `"walteros2020:large_1"`. The input is
the paper's public `problem_description.txt`. The dataset paths of the
instance JSON, the schemas and `mathematical_formulation.md`, the
optimization direction and the paper metadata go in `metadata.frontieror`.
Instance JSON is referenced by path and not embedded, because files can be
many megabytes. The formulation text is embedded only with
`load_dataset_instances(..., include_formulation=True)`. There is one suite
for the whole benchmark rather than one suite per paper as first sketched in
the issue, because an EvalPort `ResultSet` points at a single `suite_id` and a
model's run covers many papers.

## How scoring is handled

FrontierOR's score is `staged_qte` (contract `staged-qte-v1`,
`trusted_eval_infra/contracts.py`). With `g` the signed gap
(`(c-r)/D` for minimization, `(r-c)/D` for maximization,
`D = abs(r) if abs(r) >= 0.001 else max(abs(c), 0.001)`) and `b = 0.01`:

- Stage 1 (`max(0,g) > b`): `max(0, 1-g)`
- Stage 2 (`max(0,g) <= b`): `(1-g) + max(0, 1 - t/tau)`, which ranges up to
  `2 + beat_amount`
- infeasible or missing: `0.0`

EvalPort's `GraderResult.score` must be in `[0, 1]`. The adapter handles this
with three graders. Each one says what it measures.

| Grader | `score` | `passed` | What it is |
|---|---|---|---|
| `gr_frontieror_quality_only` (`frontieror:quality_only`) | `min(1, max(0, 1-g))`, or null when `g` is undefined | `max(0,g) <= b` | A normalized quality-only score. **Not staged_qte.** |
| `gr_frontieror_staged_qte` (`frontieror:staged_qte`) | always null | always false (Rule 6); the stage reached is `metadata.frontieror.stage_id` | FrontierOR's score, unchanged, in `metadata.frontieror.staged_qte` |
| `gr_frontieror_binary_qte` (`frontieror:beat_gurobi_1`) | 1.0 or 0.0 | same | The paper's binary QTE |

The maintainer made two points in the issue. This is how each one is handled.

**1. "A normalized quality-only score should be clearly distinguished from the
original staged_qte score."** The two are separate graders with separate ids
and handlers. The quality-only grader's description says it is not
staged_qte, and it sets `params.is_staged_qte: false` and
`metadata.frontieror.is_staged_qte: false`. Dropping the speed term changes
what is measured. Two real results in the example export show this.
`walteros2020:large_1` for claude-opus-4.6 matched Gurobi in 0.06 s against
Gurobi's 37.58 s: staged_qte is `1.998403` and quality-only is `1.0`.
`bierwirth2017:tiny` for gpt-5.3-codex also matched but took 4.95 s against
3.75 s: staged_qte is `1.0` and quality-only is `1.0`. Quality-only cannot
tell the two apart, and staged_qte can.

**2. "If g can be negative, max(0, 1-g) can still exceed 1, so an upper clamp
would be needed."** Correct. The quality-only score is `min(1, max(0, 1-g))`,
clamped on both sides. When `g < 0`, the unclamped value is kept as
`metadata.frontieror.clamped_from`, so the clamp is visible. The earlier
sketch in the issue used `max(0, 1-g)` and was wrong. The tests cover
`g = -0.1`, `g = -0.3` and `g = -1e9`.

**Why staged_qte's score slot is null.** staged_qte has no upper bound
(`2 + beat_amount`). Clamping it to 1 would give every stage-2 result above
parity the same score, so a 0.06 s solve and a 37 s solve would look the
same. Dividing by 2 would still exceed 1 when the candidate beats Gurobi, and
it would make up a scale FrontierOR does not define. The adapter therefore
does not put staged_qte in the `[0,1]` slot at all. The raw value and every
debug field FrontierOR produces (`stage_id`, `quality_part`, `speed_part`,
`signed_gap`, `beat_amount`, `matched`, `beat_gurobi`) are stored unchanged,
together with `contract_version` and `scorer_source`. The grader's `reason`
says why the slot is null, so a generic consumer that treats null as "skipped"
can read the explanation. Because the score is null, the grader's `passed` is
always `false`: EvalPort Validation Rule 6 says a null score means "not
verified" and MUST carry `passed: false`, and the reference validators reject
`score: null` with `passed: true` (`NULL_SCORE_PASSED`). Whether a result
reached stage 2 is `metadata.frontieror.stage_id`, and
`summary.by_grader.gr_frontieror_staged_qte` counts 0 passed / 0 failed
(null-scored results are excluded; the per-stage counts are in
`metadata.frontieror.aggregates.stage_counts`).

*On the evalport Discussion #49 alternative-B branch only (declared aggregation, not in the
spec, not on `main`):* a Result with some null-scored and some scored graders has to declare how
its `passed` was derived, so every ResultSet also carries
`metadata.openeval.aggregation: {"strategy": "producer"}`: `passed` is the binary QTE grader's
verdict alone (next paragraph), and the staged_qte null is a by-design "not placed in [0,1]"
value, not a grader that failed to run. That mismatch is one of the branch's own open questions,
since Rule 6 reads null as "not verified". On `main` the key is free-form metadata and nothing
reads it.

**`Result.passed`** is the paper's binary QTE (`beat_gurobi_1` in
`scripts/compute_benchmark_main_metrics.py`). A cell passes when it is
feasible, has gap <= 1%, and its budget-capped wall time is no slower than
Gurobi's within `max(0.01 s, 0.001 * tau)`. The adapter also applies the
paper's proven-zero-optimum table and its tiny-instance gate. The binary
check uses the one-shot CSV `gap` (`compute_gap`, denominator `abs(r)`),
while the other two graders use the contract's `D`-scaled gap, because that
is what each FrontierOR metric uses. The two differ only when `abs(r) < 0.001`.
A result can have quality-only passed and staged_qte at stage 2 but `Result.passed` false.
For example, `bierwirth2017:tiny` matched Gurobi but was slower. The paper
averages binary QTE only over `large_*` cells. Tiny results get the same
per-cell rule, marked `in_paper_beat_gurobi_1_grid: false`.

**Infeasible, failed, not run, missing.** Quality-only `score` is null (not
0.0) with a `reason`, because an infeasible solution's objective is not a
valid objective. In the example run, six liao2020 programs report objectives
91% to 99.5% "better" than Gurobi's, and all six are infeasible. staged_qte is
`0.0` with FrontierOR's own reason. `Result.passed` is false. Timeouts also
set `Result.error.type = "timeout"`. Pass `declared_instances` to get
explicit "missing" results for instances that never ran.

**Aggregation.** The contract takes the arithmetic mean over declared
instances, with missing or failed instances scored 0.0. EvalPort's
`summary.avg_score` is the mean of non-null scores, which here means the
quality-only scores of results where `g` is defined. That number is higher
and is not a FrontierOR metric. `metadata.frontieror.aggregates` therefore
carries the FrontierOR-style numbers: `staged_qte_mean` (0-filled),
`quality_only_mean_null_as_zero`, `quality_only_mean_non_null`,
`stage_counts`, and the binary QTE over `large_*` cells. Each comes with the
rule that says when it equals FrontierOR's own figure.

## Where the staged_qte number comes from

`staged_qte_source="auto"` (the default) picks the first available source:

1. `precomputed`: the row already has `score`/`stage_id`/... from
   `augment_results_with_staged_qte()`. Those values are carried over.
2. `upstream`: FrontierOR's `StagedQteScorer` from a checkout
   (`frontieror_root=` or PYTHONPATH). The adapter refuses to run if the
   checkout's `SCORING_CONTRACT_VERSION` is not `staged-qte-v1`.
3. `builtin`: a line-for-line port of `StagedQteScorer.score_instance`. The
   tests check it against the real scorer on 15 cases whenever a checkout is
   available.

`metadata.frontieror.scorer_source` records which source was used for each
result.

## Field mapping

| FrontierOR | EvalPort |
|---|---|
| `paper_id`, `instance` | `Result.test_case_id = "paper_id:instance"` |
| `model` | `ResultSet.provider.model` (one ResultSet per model, see `split_by_model`) |
| `obj` / `llm_obj`, `time` / `solve_time`, `feasible`, `status`, `fail_reason`, `error` (first 500 chars) | `Result.metadata.frontieror.*` |
| `time` / `solve_time` | also `Result.duration_ms` |
| `error` containing "timed out", or `fail_reason == "timeout"` | `Result.error = {type: "timeout"}` |
| reference `objective_value`, `runtime`, `time_limit`, `status` (public table) | `Result.metadata.frontieror.reference_*` |
| staged_qte `score` + debug fields | `grader_results[gr_frontieror_staged_qte].metadata.frontieror` |
| `problem_description.txt` | `TestCase.input` |
| paths of `mathematical_formulation.md`, the instance JSON and the schemas; `paper_meta_info.json` fields | `TestCase.metadata.frontieror` (formulation text only with `include_formulation=True`) |

**Lossy or not carried:**
- The candidate's solution JSON and convergence log (`log_*.jsonl`) are not
  embedded, and neither is `aocc`.
- `error` is cut to 500 characters, as the one-shot CSV already does.
- The one-shot CSV rounds `time` to 2 decimals and `gap`/`obj` to 6. The
  adapter works from those rounded values.
- `debug_retries`, `correction_retries`, `delta_time` and the `first_*`
  columns are not carried.
- Precomputed (self-evolution) rows do not record why a stage-0 score is 0.
  The adapter says so and does not guess a reason.
- The quality-only score is clamped. See `clamped_from`.

## Visibility policy (public instances only)

FrontierOR's `visibility_contract()` marks `reference_objective`,
`reference_runtime`, `final_per_instance_score` and
`final_instance_membership` as `TRUSTED_ONLY`. The adapter follows it:

- `references` is required and must be the dataset's **public** table,
  `metadata/gurobi_references.{parquet,csv.gz}`. It is the only source of
  reference values, and it is also the allow-list of instances that may be
  exported.
- A result for any instance not in that table (for example a hidden final
  instance from `trusted_eval_infra`) raises `NonPublicInstanceError`. With
  `on_non_public="omit"` the result is dropped, and only the number dropped is
  recorded. The paper id, instance id, score and reference are not written.
- If a row's own `gurobi_obj` disagrees with the public table, the
  conversion fails instead of exporting a score computed against another
  baseline.
- `to_openeval()` suites contain no reference objectives or runtimes.

## Example export

`examples/output/` contains a real export. It is FrontierOR output, not
hand-written data:

- **What ran:** FrontierOR's own `one_shot_eval.py` Quick Start path
  (`--reuse-code all --code-root samples/oneshot_code --exec-mode bare`) on
  2026-09-29. It re-ran the **pre-generated programs shipped in
  `samples/oneshot_code/`**. No LLM was called, and no program was written or
  edited by us. The run used public instances of three papers
  (bierwirth2017, liao2020, walteros2020) from the Hugging Face dataset at
  revision `9f5cd54`.
- **Scope:** 21 (paper, model) sample programs. The 11 that do not import
  gurobipy ran on tiny and large_1. The 10 that do ran on tiny only. The time
  limit was 300 s, which is `one_shot_eval.py`'s default. That is shorter than
  the paper's 3600 s budget for large instances, so this is not a
  paper-protocol result. The run used 2 vCPUs.
- **Excluded:** 3 rows are excluded. They failed with
  `Model too large for size-limited license`, which comes from the
  pip-installed Gurobi license in our environment, not from the programs. They
  stay verbatim in the CSV, and `export_example.py` skips them.
- **Files:** `examples/data/` holds the two raw results CSVs, the task plans,
  `RUN.json` (commands, timestamps, environment, exclusions), and the public
  reference and paper-metadata rows for the three papers. Those rows were
  copied verbatim from the dataset (CC-BY-4.0, SmartOR/FrontierOR).
  `examples/output/` holds one ResultSet per model (7 files, linked with
  `ResultSet.group`) plus `suite.json` (3 papers x the 2 instances that ran, tiny and large_1).
- **What it shows:** stage-2 scores well above 1 (walteros2020, up to
  `1.998403`); a match that is slower than Gurobi (bierwirth2017 tiny,
  staged_qte `1.0`, binary QTE false); infeasible solutions with "negative
  gaps" (liao2020); a 300 s timeout; and large_1 rows skipped by the tiny
  gate.
- **Reproduce:** `python examples/export_example.py` rebuilds every ResultSet
  from the CSVs. The test suite checks that the rebuilt numbers match the
  shipped files. Passing `--frontieror-root` uses FrontierOR's own scorer. The
  shipped files were produced that way, and the numbers are identical either
  way.

Across the 7 models, the adapter's binary-QTE count over large_1 cells
matches `compute_benchmark_main_metrics.compute_metrics()` run on the same
CSVs (5 models 1/1 or 1/2, 2 models 0/1).

## Tests

```
pip install -e ".[test]"
pytest tests/ -v                                   # evalport-sdk only: 23 pass, 21 skip
FRONTIEROR_ROOT=/path/to/FrontierOR pytest tests/  # + parity with FrontierOR: 44 pass
```

The parity tests need Python 3.11+ (`trusted_eval_infra` uses `enum.StrEnum`).
The binary-QTE parity test also needs pandas and PyYAML, because it calls
`compute_metrics()` from FrontierOR's metrics script.
