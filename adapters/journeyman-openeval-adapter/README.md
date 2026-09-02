# journeyman-openeval-adapter

Converts between [codechu/journeyman](https://github.com/codechu/journeyman)'s run records
(`cells/<id>.json`, `report.json`) and [EvalPort](https://github.com/adhabnr-ux/evalport), the
open interchange format for portable LLM evaluation test cases, graders, suites, and results.

```bash
pip install journeyman-openeval-adapter              # adapter only
pip install "journeyman-openeval-adapter[journeyman]" # + the real journeyman-bench package
```

```python
import json
from journeyman.record import RunDir
from journeyman_openeval_adapter import cells_to_testcases, cells_to_result_set

run_dir = "runs/2026-08-31_120000"          # a finished `journeyman run` directory
cells = list(RunDir.attach(run_dir).read_cells())
report = json.load(open(f"{run_dir}/report.json"))

suite = cells_to_testcases(cells, suite_id="my_run")           # the tasks, as TestCases
result_set = cells_to_result_set(cells, report,                # the graded outcome
                                  suite_id="my_run", run_id="2026-08-31_120000")
```

## Why this exists as a standalone package

Opened after reading `record.py`, `judge.py`, `scene.py`, `report.py` and
`journeyman/schema/report.schema.json` in journeyman's own source (not the docs), and refined
across two rounds of correction by journeyman's maintainer on
[codechu/journeyman#1](https://github.com/codechu/journeyman/issues/1) before this code was
written — the same "standalone package, zero footprint on the target repo" shape used by the
DeepEval, AutoGen, CrewAI, and Giskard adapters in this ecosystem.

## The mapping, corrected twice before it was code

Journeyman scores two *kinds* of axis, and the discriminator is which key an axis's data
arrives under on a cell record — **never** a field value:

| Journeyman | EvalPort | Discriminator |
|---|---|---|
| `cell["verdicts"][axis]` — a judge answered a rubric question, verdict matched against a positive label | `GraderResult(type="custom", params.kind="judged")` | key is `verdicts` |
| `cell["event_axes"][axis]` — a `[0,1]` ratio computed by replaying events, no judge involved | `GraderResult(type="custom", params.kind="counted", deterministic=True)` | key is `event_axes` |

An earlier draft of this mapping keyed off `RubricItem.positive is None` — wrong on two counts,
both caught by the maintainer, not by this adapter's own testing: `positive` is a required `str`
that is never `None` (`journeyman/scene.py`), and counted axes have no rubric entry at all, so
that branch never even reached them. Mapping a counted axis as judge-backed would dress up a
deterministic count as a model's opinion — exactly the flattening EvalPort exists to prevent.

| Journeyman (`cells/<id>.json` + `report.json`) | EvalPort |
|---|---|
| one cell (`cell_id, scene, seed, messages, final_text, budget, events, event_axes, verdicts, calls, tokens_in, tokens_out, seconds, invalid, invalid_reason`) | `TestCase` (id, input) + its `Result` |
| `report.json`'s `{schema_version, seal, judge, self_judged, nonstandard, axes, cost, invalid_cells}` | `ResultSet(runner, summary, metadata)` — `seal`/`judge`/`self_judged` are **mandatory** under `metadata["journeyman"]` |

## Why judged axes are NOT `type="llm_judge"`

The obvious first idea. Tested for real against the installed `evalport-sdk`'s
`openeval.validate.validate_grader()`, it fails: EvalPort's `llm_judge` grader requires
`params.model` and `params.prompt` containing `{output}`, `{input}`, or `{expected}`. Journeyman's
judge protocol has neither shape — one judge identity is stamped once per *run*
(`report.json`'s `judge` field), not per grader, and the real prompt template
(`judge.py`'s `JUDGE_PREAMBLE`) substitutes `{labels}`, `{question}`, `{evidence}`, `{record}`,
none of which EvalPort's schema recognizes. Rather than fabricate a prompt journeyman never
actually sends, judged axes use `type="custom"` with `params.handler = "journeyman:rubric_judge"`
and the real rubric fields preserved in `params` when a caller supplies a `rubric_index` — the
same honest "custom, not force-fit" choice the DeepEval adapter makes for its own
framework-specific metrics.

## The condition this adapter was built under

From [the maintainer's reply](https://github.com/codechu/journeyman/issues/1#issuecomment-5490023362):

> whatever a listing shows must carry the conditions it was true under. `seal`, judge identity
> and `self_judged` travel with the numbers or the numbers do not travel. A self-judged run is
> NOT COMPARABLE in our report and should read that way in yours.

`cells_to_result_set()` enforces this structurally, not by convention: it **requires** a `report`
dict carrying `seal`, `judge`, and `self_judged` and raises `ValueError` if any is missing, then
copies all three into `ResultSet.metadata["journeyman"]` verbatim alongside a derived
`comparability` field (`"NOT_COMPARABLE"` when `self_judged` or `nonstandard` is set). There is no
code path in this adapter that produces a `ResultSet` whose numbers have shed the conditions they
were true under. `judge` is copied through as an opaque identity string and never parsed —
since journeyman 0.3.0 it is already written through `public_label()`, which folds a private
host address or filesystem path before the label reaches `report.json` at all; this adapter
respects that and never tries to extract a machine address from it.

## `schema_version`: pin and stop, never guess

Per journeyman's own [`docs/versioning.md`](https://github.com/codechu/journeyman/blob/main/docs/versioning.md#what-you-can-lean-on-today-pre-10):
> Pin on `schema_version`, not on the package version... On a value your code does not know, stop
> rather than guess.

`cells_to_result_set()` raises `ValueError` when `report["schema_version"]` isn't `1` (the only
shape this adapter has been verified against) unless called with `strict_schema=False`.

## What round-trips, and the round-trip test that proves it

The maintainer's own suggested check: `journeyman report <run_dir>` re-renders `report.json` from
the sealed cells with nothing re-run; if this adapter's `ResultSet` reproduces those per-axis
scores exactly, the mapping is lossless where it counts. `tests/test_adapter.py`'s
`test_strictest_round_trip_matches_journeymans_own_rerender` runs exactly that, for real: it
drives journeyman's real pipeline to produce a run, re-renders it via the real
`journeyman report` CLI, and asserts this adapter's recomputed per-axis mean scores match
`report.json`'s own numbers to the same rounding journeyman itself uses.

Does **not** round-trip: a not-applicable verdict (`verdict="na"`, `na_means="not-applicable"`)
keeps its `GraderResult` (so nothing is silently dropped) but carries `score=None` and
`metadata.excluded_from_axis_score=True`, matching journeyman's own exclusion of that cell from
the axis's mean — recomputing the axis score from this adapter's output requires honoring that
flag, not just averaging every non-null score. An invalid cell (`cell["invalid"]`) produces a
`Result` with no `grader_results` and `error.type="invalid_cell"`, matching journeyman's own
exclusion of invalid cells from every axis score — never a silent pass.

## Testing

19 tests in `tests/test_adapter.py`, all passing against the real, installed
`journeyman-bench==0.4.0` package (its actual `driver.run_grid`, `judge.judge_cell`,
`record.RunDir`, `report.render`, `scene.REGISTRY` — the same functions `journeyman run` and
`journeyman selftest` call, not reinvented stand-ins) and the real
`openeval.validate.validate_suite()` / `validate_result_set()` from `evalport-sdk` on PyPI.
Covers: the real cells → EvalPort round trip, the maintainer's own suggested strictest
round-trip check, the `seal`/`judge`/`self_judged` mandatory-metadata condition, the
`schema_version` stop-not-guess rule, the opaque (never-parsed) `judge` field, `not-applicable`
vs. `failure` `na_means`, an unparsed judge verdict, invalid cells, and the `final_text` →
closing-tool-call fallback.

```bash
pip install -e ".[test]"
pip install journeyman-bench==0.4.0
pip install evalport-sdk   # or: pip install -e /path/to/evalport/sdk/python
pytest tests/
```

## Status

Built and tested against journeyman 0.4.0 (`report.json` `schema_version` 1) — see
[codechu/journeyman#1](https://github.com/codechu/journeyman/issues/1) for the full mapping
discussion this package implements.
