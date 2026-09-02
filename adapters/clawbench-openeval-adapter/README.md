# clawbench-openeval-adapter

Convert [ClawBench](https://github.com/TIGER-AI-Lab/ClawBench)'s run results (`run-meta.json`, its per-run judge verdict, and `rescore-summary.json`) to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM/agent evaluation results.

**Optional and additive.** This package only reads files ClawBench already writes after a run finishes. It does not change `make_run_meta()`, the two-stage scoring pipeline, `clawbench-rescore`, or any other part of the runner/eval code, and pulling it in adds no dependency for anyone who isn't using it. See [TIGER-AI-Lab/ClawBench#322](https://github.com/TIGER-AI-Lab/ClawBench/issues/322) for the discussion that scoped it this way — this is the *results* side, complementing the existing *test-case* import adapters (`clawbench-harbor-adapt`, `clawbench-edgebench-adapt`) that run the other direction.

> **This package is superseded by [TIGER-AI-Lab/ClawBench#336](https://github.com/TIGER-AI-Lab/ClawBench/pull/336).** It shipped here first as an interim home while this session had no way to fork/PR directly against `TIGER-AI-Lab/ClawBench`; that's now resolved, and #336 ports the same tested logic into `scripts/export_openeval.py` — the shape the maintainer actually asked for on #322 — rather than a standalone package. Once #336 merges, use that instead; this package is kept here for reference and for anyone who prefers installing it as a package pending that merge.

## What it maps

| ClawBench | EvalPort |
|---|---|
| `run-meta.json`'s `test_case` (or `task_id`) | `Result.test_case_id` |
| Stage 1 — `intercepted` | `GraderResult` `gr_interception` |
| Stage 2 — per-run judge verdict (`match`/`reason`, from `judge_llm.json`/`judge.json`, or a `rescore-summary.json` `tasks[]` row's `match_<rubric>`/`reason_<rubric>`) | `GraderResult` `gr_judge_match` |
| `intercepted AND judge_match is True` (the `final_pass` rule in `docs/scoring.md`) | `Result.passed` |
| `instruction`, `model`, `harness`, `result_category`, `failure_category`, `adjusted_eligible` | `Result.metadata` |
| a batch's `rescore-summary.json` | `ResultSet` |

ClawBench scores an intercepted **HTTP request**, not a text completion, so `Result.actual_output` (meant for LLM text output) is left unset — there's no honest value to put there. The instruction that was scored against is carried in `Result.metadata.instruction` instead.

## Grounded in the real, current source — not just `docs/scoring.md`

`docs/scoring.md` describes an idealized per-run shape (`judge_match`, `final_pass`, `result_category` all merged into one record). The actual current code is more spread out:

- `run-meta.json` (`make_run_meta()` in `src/clawbench/runner/run_support/metadata.py`) carries `instruction`/`model`/`harness`/`intercepted`/`result_category`/`failure_category`/`adjusted_eligible` — but **not** `judge_match` or `final_pass`.
- The judge verdict (`match`, `reason`) lives in its own per-run file — `judge_llm.json` for the default "lenient" rubric, `judge.json` for "strict" (`JUDGE_FILE` in `src/clawbench/eval/rescore.py`).
- `rescore-summary.json` (`aggregate_batch()`, same file) rolls a batch into `n_total`/`n_intercepted` plus a `tasks[]` list shaped `{"task_id", "test_case", "intercepted", "match_<rubric>", "reason_<rubric>"}` for whichever rubric(s) ran. `tasks[]` rows do **not** carry `instruction`/`model`/`harness` — only each run's own `run-meta.json` does.

This adapter's `to_openeval()` merges `rescore-summary.json` with an optional `run_metas` map (test_case → parsed `run-meta.json`) to recover that context; without it, results are still produced and spec-valid, just without the enrichment. Nothing is fabricated to paper over the gap between the two files.

## Install

```bash
pip install "clawbench-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/clawbench-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support.

## Usage

```python
import json
from pathlib import Path
from clawbench_openeval_adapter import run_to_result, to_openeval

# One run -> one Result.
run_meta = json.loads(Path("batch-dir/task-dir/run-meta.json").read_text())
judge = json.loads(Path("batch-dir/task-dir/judge_llm.json").read_text())
result = run_to_result(run_meta, judge, rubric="lenient")

# A whole batch's rescore-summary.json -> one ResultSet, optionally
# enriched with each run's own run-meta.json.
rescore_summary = json.loads(Path("batch-dir/rescore-summary.json").read_text())
run_metas = {
    m["test_case"]: m
    for p in Path("batch-dir").rglob("run-meta.json")
    for m in [json.loads(p.read_text())]
}
result_set = to_openeval(
    rescore_summary,
    run_metas,
    run_id="batch-20260830-140000",
    started_at="2026-08-30T14:00:00Z",   # ClawBench doesn't record this itself — supply your own
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

Path("resultset.json").write_text(json.dumps(result_set, indent=2))

# ...and the other direction: an EvalPort ResultSet -> ClawBench-shaped rows (lossy).
from clawbench_openeval_adapter import from_openeval
rows = from_openeval(result_set)
```

`run_id` and `started_at` are required, non-defaulted keyword arguments to `to_openeval()`: ClawBench's `rescore-summary.json` records neither a run id nor a start timestamp for the batch, so there's nothing honest to default them to.

### The reverse direction is intentionally lossy

`from_openeval()` recovers `test_case`, `intercepted`, `match`/`reason`, and whichever of `result_category`/`failure_category`/`adjusted_eligible`/`model`/`harness`/`instruction` this adapter's own `to_openeval()` put under `Result.metadata` on export. A `ResultSet` this adapter didn't produce carries through only what it finds under `metadata`, verbatim — it does not invent ClawBench-specific fields that aren't there.

## Status

Discussed and scoped in [TIGER-AI-Lab/ClawBench#322](https://github.com/TIGER-AI-Lab/ClawBench/issues/322). Maintainer Perry2004 confirmed the results-vs-test-case distinction and said a PR would be welcome, suggesting it land as a simple script rather than a new package under ClawBench's own `adapters/`. This package was built and tested first, against ClawBench's real current source (not just `docs/scoring.md`), while this session had no way to fork/PR directly against `TIGER-AI-Lab/ClawBench` yet. That's now resolved: [TIGER-AI-Lab/ClawBench#336](https://github.com/TIGER-AI-Lab/ClawBench/pull/336) ports this same logic into `scripts/export_openeval.py`, exactly the shape Perry2004 asked for — that's the PR to watch/review; this package remains here as a reference and as an installable fallback until it merges.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
