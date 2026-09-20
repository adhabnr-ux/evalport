# nasde-openeval-adapter

Convert [nasde-toolkit](https://github.com/NoesisVision/nasde-toolkit)'s
`results-export` output to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

## Why a standalone package?

nasde-toolkit's maintainer explicitly declined an EvalPort dependency inside
nasde's core: nasde ships with **zero optional extras** — everything under
`[project.dependencies]` installs for every user — so adding a package for one
export path was a cost the maintainer didn't want every install to carry. He
pointed instead to `nasde results-export` (still marked experimental) as the
surface an external converter should target, since it "needs nothing from
nasde internals and nothing from Harbor." Full discussion:
[NoesisVision/nasde-toolkit#79](https://github.com/NoesisVision/nasde-toolkit/issues/79).

This package reads *only* the files `results-export` writes to each
`<job>__<trial>/` directory — `metrics.json`, `assessment_summary.json`, and
`assessment_eval_<N>.json` — never nasde's raw `jobs/<job>/<task>__<id>/`
Harbor trial layout, and never any nasde-internal Python object.

## Install

```
pip install "nasde-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/nasde-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support (verified working).

## Usage

```python
from pathlib import Path
from nasde_openeval_adapter import to_openeval

# 1. Export a Harbor job with nasde-toolkit:
#    nasde results-export jobs/my-benchmark --to ./exported

dest = Path("./exported")
trial_dirs = sorted(p for p in dest.iterdir() if p.is_dir())

result_set = to_openeval(trial_dirs)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

import json
Path("results.json").write_text(json.dumps(result_set, indent=2))
```

If `dest` contains trials from more than one nasde `source` (benchmark), call
`to_openeval()` once per source — a single `ResultSet` must represent one
suite, not a mix, and the adapter raises `ValueError` rather than guessing.

## How the mapping works

Each `<job>__<trial>/` directory becomes one EvalPort `Result`:

- `test_case_id` — the trial's `task_name` (from `metrics.json`).
- `grader_results` — one `GraderResult` per dimension in the trial's
  **dominant** `(evaluator_model, dimensions_fingerprint)` cluster (from
  `assessment_summary.json`'s `groups[]`), rescaled from nasde's own
  per-dimension `max_score` scale (see nasde's
  [ADR-008](https://github.com/NoesisVision/nasde-toolkit/blob/main/docs/adr/008-independent-dimension-scales.md))
  into EvalPort's required `[0, 1]` grader-score range. The original
  `mean`/`std`/`min`/`max`/`max_score` are preserved verbatim in each grader
  result's `metadata`. `harbor_reward` (nasde's separate verifier signal) gets
  its own `grader_id="harbor_reward"` entry rather than being folded into the
  rubric dimensions.
- `passed` (per dimension and overall) — nasde does not define a pass
  threshold anywhere in its schema, so this adapter does not invent a silent
  one: pass a `pass_threshold` (default `0.5`) to `to_openeval()` to set your
  own bar. When no judge assessment ran at all (e.g. the agent crashed before
  producing artifacts), the adapter falls back to
  `harbor_reward >= reward_fallback_threshold` (default `1.0`) as the only
  available signal, and always records which of the two bases was used in
  `Result.metadata["nasde"]["passed_basis"]` so a fallback verdict is never
  mistaken for a real judge verdict.
- **Cluster safety.** A trial is judged `N` times and nasde aggregates
  repetitions only within one `(evaluator_model, dimensions_fingerprint)`
  cluster — a different judge model or a changed rubric is a different
  benchmark and nasde never averages across them. This adapter mirrors that
  exactly: only the dominant cluster feeds `grader_results`. Every
  *non*-dominant cluster is preserved (not dropped) under
  `Result.metadata["nasde"]["non_dominant_clusters"]`, purely for archival —
  **never average or otherwise mix a `grader_results` score with anything
  under `non_dominant_clusters`.** If you build any rollup across multiple
  `Result`s, only compare `Result`s whose
  `metadata["nasde"]["dominant_cluster"]` shares the same
  `(evaluator_model, dimensions_fingerprint)` pair.
- **Economics.** `token_usage`, `cost_usd`, `pricing_as_of`, `model_name`, and
  `reasoning_effort` are copied verbatim from `metrics.json` into
  `Result.metadata["nasde"]["economics"]` — nothing is dropped, since these
  are exactly the fields the maintainer said make or break the export's
  usefulness for a quality-vs-cost Pareto analysis.
- `actual_output` — best-effort: the dominant cluster's most recent
  `assessment_eval_<N>.json` free-text `summary`, when one exists.
- `duration_ms` — `metrics.json`'s `duration_sec * 1000`.
- `error` — set from `metrics.json`'s `exception_info` when the agent run
  itself failed (no assessment is possible in that case).

`ResultSet.suite_id` is nasde's `source` field (the benchmark/challenge-set
identifier); `ResultSet.run_id` defaults to the shared `<job>` name parsed
from the `<job>__<trial>` directory names.

## `from_openeval()` is lossy — by necessity

`from_openeval()` reconstructs only the flattened, `metrics.json`-shaped view
of a `Result` this adapter itself produces. It **cannot** reconstruct a real
nasde trial directory: there is no way back to the per-repetition
`assessment_eval_N.json` files, the agent's `trajectory.json`, `changes.patch`,
or raw verifier stdout from an aggregated `Result` alone. Treat it as a
round-trip check of the summary fields, not a general EvalPort → nasde
importer.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 — see LICENSE.
