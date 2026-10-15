# argus-openeval-adapter

Convert Argus (https://github.com/gy15901580825/Argus) redteam run results
to and from EvalPort (https://github.com/adhabnr-ux/evalport), the open
interchange format for portable LLM evaluation/red-team datasets.

## Why a standalone package?

Argus's `api_service/redteam/reports.py` already ships three renderers for
a probe run (`render_html`, `render_sarif`, `render_junit`), each consuming
the same `run` dict: `{"id", "status", "started_at", "finished_at",
"findings": [Finding, ...]}`. Per the maintainer's confirmation in
[Argus#17](https://github.com/gy15901580825/Argus/issues/17#issuecomment-6185241349),
an EvalPort adapter is a natural fourth renderer next to those three, built
against that same `run` dict contract from the outside — so it ships here,
as an independent package, rather than requiring a code change inside
Argus (which, as of Argus #21, now keeps that renderer in two
byte-identical copies — `api_service/redteam/reports.py` and
`cli/argus_probe/reports.py` — that a test asserts never drift).

Argus itself is a monorepo of separate per-service packages
(`argus-orchestrator`, `argus-probe`, an unnamed `api_service`, ...), none
published to PyPI, so there is no upstream package this adapter could
depend on and pin a version against (see `pyproject.toml`'s comment for
specifics). It is written entirely against the public `run`/`Finding`
dict-or-attribute shape, never importing Argus itself.

## Install

```
pip install "argus-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/argus-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support.

## Usage

```python
from argus_openeval_adapter import to_openeval, from_openeval

# `run` here is the same dict shape reports.render_sarif(run) consumes:
# {"id": ..., "status": ..., "started_at": ..., "finished_at": ..., "findings": [Finding, ...]}
result_set = to_openeval(run)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

import json
with open("argus-run.openeval.json", "w") as f:
    json.dump(result_set, f, indent=2)

run_again = from_openeval(result_set)
```

`run` may be a plain dict (as above) or any object exposing the same
fields by attribute — Argus's real `Runner`/`Finding` classes work
directly, with no import of `argus` required.

By default the ResultSet's `suite_id` is the constant
`"argus-redteam-probes"`, since Argus redteam runs aren't driven by an
EvalPort `EvalSuite` file. Pass your own if you track probe sets under a
more specific id:

```python
result_set = to_openeval(run, suite_id="argus-owasp-llm-top10")
```

## Field mapping

| Argus `Finding` field | EvalPort field |
| --- | --- |
| `id` | `Result.metadata.argus.finding_id` |
| `probe_id` | `Result.test_case_id` |
| `verdict` | `Result.passed` + `GraderResult.passed` (derived — see below); raw string preserved at `Result.metadata.argus.verdict` |
| `confidence` | `GraderResult.score`, clamped to `[0, 1]` — **only for a judged verdict** (see below); otherwise `GraderResult.metadata.categorization_confidence` |
| `severity` | `GraderResult.metadata.severity` |
| `reasoning` | `GraderResult.reason` |
| `attack_prompt` | `Result.metadata.argus.attack_prompt` |
| `target_response` | `Result.actual_output` |
| `target_latency_ms` | `Result.duration_ms` (rounded to the nearest int) |
| `probed_at` | `Result.completed_at` |
| `judge_model` | `GraderResult.metadata.judge_model` (also selects `GraderResult.type = "llm_judge"`) |
| `escalated_model` | `GraderResult.metadata.escalated_model` |
| `atlas_id` / `owasp_id` / `nist_id` / `eu_ai_act_id` (each a `list[str]`) | `GraderResult.metadata.labels.*` |
| `evidence` | `Result.metadata.argus.evidence` |

`run.status` (e.g. `"completed"`) is carried through at
`ResultSet.metadata.argus.status`, since EvalPort's `ResultSet` has no
native `status` field.

### Judged vs. untested-only verdicts

Argus's own `orchestrator/orchestrator/redteam/coverage.py` already splits
`Finding.verdict` into two groups, and this adapter uses the same split
rather than inventing its own:

- **Judged** (`_TESTED_VERDICTS`): `"pass"`, `"fail"`, `"warn"` — the judge
  model actually ran and rendered a verdict (Argus's own test suite shows
  the judge literally emitting `{"verdict": "warn", ...}`). For these,
  `confidence` becomes a real `GraderResult.score`, and `passed` is `true`
  only for `"pass"` (`"warn"` is judged but not a clean pass, so it maps to
  `passed: false` with its real score still attached — distinguishable
  from a hard `"fail"` via the preserved `metadata.argus.verdict`).
- **Untested-only** (`_UNTESTED_ONLY_VERDICTS`): `"skipped"`,
  `"blocked_by_target"`, `"aborted_cost"`, `"error"` — the probe never
  reached a real judgment (a target-class mismatch, the target's own
  gateway blocking the request with an HTTP 4xx, a cost cap, or the
  judge/runner itself erroring). `Finding.confidence` on these findings is
  **not** a pass/fail score — `Runner._skip()` hard-codes `confidence=1.0`
  for every skip, meaning "confident this probe didn't apply," not
  "confident in a judgment." This adapter never surfaces that value as
  `GraderResult.score`; it goes to `GraderResult.metadata.categorization_confidence`
  instead, and the result is emitted as `score: null, passed: false` —
  EvalPort's spec requires a null-scored ("not verified") grader result to
  report `passed: false` (SPEC.md Validation Rules, Rule 6).

A judged verdict with no numeric `confidence` attached still gets a score
(`1.0` for `"pass"`, `0.0` for `"fail"`/`"warn"`), treating the verdict
itself as full confidence rather than discarding the signal.

The original `verdict` string is always preserved under
`metadata.argus.verdict`, so `from_openeval()` round-trips it exactly
instead of collapsing every non-pass outcome to a bare `"fail"`.

## Credit

Built from the real `Runner`/`Finding` shape and the `pass`/`fail`/`warn`
vs. `skipped`/`blocked_by_target`/`aborted_cost`/`error` split read in
`orchestrator/orchestrator/redteam/runner.py` and
`orchestrator/orchestrator/redteam/coverage.py`, the renderer contract in
`api_service/redteam/reports.py`, and the mapping confirmed by the Argus
maintainer in
[Argus#17](https://github.com/gy15901580825/Argus/issues/17).

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 - see LICENSE.
