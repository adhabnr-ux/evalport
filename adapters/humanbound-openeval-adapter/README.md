# humanbound-openeval-adapter

Convert [humanbound](https://github.com/humanbound/humanbound)'s adversarial
LLM-testing output — `ExperimentResults` (`stats`, `insights`, `posture`,
`exec_t`) and `LogEntry` (per-conversation verdicts) — to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

## Why a standalone package

Built at the humanbound maintainers' explicit invitation: see
[humanbound#131](https://github.com/humanbound/humanbound/issues/131#issuecomment-5885427531)
("We'd prefer the standalone package in the EvalPort repo, as you
suggested... Once it's published, share the link here and we'll be happy to
review it for a mention in our docs.").

Pinned against humanbound 2.12.0's public schema surface
(`humanbound.schemas` / `humanbound_cli.engine.schemas`) and
`humanbound.runner.LocalRunner`'s on-disk result layout. `humanbound` itself
stays an optional extra — every conversion function accepts plain dicts with
the same field names, so nothing here requires the package at import time.

## Install

```
pip install "humanbound-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/humanbound-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support. Add the `[humanbound]` extra (or just
`pip install humanbound`) to construct real `LogEntry`/`ExperimentResults`
instances, or to use `read_local_results()` against real local run output.

## Usage

### From a local `hb test --wait` run

```python
from humanbound_openeval_adapter import local_results_to_openeval
from openeval.validate import validate_result_set

result_set = local_results_to_openeval(
    ".humanbound/results/exp-20260929-153000-abcd1234",
    suite_id="my-agent-adversarial-suite",
    run_id="run-2026-09-29",
)
assert validate_result_set(result_set).valid
```

### From `LocalRunner.get_result()` / `.get_logs()` directly

```python
from humanbound.runner import LocalRunner
from humanbound_openeval_adapter import to_openeval

runner = LocalRunner()
experiment_id = runner.start(config)
# ... wait for completion ...
result = runner.get_result(experiment_id)
logs = runner.get_logs(experiment_id, size=10_000).data

result_set = to_openeval(
    result,          # or {"stats": result.stats, "insights": result.insights, ...}
    logs,
    suite_id="my-agent-adversarial-suite",
    run_id=experiment_id,
    experiment_id=experiment_id,
)
```

### What each field becomes

Each `LogEntry` (one per adversarial conversation) becomes one EvalPort
`Result`: `thread_id` &rarr; `test_case_id`, the rendered conversation
transcript &rarr; `actual_output` (the full turn-by-turn conversation is also
preserved verbatim under `metadata.humanbound.conversation`), `result`
(`pass`/`fail`/`error`) &rarr; `passed` (plus `Result.error` for `error`),
and `exec_t` (seconds) &rarr; `duration_ms`. `gen_category`, `fail_category`,
`severity`, `confidence`, and `explanation` become one `GraderResult`
(`grader_id="humanbound_verdict"`, `type="custom"`).

humanbound has no single "how good was this" number on a `LogEntry` —
`severity` grades how bad a *failure* is (0-100) and `confidence` grades the
judge's certainty in its *own verdict* (1-100), which are different axes.
This adapter derives `GraderResult.score = 1 - severity/100` (a clean pass,
with `severity` defaulting to `0`, scores `1.0`) and documents in the source
that this is **this adapter's own derived score**, not a humanbound metric.
`result == "error"` leaves `score` unset (`None`), since severity isn't
meaningful for a conversation the judge never rated.

`ExperimentResults.stats` / `.posture` / `.insights` / `.exec_t` — the
experiment-level aggregate — have no per-`Result` home in EvalPort's schema,
so they're carried through verbatim under `ResultSet.metadata.humanbound`.
`ResultSet.summary` itself is computed fresh from the `Result`s this adapter
actually emits (not copied from `Stats`), so it can't drift from what's in
`results`.

### Round-tripping back

```python
from humanbound_openeval_adapter import from_openeval

back = from_openeval(result_set)
back["logs"]                # list of LogEntry-shaped dicts; LogEntry(**log) works
back["experiment_results"]  # {"stats": ..., "insights": ..., "posture": ..., "exec_t": ...}
```

Fields this adapter wrote round-trip exactly. For a `ResultSet` from another
producer (no `metadata.humanbound`), `stats` is recomputed from the
reconstructed logs (pass/fail/error/total only); `posture` and `insights`
are left `None`/empty rather than guessed, since this adapter has no
faithful way to derive humanbound's own posture/insight formulas (see
`humanbound_cli/engine/presenter.py`) from `Result` data alone.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 — see LICENSE.
