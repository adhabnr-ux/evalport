# Staged: `trulens-openeval` for truera/trulens

**This directory is not part of the EvalPort package.** It's a complete, tested `to_openeval()`/`from_openeval()` module built for [truera/trulens#2680](https://github.com/truera/trulens/issues/2680), staged here only because opening a PR directly against `truera/trulens` needs a fork of that repo, which isn't set up yet. This is a temporary home so the code is reviewable now rather than sitting only in a chat transcript.

## Where this belongs once a fork exists

- `src/openeval/pyproject.toml` → `truera/trulens/src/openeval/pyproject.toml`
- `src/openeval/README.md` → `truera/trulens/src/openeval/README.md`
- `src/openeval/trulens/openeval/__init__.py` → `truera/trulens/src/openeval/trulens/openeval/__init__.py`
- `src/openeval/trulens/openeval/py.typed` → `truera/trulens/src/openeval/trulens/openeval/py.typed`
- `tests/unit/test_openeval.py` → `truera/trulens/tests/unit/test_openeval.py`

This mirrors the exact layout of `truera/trulens/src/hotspots` (a real existing package in that monorepo): a poetry-managed `trulens-openeval` package under `src/openeval/`, importable as `trulens.openeval`, with its tests living centrally under the repo's own `tests/unit/`, matching `truera/trulens`'s own convention (confirmed by reading `tests/unit/test_hotspots.py` in that repo).

## Status

All 17 tests pass locally against the **real** `trulens-core` package (`pip install trulens-core`) and the **real** `evalport-sdk` validator (`openeval.validate.validate_result_set` / `validate_suite`) — not mocks. See the test file for what's covered: basic conversion, TruLens's `_calls` companion-column exclusion, score clamping to EvalPort's required `[0, 1]` range, custom pass thresholds, latency-unit handling, and a full suite → TruLens input_df → simulated run → ResultSet round trip, each validated against the real spec.

## Why it converts DataFrames, not `Run`/`RunConfig` objects directly

`trulens.core.run.Run` requires a live `RunDaoBase`, `TruSession`, and app instance just to construct — it's inherently tied to a running TruLens session with a backing database, not a portable, serializable object. `Run.get_records()` / `Run.get_record_details()` (and `Run.start()`) are the actual portable surface, confirmed by reading `trulens/core/run.py` directly, so this module converts at that boundary — the same reason `to_openeval()` converts at Ragas's `to_pandas()` and Opik's plain-dict boundaries rather than requiring their live client objects.
