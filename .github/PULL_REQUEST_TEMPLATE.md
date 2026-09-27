## What this changes

<!-- One or two sentences. Link the issue or Discussion: "Closes #NN" / "Implements Discussion #NN". -->

## How it was tested

<!-- Paste the commands you ran and the tail of their output. -->

## Checklist

**Everyone**
- [ ] Linked the issue or Discussion this addresses.
- [ ] New or changed behavior has tests, and those tests check output against the **real** validators (`openeval.validate.validate_suite()` / `validate_result_set()` in Python, `validateSuite()` / `validateResultSet()` in TypeScript), not a mock or a hand-written expectation.
- [ ] I ran the CI checks that apply to what I touched (from the repo root):
  - Python SDK: `(cd sdk/python && pip install -e ".[test]" && python -m pytest tests/ -v)`
  - TypeScript SDK: `(cd sdk/typescript && npm install && npm test)`
  - CLI: `(cd cli && npm install && npm run typecheck && npm test)`
  - Conformance fixtures: `python3 spec/conformance/run.py`
  - Benchmarks: `python3 benchmarks/_tools/validate_all.py --quiet`
- [ ] I did **not** bump `OPENEVAL_VERSION` or the SDK package versions (`sdk/python/pyproject.toml`, `sdk/typescript/package.json`) unless a maintainer asked. Those are bumped at release time.

**Adapters** (`adapters/<framework>-openeval-adapter/`, skip if not applicable)
- [ ] Follows the layout of `adapters/autogen-openeval-adapter`: `pyproject.toml` depending on `evalport-sdk`, `src/<pkg>/__init__.py`, `tests/`, and a `README.md` with install and usage.
- [ ] `pyproject.toml` declares a **named, pinned extra** for the target framework with a verified minimum version, and `test` self-references it (`test = ["pytest", "<adapter>[<extra>]"]`). For a static dataset with no installable package, `test = ["pytest"]` plus a comment saying why (see `financebench-openeval-adapter`).
- [ ] I ran `pip install -e ".[test]"` in a fresh venv and `pytest tests/ -v` passes against that pinned version.
- [ ] Anything lossy or unsupported is written down in the adapter README, not silently dropped.

**Validators, schemas, or spec** (skip if not applicable)
- [ ] For a spec or schema change: the `[Spec Change]` Discussion is linked and its two-week comment period has run (see [spec/SPEC.md#governance](https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md#governance)). Breaking changes also need spec lead sign-off.
- [ ] `spec/schemas/*.json` and `schema/*.json` are **byte-identical**. The `validate-schemas` CI job diffs `suite`, `testcase`, `grader` and `resultset` and fails if they differ. Check with `for f in suite testcase grader resultset; do diff spec/schemas/$f.json schema/$f.json; done`.
- [ ] `spec/SPEC.md` and the root `SPEC.md` are byte-identical (`cmp spec/SPEC.md SPEC.md`). CI does not check this, so please check it yourself. The same goes for any other root mirror you touch (`CRITIQUE.md`, `ADOPTION.md`).
- [ ] Python (`sdk/python/openeval/validate.py`) and TypeScript (`sdk/typescript/src/validate.ts`) validators change together, rule for rule, with matching tests in both. If the JSON Schema and the hand-rolled validators should agree, there's a case in `test_schema_consistency.py` / `schema-consistency.test.ts`.
- [ ] A conformance fixture is added under `spec/conformance/fixtures/` if the rule can be shown as a single document, plus a row in `spec/conformance/README.md`.
- [ ] A Change Log row is added in `spec/SPEC.md` for any spec change.
