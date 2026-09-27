# Contributing to EvalPort

## Where to start

- **Pick up a scoped task:** [open `good first issue`s](https://github.com/adhabnr-ux/evalport/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22). Each one names the files to change, the tests to add, and the exact commands to run, so you can finish it without waiting on a reply. Larger pieces of work are labelled [`help wanted`](https://github.com/adhabnr-ux/evalport/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22). Say in a comment that you're taking one, so two people don't do the same work.
- **Want an adapter for a framework you use?** File an [Adapter request](https://github.com/adhabnr-ux/evalport/issues/new?template=adapter-request.yml), or read [Issue #6](https://github.com/adhabnr-ux/evalport/issues/6) and [Adding a New Converter](#adding-a-new-converter) below and build it yourself. That's the lowest-friction way in.
- **Have a question?** Ask in [Q&A Discussions](https://github.com/adhabnr-ux/evalport/discussions/categories/q-a).
- **Want to change the spec?** Start a `[Spec Change]` Discussion in [Ideas](https://github.com/adhabnr-ux/evalport/discussions/categories/ideas) (see [Spec Changes](#spec-changes) below). Open RFCs where comments are useful right now: [#47](https://github.com/adhabnr-ux/evalport/discussions/47) (`Result.constraint_violations`), [#49](https://github.com/adhabnr-ux/evalport/discussions/49) (FAILED vs. UNVERIFIED), and [#67](https://github.com/adhabnr-ux/evalport/discussions/67) (rate-based `GraderResult` denominators).
- **Before opening a PR,** run the same checks CI runs for the parts you touched. The [PR template](PULL_REQUEST_TEMPLATE.md) lists them.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/evalport.git`
3. Create a branch: `git checkout -b my-feature`
4. Make your changes
5. Run tests: `cd sdk/typescript && npm test` and `cd sdk/python && python -m pytest`
6. Commit: `git commit -m 'Add my feature'`
7. Push: `git push origin my-feature`
8. Open a PR

## Spec Changes

Changes to `SPEC.md` or JSON Schemas follow an RFC process (also described, in full, inside the spec itself: see [`spec/SPEC.md`'s Governance section](../spec/SPEC.md#governance)):

1. Open a GitHub [Discussion](https://github.com/adhabnr-ux/evalport/discussions) in the **Ideas** category, with the `[Spec Change]` prefix
2. Describe the change, motivation, and impact
3. 2-week comment period
4. If consensus, implement in a PR
5. Spec lead sign-off required for changes that break backward compatibility

Four open examples, if it helps to see the shape of a real one before writing your own: [suite/result signing](https://github.com/adhabnr-ux/evalport/discussions/8), [a formal conformance test suite](https://github.com/adhabnr-ux/evalport/discussions/9), [resuming interrupted runs](https://github.com/adhabnr-ux/evalport/discussions/10), and [whether `llm_judge` injection mitigations should be mandatory](https://github.com/adhabnr-ux/evalport/discussions/11) — all things `spec/CRITIQUE.md` flags as deliberately deferred, now with an actual venue for deciding them. Feel free to weigh in on any of these even if you're not proposing a change of your own.

## Code Style

- TypeScript: strict mode, no `any` without justification
- Python: type hints required, `mypy` clean
- Tests required for new grader types and converters

## Adding a New Grader Type

A new *standard* grader type is a spec change, so it goes through the RFC process in [Spec Changes](#spec-changes) first. A framework-specific type needs none of this: any non-empty `type` string is already valid as long as it sets `params.handler`.

1. Add the type to the Grader Type System table in `spec/SPEC.md` and keep the root `SPEC.md` byte-identical
2. Add it to `STANDARD_GRADER_TYPES` and its params rules in both `sdk/python/openeval/validate.py` and `sdk/typescript/src/validate.ts` (plus the `GraderType` union in `sdk/typescript/src/types.ts`)
3. List it in the `type` description/`examples` in `spec/schemas/grader.json` (the field is an open string, not an enum) and copy the file byte-for-byte to `schema/grader.json`. CI fails if the two differ
4. Add a test case in both SDKs
5. Update `docs/grader-reference/README.md`

## Adding a New Converter

There are two places a converter can live, depending on scope:

**Core converters** (maintained in this repo, for frameworks with an established relationship):
1. Add `sdk/python/openeval/converters_FRAMEWORK.py` and/or a `from_FRAMEWORK.ts` function in `sdk/typescript/src/convert.ts`
2. Include before/after test files
3. Document in `docs/migration-guides/`

**Standalone adapter packages** (the easiest way to contribute — start here):
1. Create a new directory under `adapters/FRAMEWORK-openeval-adapter/`
2. Follow the structure of [`adapters/autogen-openeval-adapter`](adapters/autogen-openeval-adapter) as a reference: a `pyproject.toml` depending on `evalport-sdk`, a `src/FRAMEWORK_openeval_adapter/__init__.py` exposing `to_openeval()` / `from_openeval()`, a `tests/` directory with a round-trip test that validates against `openeval.validate.validate_suite()`, and a `README.md` explaining install/usage
3. **Declare a named, pinned extra for the target framework** — not just in `test`. `pyproject.toml` needs a `[project.optional-dependencies]` entry named after the target package (or a short alias, for multi-package targets) with a real, verified minimum version, and `test` should self-reference it rather than duplicate the version string, so the two can never silently drift apart:
   ```toml
   [project.optional-dependencies]
   yourframework = ["yourframework-package>=X.Y.Z"]
   test = ["pytest", "your-adapter-name[yourframework]"]
   ```
   Pin the real minimum you verified — run `pip install -e ".[test]"` in a fresh venv and `pytest tests/ -v` against it before choosing a number, don't guess. This convention exists because a version-pinned extra is what actually catches a breaking upstream API change automatically (a user on a newer, incompatible release gets a clear dependency-resolution failure instead of a confusing runtime error with no indication it's a version-skew problem) — see the real example, including a genuine breaking-API-change catch on `giskard-openeval-adapter`, in [Discussion #13](https://github.com/adhabnr-ux/evalport/discussions/13).

   **Exception — static-dataset targets with no installable package:** some targets (e.g. [`financebench-openeval-adapter`](adapters/financebench-openeval-adapter), which converts the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark) aren't Python packages at all — just data files (JSONL, CSV, Parquet, ...) in a repo. There's no upstream version to pin an extra against. In that case, skip the named extra, keep `test = ["pytest"]` only, and say so explicitly in a `pyproject.toml` comment (so it reads as a deliberate exception, not a forgotten step) — see `financebench-openeval-adapter/pyproject.toml` for the real example.
4. Open a PR — see [Issue #6, "Adapters wanted"](https://github.com/adhabnr-ux/evalport/issues/6) for a list of frameworks that don't have one yet, several already scoped as `good first issue`

This is the lowest-friction way to contribute: it doesn't touch the core SDK, ships independently on PyPI/npm under its own name, and doesn't require waiting on a review of core repo code.

## License

All contributions are licensed under Apache 2.0.
