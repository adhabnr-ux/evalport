# Contributing to OpenEval

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/openeval.git`
3. Create a branch: `git checkout -b my-feature`
4. Make your changes
5. Run tests: `cd sdk/typescript && npm test` and `cd sdk/python && python -m pytest`
6. Commit: `git commit -m 'Add my feature'`
7. Push: `git push origin my-feature`
8. Open a PR

## Spec Changes

Changes to `SPEC.md` or JSON Schemas follow an RFC process:

1. Open a GitHub Discussion with the `[Spec Change]` prefix
2. Describe the change, motivation, and impact
3. 2-week comment period
4. If consensus, implement in a PR
5. Working group approval for major version changes

## Code Style

- TypeScript: strict mode, no `any` without justification
- Python: type hints required, `mypy` clean
- Tests required for new grader types and converters

## Adding a New Grader Type

1. Add the type to `SPEC.md` grader type table
2. Add validation rules in `sdk/typescript/src/validate.ts` and `sdk/python/openeval/validate.py`
3. Add to the JSON Schema `grader.json` enum
4. Add a test case
5. Update `docs/grader-reference.md`

## Adding a New Converter

1. Create `sdk/converters/from_FRAMEWORK.py` and/or `from_FRAMEWORK.ts`
2. Include before/after test files
3. Document in `docs/migration-guides/`

## License

All contributions are licensed under Apache 2.0.
