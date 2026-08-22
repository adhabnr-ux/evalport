# EvalPort Conformance Test Suite

Resolves [Discussion #9](https://github.com/adhabnr-ux/evalport/discussions/9) ("Formal conformance test suite for runners"), tracked in `spec/SPEC.md`'s Open Design Questions table.

## What this is

Before this existed, "EvalPort-compliant" was only enforced by the JSON Schema files in `spec/schemas/` and the two reference SDKs' hand-rolled validators (`sdk/python/openeval/validate.py`, `sdk/typescript/src/validate.ts`) agreeing with *each other* — see `sdk/python/tests/test_schema_consistency.py`. There was no independent, portable fixture set a third-party runner in another language (Rust, Go, a browser-only build) could test its own implementation against without cloning this repo's Python or TypeScript code.

`fixtures/*.json` fills that gap. Each fixture is a self-contained JSON file:

```json
{
  "description": "human-readable explanation of what this fixture exercises and why",
  "type": "testcase | grader | suite | resultset",
  "expect": {
    "valid": true,
    "error_paths": ["$.results[0].grader_results[0].score"]
  },
  "document": { "...": "the actual document to validate" }
}
```

- `type` says which of the four EvalPort document types `document` is.
- `expect.valid` is whether a conforming validator should accept it.
- `expect.error_paths` (only present on invalid fixtures) lists JSON-Pointer-style paths that a validator's error output is expected to include, so an implementation with structured error reporting can check it flagged the *right* problem, not just *a* problem. This is advisory, not binding — a conformant validator only has to agree on `valid`/`invalid`; matching the exact error path is a nice-to-have this repo's own validators happen to support.

A conformance implementation in any language: load every file in `fixtures/`, run your own validator on `document`, and assert your answer matches `expect.valid`. No dependency on this repo's code required.

## Running the reference check

`run.py` is this repo's own self-check — it runs every fixture through the real `openeval.validate` functions (the same hand-rolled Python validator used everywhere else in this repo) and confirms each fixture's own `expect.valid` is correct. It's wired into CI so a fixture can never silently drift from what the reference implementation actually accepts:

```bash
python3 spec/conformance/run.py
```

Every fixture here has also been independently checked against the raw JSON Schema files in `spec/schemas/` (via the same `Draft202012Validator` machinery `test_schema_consistency.py` uses) — not just the hand-rolled validator — so `expect.valid` reflects genuine agreement between both validation paths this project maintains, not just one of them.

## What's covered so far

| Fixture | Exercises |
|---|---|
| `null_score_not_scored_failure.json` | Validation Rule 6: an unparseable `llm_judge` verdict is `score: null, passed: false`, distinct from a scored failure. |
| `categorical_grader_invalid_category.json` | The same Rule 6 distinction reached from a categorical (non-binary) grader's "couldn't judge" category. |
| `score_out_of_range_rejected.json` | Validation Rule 5: an unclamped native score outside `[0.0, 1.0]` is rejected. |
| `boolean_score_rejected.json` | `score` must be `number \| null`, never boolean — a cross-language gotcha (Python's `bool` is an `int` subclass). |
| `partial_resultset_resumable_run.json` | The resumable-run convention from Discussion #10: per-result `completed_at` plus `metadata.openeval.partial`. |
| `judge_hardening_self_report.json` | The `openeval.judge_hardening` self-report convention from Discussion #11. |
| `custom_grader_missing_handler_rejected.json` | A `custom` (or any non-standard) grader type without `params.handler` is rejected. |
| `non_standard_grader_type_with_handler_valid.json` | Grader `type` is open, not a closed enum, as long as `params.handler` is present. |

This set is deliberately not exhaustive — it's the fixtures that came directly out of building 30 real framework adapters and encountering these exact edge cases in practice (see the `description` field on each fixture for which adapter surfaced it), plus the two open-RFC conventions (#10, #11) it made sense to ship fixtures for at the same time their spec text landed. Contributions of new fixtures — especially ones derived from a *real* edge case you hit building or consuming an EvalPort document, not a hypothetical one — are welcome via the same RFC process as any other spec change (see `spec/SPEC.md`'s Governance section); a new fixture that isn't also a spec/behavior change doesn't need the full two-week comment period, just a PR.

## What this doesn't cover (yet)

This suite currently only exercises `spec/schemas/*.json`-level structural validity — whether a document is a well-formed EvalPort document. It does not (yet) have fixtures for the CLI's runtime behavior (`openeval run`'s cost estimation, retry logic, grader execution) or cross-document referential rules beyond what `validate_suite()` already checks (dangling grader references, duplicate IDs). Those would be reasonable extensions of this suite if someone wants to take them on — flagged here rather than silently treated as "done" by this README's existence.
