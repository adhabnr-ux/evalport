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
- `mode` (optional, **PROPOSED** in [Discussion #108](https://github.com/adhabnr-ux/evalport/discussions/108), not yet normative) selects the validation mode. Absent (or `"strict"`) means the default strict validation, which is the mode conformance claims use: a property the schema doesn't define on a closed (`additionalProperties: false`) object is rejected (`UNKNOWN_FIELD`). `"allow_unknown"` means lenient consumption, for a reader of a document from a newer minor version: unknown properties are accepted and every other rule still applies. An implementation without a lenient mode may skip fixtures that declare `"mode": "allow_unknown"`.
- `expect.error_paths` (only present on invalid fixtures) lists JSON-Pointer-style paths that a validator's error output is expected to include, so an implementation with structured error reporting can check it flagged the *right* problem, not just *a* problem. This is advisory, not binding — a conformant validator only has to agree on `valid`/`invalid`; matching the exact error path is a nice-to-have this repo's own validators happen to support.

A conformance implementation in any language: load every file in `fixtures/`, run your own validator on `document`, and assert your answer matches `expect.valid`. No dependency on this repo's code required.

## Running the reference check

`run.py` is this repo's own self-check — it runs every fixture through the real `openeval.validate` functions (the same hand-rolled Python validator used everywhere else in this repo) and confirms each fixture's own `expect.valid` is correct. It's wired into CI so a fixture can never silently drift from what the reference implementation actually accepts:

```bash
python3 spec/conformance/run.py
```

The TypeScript SDK runs the same portable fixtures as part of `cd sdk/typescript && npm test`, so additions are checked automatically by both reference validators.

Every fixture here has also been independently checked against the raw JSON Schema files in `spec/schemas/` (via the same `Draft202012Validator` machinery `test_schema_consistency.py` uses) — not just the hand-rolled validator — so `expect.valid` reflects genuine agreement between both validation paths this project maintains, not just one of them. The documented exceptions are the hand-rolled-only cross-field rules (named in each such fixture's `description`) and the `"mode": "allow_unknown"` fixtures: the raw JSON Schema models strict validation only, so it rejects the two `allow_unknown` documents that are valid in lenient mode.

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
| `multi_attempt_resultset_valid.json` | The repetition/attempt tracking convention from Discussion #22 / issue #20: multiple `Result`s for one `test_case_id` distinguished by ascending `attempt`, plus a single `ResultSet`-level `isolation`. |
| `duplicate_attempt_collision_rejected.json` | Discussion #22 / issue #20's uniqueness rule: a duplicate `(test_case_id, run_id, attempt)` is rejected. |
| `group_membership_valid.json` | **Landed in [PR #54](https://github.com/adhabnr-ux/evalport/pull/54), following [Discussion #45](https://github.com/adhabnr-ux/evalport/discussions/45).** A valid grouped `ResultSet` (`group.group_id`/`role`/`label`/`sequence`), composed with `attempt`/`isolation` from the previous row. |
| `group_missing_group_id_rejected.json` | **Landed (Discussion #45 / PR #54).** `group` present without the required `group_id` sub-field is rejected. |
| `group_hyperparameter_sweep_valid.json` | **Landed (Discussion #45 / PR #54).** `group` grounded in a second, unrelated domain — an Optuna hyperparameter grid-search trial — proving `role`/`sequence` aren't silently mutation-testing-shaped. |
| `group_role_metadata_real_gap_valid.json` / `group_role_metadata_inert_survivor_valid.json` | **Landed (Discussion #45 / PR #54).** The maintainer-confirmed refinement from `AshwinUgale/muteval`: `role: "survived"` stays coarse, while `metadata.output_changed` (reusing muteval's own field name) distinguishes a real coverage gap from an inert/equivalent mutant — both branches. |
| `group_multi_model_comparison_valid.json` | **Landed (Discussion #45 / PR #54).** `group` grounded in a third domain — a promptfoo multi-model comparison, using the real provider ids and premise from promptfoo's own `examples/compare-claude-vs-gpt-image/promptfooconfig.yaml` — closing out issue #36's own two named use cases beyond mutation testing. |
| `group_nested_parent_group_valid.json` | **Landed (Discussion #45 / PR #54).** `group.parent_group_id` nesting one group under a parent group (a sweep-of-sweeps), MLflow-nested-runs style. |
| `group_self_parent_rejected.json` | **Landed (Discussion #45 / PR #54).** `group.parent_group_id` equal to the object's own `group_id` is rejected (`SELF_PARENT`) — a hand-rolled-only cross-field rule the raw JSON Schema cannot express. |
| `suite_minimal_valid.json` | A minimal valid suite with one shared grader and one test case that references it. |
| `suite_duplicate_test_case_id_rejected.json` | Suite test-case ids are unique; this cross-item rule cannot be expressed in JSON Schema. |
| `suite_duplicate_grader_id_rejected.json` | Suite-level grader ids are unique; this cross-item rule cannot be expressed in JSON Schema. |
| `suite_dangling_grader_reference_rejected.json` | Test-case grader references resolve to a suite-level grader; referential integrity cannot be expressed in JSON Schema. |
| `testcase_empty_string_input_rejected.json` | A TestCase with an empty-string `input` is rejected because string input requires `minLength: 1`. |
| `actual_output_list_rejected.json` | `Result.actual_output` is a string; a list is rejected (`TYPE_ERROR`). Found by ChelseaKR in ChelseaKR/gauntlet#76 — both reference validators used to type-check only required fields. |
| `started_at_date_only_rejected.json` | `started_at` is an RFC 3339 `date-time` (seconds and offset required); a date-only value is rejected (`INVALID_DATE_TIME`). JSON-Schema-based implementations must enable `format` assertion to agree. |
| `result_completed_at_without_offset_rejected.json` | Same rule for the per-result `completed_at`: an offset-less local time (a naive `datetime.isoformat()`) is rejected. |
| `optional_fields_well_typed_valid.json` | Every optional `ResultSet`/`Result`/`GraderResult` field the schema types, present and correctly typed (incl. lowercase `t`/`z` and a `+05:30` offset), is accepted. |
| `null_score_passed_true_rejected.json` | Validation Rule 6: `score: null` MUST have `passed: false` (`NULL_SCORE_PASSED`) — a hand-rolled-only cross-field rule the raw JSON Schema does not encode. Surfaced by frontieror-openeval-adapter. |
| `all_null_scored_result_passed_rejected.json` | Aggregation Extension: a `Result` whose `grader_results` are all null-scored MUST have `passed: false` (`UNSCORED_RESULT_PASSED`) — hand-rolled only. |
| `unknown_field_resultset_rejected.json` | **PROPOSED (issue #107 / Discussion #108).** Strict validation: issue #107's reproduction (a ResultSet-level `verdict`, a Result-level and a GraderResult-level unknown key) is rejected with `UNKNOWN_FIELD` at each key's path. |
| `unknown_field_resultset_allow_unknown_valid.json` | **PROPOSED (#108).** `mode: allow_unknown`: the same document is accepted by a lenient consumer. |
| `unknown_field_resultset_nested_objects_rejected.json` | **PROPOSED (#108).** Unknown keys inside `provider`, `runner`, `summary`, `group` and `Result.error` are each rejected; `summary.scenarios` and `error.detail` come from real adapters the #108 impact sweep found. |
| `unknown_field_metadata_stays_open_valid.json` | **PROPOSED (#108).** Strict mode leaves `metadata` (all three levels), `provider.extra` and `summary.by_grader` entries open. |
| `unknown_field_suite_rejected.json` | **PROPOSED (#108).** A suite-level `threshold` (ChelseaKR/gauntlet#76) and unknown keys in `config`, `config.retry` and a TestCase are rejected. |
| `unknown_field_suite_allow_unknown_valid.json` | **PROPOSED (#108).** `mode: allow_unknown`: the same suite is accepted. |
| `unknown_field_grader_config_instead_of_params_rejected.json` | **PROPOSED (#108).** An `llm_judge` grader with `config` where `params` was meant (meta-llama/llama-cookbook#1072) is rejected at `$.config`. |
| `unknown_field_testcase_rejected.json` | **PROPOSED (#108).** A TestCase's unknown top-level key and unknown `provider` key are rejected; its `metadata`/`params` stay open. |
| `unknown_field_allow_unknown_still_type_checks_rejected.json` | **PROPOSED (#108).** `mode: allow_unknown` relaxes only the unknown-field check: a list-typed `actual_output` is still a `TYPE_ERROR`. |

This set is deliberately not exhaustive — it's the fixtures that came directly out of building 30 real framework adapters and encountering these exact edge cases in practice (see the `description` field on each fixture for which adapter surfaced it), plus the RFC conventions (#10, #11, and #45 — all landed; see `spec/SPEC.md`'s Grouped/Sibling ResultSets section for #45's history) it made sense to ship fixtures for at the same time their spec text landed. Contributions of new fixtures — especially ones derived from a *real* edge case you hit building or consuming an EvalPort document, not a hypothetical one — are welcome via the same RFC process as any other spec change (see `spec/SPEC.md`'s Governance section); a new fixture that isn't also a spec/behavior change doesn't need the full two-week comment period, just a PR.

## What this doesn't cover (yet)

This suite currently only exercises `spec/schemas/*.json`-level structural validity — whether a document is a well-formed EvalPort document. It does not (yet) have fixtures for the CLI's runtime behavior (`openeval run`'s cost estimation, retry logic, grader execution) or cross-document referential rules beyond what `validate_suite()` already checks (dangling grader references, duplicate IDs). Those would be reasonable extensions of this suite if someone wants to take them on — flagged here rather than silently treated as "done" by this README's existence.
