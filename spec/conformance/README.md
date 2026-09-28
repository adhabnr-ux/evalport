# Conformance Test Suite

This directory contains concrete test fixtures (JSON files) designed to test conformance for EvalPort SDK implementations and validators.

## Overview

This set is deliberately not exhaustive — it's the fixtures that came directly out of building 30 real framework adapters and encountering these exact edge cases in practice (see the `description` field on each fixture for which adapter surfaced it), plus the RFC conventions (#10, #11, and #45 — all landed; see `spec/SPEC.md`'s Grouped/Sibling ResultSets section for #45's history) it made sense to ship fixtures for at the same time their spec text landed. Contributions of new fixtures — especially ones derived from a *real* edge case you hit building or consuming an EvalPort document, not a hypothetical one — are welcome via the same RFC process as any other spec change (see `spec/SPEC.md`'s Governance section); a new fixture that isn't also a spec/behavior change doesn't need the full two-week comment period, just a PR.

## Raw Schema Agreement and Hand-Rolled Validation

`expect.valid` reflects a conformant validator's expected output. For most fixtures, the raw JSON Schema (`spec/schemas/*.json`) strictly enforces this. 

However, some rules (such as cross-field or cross-item constraints like `duplicate_attempt_collision_rejected.json` and `group_self_parent_rejected.json`, or suite-level duplicate IDs and dangling references) cannot be fully expressed in plain JSON Schema alone. For these specific fixtures, the raw JSON Schema accepts them, but the SDKs enforce them using hand-rolled validators. Both validation paths are maintained, and test suites (`test_schema_consistency.py` and `schema-consistency.test.ts`) document this distinction explicitly.

## What this doesn't cover (yet)

This suite currently exercises structural validity and known edge-case fixtures. It does not (yet) have fixtures for the CLI's runtime behavior (`openeval run`'s cost estimation, retry logic, grader execution). Those would be reasonable extensions of this suite if someone wants to take them on — flagged here rather than silently treated as "done" by this README's existence.
