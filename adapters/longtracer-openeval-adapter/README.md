# longtracer-openeval-adapter

Convert [LongTracer](https://github.com/ENDEVSOLS/LongTracer)'s citation-verification
results (`VerificationResult`) into [EvalPort](https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM evaluation results.

Proposed in [ENDEVSOLS/LongTracer#15](https://github.com/ENDEVSOLS/LongTracer/issues/15).
This package implements exactly the v1 scope the LongTracer maintainer asked for
in their review of that issue: `VerificationResult` + batches, response-level
fields preserved, claim-level evidence preserved, unsupported claims kept
distinct from confirmed hallucinations, tests against EvalPort's real
validators, a documented field mapping, and no recalculated scores. It is a
one-way `to_openeval()` only — there is no `from_openeval()`, since LongTracer
produces verification results rather than consuming eval suites.

## Why a standalone package (for now)?

The maintainer's stated preference is for this to eventually live inside
LongTracer itself, as an optional `longtracer/adapters/evalport.py` with no
hard EvalPort dependency added to LongTracer's own install. This package is
written to make that easy: it's a single, dependency-light module (`src/
longtracer_openeval_adapter/__init__.py`) that duck-types against
`VerificationResult`'s public shape rather than importing `longtracer`
directly — the same pattern already used by every adapter in this repo (e.g.
[crewai-openeval-adapter](../crewai-openeval-adapter)) to avoid a hard
dependency on the framework being adapted. It's published here first so the
mapping can be reviewed and exercised against real EvalPort validators before
anything is proposed for inclusion in LongTracer's own tree.

## Install

```bash
pip install "longtracer-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/longtracer-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support.

## Usage

```python
from longtracer.guard.verifier import CitationVerifier
from longtracer_openeval_adapter import to_openeval
from openeval.validate import validate_result_set

verifier = CitationVerifier()
result = verifier.verify_parallel(response, sources)  # a VerificationResult

result_set = to_openeval(result, response_texts=None)
# or, with the original response text preserved as `actual_output`:
result_set = to_openeval(result, response_texts=[response])

assert validate_result_set(result_set).valid

import json
with open("run.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

Batches (as produced by `CitationVerifier.verify_batch()`) work the same way,
passing a list instead of a single result:

```python
results = verifier.verify_batch(items)  # list[VerificationResult]
result_set = to_openeval(
    results,
    test_case_ids=[f"item_{i}" for i in range(len(items))],
    response_texts=[item["response"] for item in items],
    run_id="my_eval_run_2026-08-31",
)
```

## Mapping

Each `VerificationResult` becomes one EvalPort `Result`; each claim in
`VerificationResult.claims` becomes one `GraderResult` (type `"custom"`)
inside that `Result.grader_results`.

### Response-level (`VerificationResult` → `Result`)

| LongTracer field | EvalPort field | Notes |
|---|---|---|
| `verdict` | `Result.passed` (as `verdict == "PASS"`) | Direct rename, not recomputed — `verdict` is already `all_supported and hallucination_count == 0`. |
| `trust_score` | `Result.metadata.trust_score` | Passthrough. |
| `verdict` | `Result.metadata.verdict` | Passthrough (also drives `passed`, above). |
| `summary` | `Result.metadata.summary` | Passthrough. |
| `all_supported` | `Result.metadata.all_supported` | Passthrough. |
| `hallucination_count` | `Result.metadata.hallucination_count` | Passthrough. |
| `len(flagged_claims)` | `Result.metadata.flagged_claim_count` | Simple count, not a new judgment. |
| `latency_stats` | `Result.metadata.latency_stats` | Passthrough (whole dict, unmodified). |
| `latency_stats["total_ms"]` | `Result.duration_ms` | Copied when present; EvalPort's field is an int, so this is `int()`-truncated from LongTracer's float ms. |
| *(not tracked)* | `Result.actual_output` | LongTracer's `VerificationResult` does not retain the response text it verified. Omitted unless the caller supplies it via `response_texts=`. |
| *(not tracked)* | `ResultSet.started_at` | LongTracer does not track a run start time. Defaults to the conversion timestamp unless `started_at=` is given. |
| *(caller-supplied)* | `ResultSet.run_id` / `suite_id` | LongTracer has no run/suite concept; sane defaults are used (`"longtracer_run"` / `"longtracer_citation_verification"`) if not supplied. |

### Claim-level (each entry of `VerificationResult.claims` → `GraderResult`)

| LongTracer field | EvalPort field | Notes |
|---|---|---|
| `supported` | `GraderResult.passed` | Direct passthrough. |
| `score` | `GraderResult.score` | Clamped to EvalPort's required `[0.0, 1.0]` range (cosine similarity is unbounded to `[-1.0, 1.0]`; LongTracer's own models effectively never produce a negative value in practice, but the schema requires the clamp). The **unclamped original is always preserved** in `GraderResult.metadata.openeval.raw_score` — this conversion never silently changes a LongTracer score. |
| `is_hallucination` | `GraderResult.metadata.is_hallucination` | Passthrough. |
| `best_source` | `GraderResult.metadata.best_source` | Passthrough. |
| `claim`, `best_score`, `best_source_index`, `best_source_metadata`, `contradiction_score`, `entailment_score`, `nli_ran`, `is_meta_statement`, `has_hallucination_pattern`, `sentence_results` | `GraderResult.metadata.*` | Passthrough, same key names, for full fidelity. |
| *(derived)* | `GraderResult.metadata.openeval.claim_status` | One of `"supported"` / `"unsupported"` / `"hallucination"` — a pure relabeling of `supported` + `is_hallucination`, computed **not** to lose information: an unsupported-but-not-flagged claim (`claim_status="unsupported"`) is always distinguishable from a confirmed hallucination (`claim_status="hallucination"`), even though both have `passed=False`. This is what satisfies "keep unsupported claims distinct from confirmed hallucinations" from the issue #15 review. |
| *(derived)* | `GraderResult.type` | Always `"custom"` — LongTracer's citation check isn't one of EvalPort's standard grader types, and this package does not emit a companion `EvalSuite`/`Grader` definition (see below), so `"custom"` with per-claim metadata is the closest honest fit. |
| *(derived)* | `GraderResult.reason` | A short, deterministic human-readable string built from `supported`/`is_hallucination` only — no new scoring judgment. |

### What is *not* representable losslessly

- **The original response text.** Not retained by `VerificationResult` — pass
  it explicitly via `response_texts=` if you want it in `actual_output`.
- **A formal `EvalSuite`/`Grader` definition.** This adapter produces a
  `ResultSet` only, not a companion suite with `test_cases`/`graders` arrays.
  Each `test_case_id` is a synthetic id scoped to the conversion call (or
  caller-supplied), not a cross-reference into a persisted `EvalSuite` — so
  the "referential integrity" rules in `spec/SPEC.md` that apply to a
  `TestCase`'s `graders` array don't apply here; there is no suite for them
  to reference.
- **Sub-sentence granularity beyond what's copied.** `sentence_results` is
  copied into `GraderResult.metadata` verbatim (it's already a short,
  truncated list in LongTracer's own output), but it is not restructured
  into further EvalPort-native fields — it's opaque metadata to any consumer
  that doesn't already know LongTracer's shape.

## Tests

```bash
pip install -e ".[test]"
pytest tests/
```

Tests build a hand-rolled stand-in for `VerificationResult` that matches the
real dataclass field-for-field (see `tests/test_adapter.py` docstring for the
exact source commits this was checked against), and validate every produced
`ResultSet` against EvalPort's real `openeval.validate.validate_result_set()`
— not a hand-rolled schema check.

## Spec

See the full EvalPort specification at
<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
