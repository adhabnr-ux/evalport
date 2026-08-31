# ragrank-openeval-adapter

Convert [ragrank](https://github.com/izam-mohammed/ragrank) evaluation results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets and results.

Tracking issue: [izam-mohammed/ragrank#63](https://github.com/izam-mohammed/ragrank/issues/63).

## Why a standalone package?

The maintainer confirmed on #63 that this belongs as a separate package rather than in ragrank core: ragrank's own roadmap direction is a plain, framework-agnostic JSONL interchange (`Dataset.to_records()` / `to_json()` / `to_jsonl()`), and an EvalPort adapter naturally sits on top of that rather than beside it inside ragrank itself. This package works against ragrank's public `EvalResult` / `Dataset` / `DataNode` / `BaseMetric` shapes from the outside, so it needs nothing merged into ragrank's core to be useful today.

## Install

```bash
pip install "ragrank-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/ragrank-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support. `to_openeval()` has no hard dependency on `ragrank` (it duck-types over attribute access, same as this repo's other adapters); `from_openeval()` constructs a real ragrank `Dataset`, so it needs the `ragrank` extra: `pip install "ragrank-openeval-adapter[ragrank] @ git+..."`.

## Usage

```python
from ragrank_openeval_adapter import to_openeval, from_openeval

# result is whatever ragrank's Evaluation returned (an EvalResult),
# or anything exposing the same attributes.
out = to_openeval(result)
suite, result_set = out["suite"], out["result_set"]

from openeval.validate import validate_suite, validate_result_set
assert validate_suite(suite).valid
assert validate_result_set(result_set).valid

import json
json.dump(suite, open("suite.json", "w"), indent=2)
json.dump(result_set, open("results.json", "w"), indent=2)

# ...and the other direction: an EvalPort suite (optionally paired with a
# ResultSet from some other tool) becomes a ragrank Dataset.
dataset = from_openeval(suite)
```

## Design decisions

Three things came up discussing this adapter on ragrank#63 that the sketch there didn't handle correctly (or at all). All three are also exercised directly in `tests/test_adapter.py`.

### Null scores

`EvalResult.scores[m][i]` is `float | None` — a row a metric couldn't score (missing reference, no relevant documents, an unparseable judge answer) is `None`, not a fabricated zero. This maps cleanly onto EvalPort's own convention: `GraderResult.score` is `number | null`, and Validation Rule 6 says a score-less result is `score: null, passed: false`, explicitly excluded from pass-rate and average-score denominators rather than counted as a scored failure.

This adapter follows that rule end to end, not just at the `GraderResult` level: the `summary.by_grader[...]` breakdown it emits tracks `passed` / `failed` / `skipped` separately, so a null-scored row is visible (`skipped`) without silently dragging `avg_score` or the pass count down as if it had run and failed. `MetricResult.error`, when the full per-row `results` detail is available on the `EvalResult`, is carried into `GraderResult.reason` rather than dropped — it survives the trip even though `GraderResult` has no dedicated `error` field.

### Per-metric score_range normalization

`BaseMetric.score_range` defaults to `(0.0, 1.0)` but is configurable per metric — a custom Likert-style metric might use `(1.0, 5.0)`, for instance. EvalPort's `GraderResult.score` is hard-clamped to `[0.0, 1.0]` with no `score_range` extension (Rule 5): every source scale has to be normalized before it's a valid document.

The adapter linearly rescales `(raw - low) / (high - low)`, clamped defensively to `[0.0, 1.0]`, and — only when a metric's `score_range` isn't already the unit interval — preserves the original value under the reserved `metadata.openeval.raw_score` key (Appendix B), so nothing is lost to the normalization. Pass/fail (`GraderResult.passed`) is computed against the metric's own `threshold` in its **native**, unnormalized scale (ragrank thresholds are expressed in the metric's own terms, e.g. `3.0` on a 1–5 scale), not against the rescaled `[0, 1]` score — rescaling a threshold comparison would be a second, unrelated place to get wrong.

A metric with no `threshold` at all (common for ragrank's deterministic/ranking metrics) has no native pass/fail criterion, but `GraderResult.passed` is a required boolean with no "n/a" value in EvalPort. This adapter's convention, since ragrank itself doesn't state one: **a produced score with no threshold counts as passed** — the metric ran and had nothing to fail against, which is a different situation from a null score (which ran and produced nothing at all, and is `passed: false` per Rule 6). This is a real convention this adapter had to invent, not something read off either spec; it's called out here so it's not a silent surprise.

### `test_case_id` referential integrity

This was the one open question left on ragrank#63 at the point this package was built. EvalPort Rule 2 requires every `Result.test_case_id` in a `ResultSet` to reference a real `TestCase.id` in a paired `Suite`. Ragrank's `Dataset`/`DataNode` has no native `id` concept — nothing on `EvalResult` supplies one, so the original issue sketch's `str(i)` didn't satisfy this against anything real.

**Resolution:** `to_openeval()` defaults to emitting *both* documents — a minimal synthetic `Suite` (via `build_suite()`) alongside the `ResultSet` (via `build_result_set()`) — using positional ids (`tc_0`, `tc_1`, ...) generated identically in both. Referential integrity holds by construction: there is no step where the two id sequences could drift apart, because they come from the same `enumerate()` over the same dataset in the same function call. This was chosen over the "results-only, document that the caller supplies matching ids" alternative because that alternative pushes a correctness requirement (id agreement across two independently-produced documents) onto every caller, silently, with no enforcement — exactly the kind of gap Rule 2 exists to prevent. Defaulting to "just make it correct" is a small amount of extra output (the synthetic suite ~doubles the emitted JSON) for a real guarantee.

The escape hatch: pass `test_case_ids=[...]` (and, in that case, `suite_id` becomes required) when you already have a real, independently-built `EvalSuite` you want these results paired with instead — `to_openeval()` then returns `suite: None`, and skips building anything synthetic. Pairing the returned `result_set` correctly against your real suite (matching `suite_id`, matching per-row ids) is the caller's responsibility at that point, same as the alternative design would have required for every call.

`build_suite()` and `build_result_set()` are also exported individually, in case you want the suite (e.g. to check into version control as `suite.json`) without recomputing the whole result set, or vice versa.

### Other field mappings, briefly

- `DataNode.context` → `TestCase.retrieval_context` (not the generic `TestCase.context`) — ragrank is a RAG evaluation library, so its `context` field *is* retrieved-document context, and `retrieval_context` is EvalPort's more precise field for exactly that.
- `DataNode.reference` → `TestCase.expected_output`; `DataNode.retrieved_ids` / `reference_ids` → `TestCase.metadata` (no first-class EvalPort field for these).
- Every ragrank metric becomes a `custom`-type `Grader` with `params.handler = "ragrank:<slug>"` — ragrank metrics (a `DeterministicMetric` subclass, an `LLMJudge` with an arbitrary rubric, a chunkwise judge) don't map onto EvalPort's well-known grader types in any generic, reliable way, and the spec's type-openness rule exists precisely so a framework-specific grader can identify itself with a handler instead of guessing at one of the 11 standard types.
- `MetricResult.process_time` (when per-row detail is available) sums into `Result.duration_ms`; `EvalResult.usage` and `response_time` land under `ResultSet.metadata.ragrank` (not under the reserved `openeval.*` namespace, since `openeval.cost` per Appendix B is scoped to a single `Result`, not a whole `ResultSet`, and doesn't fit a run-level token total).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
