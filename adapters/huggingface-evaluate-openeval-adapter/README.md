# huggingface-evaluate-openeval-adapter

Convert [Hugging Face `evaluate`](https://github.com/huggingface/evaluate) metric definitions and computed scores to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install huggingface-evaluate-openeval-adapter
```

This installs `evalport-sdk` as its only hard dependency. `evaluate` itself is an opt-in extra:

```bash
pip install "huggingface-evaluate-openeval-adapter[evaluate]"
```

## Why this adapter looks a little different from the others

Every other adapter in this repository converts a framework's own suite/scenario object model. `evaluate` doesn't have one — its core object is an `EvaluationModule` (`evaluate.load(name)`, covering the library's Metric/Comparison/Measurement subtypes), and its whole interface is `.compute(predictions=..., references=..., **kwargs) -> dict`: one aggregate number (or a small dict of them) for a whole batch, computed on demand. There's no "suite of test cases to be scored later" concept to convert from, and — more importantly — **no per-example score in the return value**, by design. That's true of `evaluate.load("exact_match")` just as much as `evaluate.load("bleu")`.

EvalPort's `ResultSet`, by spec, requires one `Result` per test case with its own `GraderResult.score` (`spec/schemas/resultset.json`: `results` is `minItems: 1`, and each item requires `test_case_id`/`grader_results`/`passed`). Bridging that gap honestly — without fabricating N scores out of one aggregate number — is the actual design problem this adapter solves.

## How it solves that: real per-example calls, not interpolation

`compute_per_example(metric_name, predictions, references, **kwargs)` gets real per-example scores the only honest way available: it calls the *same* metric's real `.compute()` once for the whole batch (the true aggregate `evaluate` itself would report) **and** once per individual example (`predictions=[p]`, `references=[r]`), and returns both. Every number this function returns came from an actual `EvaluationModule.compute()` call — none are interpolated, divided, or otherwise derived from the aggregate.

This is meaningful for metrics whose scoring is honestly independent per example — `exact_match`, `accuracy`, `f1` on a single-item basis, and similar. It is **not** meaningful in the usual sense for a metric that's inherently a corpus-level statistic: classic corpus BLEU's smoothing and brevity-penalty terms are computed over the whole corpus, not per sentence, so calling it once per example still returns a real number from the real metric, but that number means something subtly different from what "BLEU" conventionally refers to. This module has no general, reliable way to detect that distinction from the `evaluate` API alone — the caller is expected to know whether their metric's per-example score is the right thing to report. See "What round-trips losslessly, and what doesn't" below.

## Usage

### Suite definition — the (inputs, references, metric) triple → EvalPort suite, and back

```python
from huggingface_evaluate_openeval_adapter import to_openeval, from_openeval

inputs = ["What is the capital of France?", "What is 2+2?"]
references = ["Paris", "4"]

suite = to_openeval(inputs, references, "exact_match", suite_id="geo_and_math")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

# ...and back: grouped by metric, ready to feed to compute_per_example().
groups = from_openeval(suite)
for group in groups:
    print(group["metric_name"], group["inputs"], group["references"])
```

`inputs` is a required, separate argument — not inferred from `references` — because EvalPort's `TestCase.input` must be a real, non-empty prompt (`minLength: 1`), and fabricating one from the reference would misrepresent what was actually asked of the system under test. `evaluate` itself has no concept of "the prompt that produced this prediction," so the caller supplies it.

`references` may be any type the target metric expects — `int` class labels for `accuracy`/`f1`, not just `str`. EvalPort's `expected_output` is string-typed by spec, so non-string references are stored via `str()`, with the original Python type preserved under `metadata.huggingface_evaluate.reference_type` so `from_openeval()` casts back precisely (`int("1") == 1`) rather than guessing a type from the string's shape.

### Executed scores → EvalPort ResultSet, using real per-example computation

```python
from huggingface_evaluate_openeval_adapter import compute_per_example, metric_result_to_openeval

predictions = ["Paris", "5", "Berlin"]  # what your system actually produced
references = ["Paris", "4", "Berlin"]

item_scores, aggregate = compute_per_example("exact_match", predictions, references)
# item_scores == [1.0, 0.0, 1.0] -- each from its own real .compute() call
# aggregate == {"exact_match": 0.666...} -- the real whole-batch number

result_set = metric_result_to_openeval(
    predictions, references, "exact_match", item_scores,
    suite_id="geo_and_math", run_id="run-2026-08-20", started_at="2026-08-20T00:00:00Z",
    aggregate=aggregate,
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

The real aggregate, when supplied, is preserved verbatim under `result_set["metadata"]["huggingface_evaluate"]["aggregate"]` — nothing is thrown away in favor of the derived per-example view; both are available side by side.

## Grader mapping

| `evaluate` metric | EvalPort grader `type` | Direction |
|---|---|---|
| `exact_match` | `exact_match` | both |
| any other metric (`accuracy`, `f1`, `bleu`, `rouge`, `bertscore`, a community metric, ...) | `custom` (`params.handler = "huggingface_evaluate:<metric_name>"`, `params.metric_kwargs` carries any extra `.compute()` keyword arguments) | both |

`exact_match` is the one metric with a direct, zero-required-param EvalPort grader equivalent — per `spec/schemas/grader.json`, `exact_match` has no `allOf` branch, meaning no required params at all. Every other metric takes metric-specific keyword arguments (`f1`'s `average`, `rouge`'s `rouge_types`, `bertscore`'s `model_type`, ...) this adapter has no honest default for, so — following the same rule every adapter in this ecosystem uses (see the [Haystack](../haystack-openeval-adapter) or [Evidently](../evidently-openeval-adapter) adapters for the same convention) — it maps to `custom` with the real handler name and kwargs preserved, rather than guessing at required params for a native grader type.

## What round-trips losslessly, and what doesn't

**Suite side (`to_openeval`/`from_openeval`) round-trips cleanly**, including the reference's original Python type (`int`, `float`, `str`) via `metadata.huggingface_evaluate.reference_type`, and metric keyword arguments via `params.metric_kwargs`.

**Results side is honestly lossy in one specific, documented way:** `compute_per_example()`'s per-example scores are real numbers from real `.compute()` calls, but for a metric whose scoring is a genuine corpus-level statistic rather than an independent per-example one, "the per-example score" isn't a well-defined concept the same way it is for `exact_match`/`accuracy`. This adapter doesn't try to detect which category a given metric name falls into (there's no reliable signal for that in the `evaluate` API), so it's on the caller to know their metric well enough to judge whether per-example scoring means what they expect for it. The real whole-batch aggregate is always available alongside the per-example breakdown specifically so a consumer isn't stuck trusting only the potentially-misleading per-example view — `metadata.huggingface_evaluate.aggregate` is where the number `evaluate` itself would actually report lives.

A `custom`-typed grader's full `evaluate` metric name and keyword arguments are preserved under `grader.params` on export — nothing is silently dropped, even though this adapter doesn't attempt to reconstruct arbitrary metric-specific scoring logic on import (the same `"custom"` clean-skip convention this whole ecosystem uses).

## Running the tests

```bash
cd adapters/huggingface-evaluate-openeval-adapter
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest tests/ -v
```

25 tests, all running against the real `evaluate` package (0.4.6 at last verification) — real `exact_match`/`accuracy`/`f1` `EvaluationModule.compute()` calls, both whole-batch and per-example — and the real `openeval.validate.validate_suite()`/`validate_result_set()`, not mocks. Network access is required the first time a given metric is used: `evaluate.load()` fetches that metric's builder script from the Hugging Face Hub even for metrics (like `exact_match`) with no other runtime dependency. `accuracy`/`f1` additionally require `scikit-learn`, included in the `test` extra.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
