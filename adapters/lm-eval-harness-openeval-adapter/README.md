# lm-eval-harness-openeval-adapter

Convert [EleutherAI `lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) (`lm-eval`) per-document samples and aggregate results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

`lm-evaluation-harness` is the de facto standard few-shot evaluation harness for language models — the benchmark numbers in most open model cards and leaderboards (MMLU, GSM8K, ARC, HellaSwag, and hundreds of other tasks) come from it. Adapting it puts EvalPort in a position to interchange with results from the tool most of the field already measures against.

## Install

```bash
pip install lm-eval-harness-openeval-adapter
```

This installs `evalport-sdk` as its only hard dependency. `lm-eval` itself is an opt-in extra:

```bash
pip install "lm-eval-harness-openeval-adapter[lm-eval]"
```

## Why this adapter only needs one export function, not a workaround

Unlike the [`huggingface-evaluate`](../huggingface-evaluate-openeval-adapter) adapter in this same repo, `lm-eval` doesn't have an aggregate-only API problem to work around. Its `simple_evaluate(..., log_samples=True)` already returns exactly the granularity EvalPort's `ResultSet` schema requires: one `SampleResult` per (document, filter) pair, each carrying that document's own real score (`spec/schemas/resultset.json`'s `results` array, `minItems: 1`, each item requiring `test_case_id`/`grader_results`/`passed` — that's a natural fit for `lm-eval`'s per-document `samples[task_name]` list). The catch is that `log_samples` is opt-in and off by default — this adapter only works against runs that passed it.

## Two real discrepancies found and handled, not assumed away

Everything in this adapter was verified directly against the actually-installed `lm-eval` package (0.4.12), not just against `lm_eval/result_schema.py`'s docstrings:

1. **`SampleResult.arguments`'s docstring describes a dict** (`{"gen_args_N": {"arg_0": ..., "arg_1": ...}}`). The real, installed 0.4.12 returns a **list of lists** instead — `[[context, continuation_or_gen_kwargs], ...]`, one entry per model request for that document. This module reads the real list shape, not the docstring's.
2. **Per-sample metric scores come back as `numpy.float64`**, not plain Python `float` (`sample["exact_match"]`, `sample["acc"]`, etc.) — not JSON-serializable as-is. Every score this module reads is explicitly cast with `float(...)` before it goes anywhere near a `Suite`/`ResultSet` document.

## Usage

### Suite side — real per-document samples to an EvalPort suite, and back

```python
from lm_eval import simple_evaluate
from lm_eval_harness_openeval_adapter import to_openeval, from_openeval

results = simple_evaluate(
    model="hf", model_args="pretrained=your-model",
    tasks=["gsm8k"], log_samples=True,
)
samples = results["samples"]["gsm8k"]

suite = to_openeval("gsm8k", samples, suite_id="my_gsm8k_run")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

# ...and back: recovers prompt/target/doc info for every test case this
# module produced.
recovered = from_openeval(suite)
for r in recovered:
    print(r["doc_id"], r["prompt"][:50], r["target"][:50])
```

A document can appear more than once in `samples` — once per filter (e.g. `gsm8k`'s `"strict-match"`/`"flexible-extract"`). `to_openeval()` deduplicates on `doc_id` into one `TestCase` per document, with one grader per (metric, filter) pair seen for that document across *all* its sample entries — `strict-match` and `flexible-extract` are two meaningfully different scores for the same document, not duplicates to collapse into one.

### Results side — real per-document scores to an EvalPort ResultSet

```python
from lm_eval_harness_openeval_adapter import result_to_openeval

result_set = result_to_openeval(
    "gsm8k", samples,
    suite_id="my_gsm8k_run", run_id="run-2026-08-20",
    started_at="2026-08-20T00:00:00Z",
    aggregate=results["results"]["gsm8k"],  # the real whole-task aggregate, preserved verbatim
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every score in the produced `ResultSet` is the real per-document number `lm-eval` itself computed — nothing here is derived from, or interpolated out of, the aggregate. The real aggregate, when supplied, is preserved verbatim under `result_set["metadata"]["lm_eval"]["aggregate"]` so a consumer isn't limited to only the per-document view.

## Grader mapping

| `lm-eval` metric | EvalPort grader `type` | Direction |
|---|---|---|
| `exact_match` | `exact_match` | both |
| any other metric (`acc`, `acc_norm`, `f1`, `bleu`, `rouge`, `word_perplexity`, a task-specific metric, ...) | `custom` (`params.handler = "lm-evaluation-harness:<metric_name>"`, `params.filter` carries which filter produced this score) | both |

`exact_match` is the one `lm-eval` metric name with a direct, zero-required-param EvalPort grader equivalent — per `spec/schemas/grader.json`, `exact_match` has no `allOf` branch, meaning no required params at all (it's exercised for real by `gsm8k`, whose scoring metric is literally named `exact_match`). Every other metric — `acc`/`acc_norm` for multiple-choice tasks, `f1`/`bleu`/`rouge` for various generation tasks, task-specific metrics — has no honest EvalPort-native equivalent this module could map to without fabricating required params (`semantic_similarity` needs a `threshold` this module has no basis for, `llm_judge` needs a real `model`/`prompt`), so it maps to `custom` with the real metric name, filter, and handler preserved — the same convention every other adapter in this ecosystem uses for a grader type it can't natively represent.

## What round-trips losslessly, and what doesn't

**Suite side round-trips cleanly** for anything this module wrote: prompt, target, the original `doc` dict, and `doc_hash`/`prompt_hash`/`target_hash` (lm-eval's own reproducibility-verification hashes) are all preserved under `metadata["lm_eval"]` and recovered by `from_openeval()`. A `Suite` (or individual `TestCase`) not produced by this module's `to_openeval()` — i.e. missing the `metadata["lm_eval"]` namespace — is cleanly skipped by `from_openeval()` rather than guessed at, the same convention every adapter in this ecosystem uses for data it didn't originate.

**Results side is honestly partial in one specific way:** `from_openeval()` recovers enough to re-drive an equivalent evaluation loop against the *same documents* (prompt, target, original `doc`, `arguments`) — it does not reconstruct a live `lm_eval.api.task.Task` object, which needs a full YAML task config, registered filters, and aggregation functions that aren't portable data. That's consistent with how this ecosystem's other framework-object adapters (e.g. [Vertex AI](../vertexai-openeval-adapter)'s `PairwiseMetric`, [LlamaIndex](../llamaindex-openeval-adapter)'s `PairwiseComparisonEvaluator`) handle objects that need more live framework state than a portable data format can carry.

## Running the tests

```bash
cd adapters/lm-eval-harness-openeval-adapter
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest tests/ -v
```

21 tests, all running real `simple_evaluate(model="dummy", ..., log_samples=True)` calls against `lm-eval` 0.4.12 (a genuine evaluation loop using `lm-eval`'s own built-in stub model — no real model weights, but real tasks/documents pulled live from the Hugging Face Hub) — `copa`/`boolq` (loglikelihood-based multiple choice) and `gsm8k` (generation, two filters, `exact_match` metric) — plus the real `openeval.validate.validate_suite()`/`validate_result_set()`. Not mocks. Network access is required the first time a given task's dataset is used, since `lm-eval` fetches it live from the Hub.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
