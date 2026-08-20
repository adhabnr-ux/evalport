# lighteval-openeval-adapter

Convert [Hugging Face `lighteval`](https://github.com/huggingface/lighteval) per-document evaluation details and results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

`lighteval` is the evaluation library behind the Hugging Face Open LLM Leaderboard, and it's built directly on `inspect_ai` — a declared hard dependency, and several of its own task definitions (e.g. `gsm8k`) are written using `inspect_ai`'s `Sample`/`solver`/`scorer` primitives directly, confirmed by reading `lighteval/tasks/tasks/gsm8k.py`'s source. EvalPort already has a merged integration into `inspect_ai` itself ([UKGovernmentBEIS/inspect_ai#4797](https://github.com/UKGovernmentBEIS/inspect_ai/pull/4797)); this adapter closes the loop on the other side of that dependency.

## Install

```bash
pip install lighteval-openeval-adapter
```

This installs `evalport-sdk` as its only hard dependency. `lighteval` itself is an opt-in extra:

```bash
pip install "lighteval-openeval-adapter[lighteval]"
```

## A real bug found while building this adapter

A completely fresh `pip install lighteval` today resolves `xxhash` (an unpinned transitive dependency) to whatever's newest on PyPI — currently 4.0.1. `lighteval/logging/info_loggers.py`'s `DetailsLogger.log()` (and three sibling call sites) calls `xxhash.xxh64(doc.query)` with a raw Python `str`, relying on `xxhash`'s old implicit str-to-bytes encoding, which `xxhash` 4.0 removed. The result: `Pipeline.evaluate()` — the core of `lighteval`'s Python API, and the method every CLI backend (`accelerate`/`vllm`/`endpoint`/etc.) calls under the hood — crashes with `TypeError: Strings must be encoded before hashing` on every run, for anyone installing `lighteval` fresh today. Confirmed two ways: reproduced the bare `xxhash.xxh64("hello")` call in isolation (crashes with the identical message, no lighteval involved), then reproduced the exact same crash from a real `Pipeline.evaluate()` call built for this adapter's own tests.

This was already tracked upstream — [huggingface/lighteval#1330](https://github.com/huggingface/lighteval/issues/1330) — with a fix already up as [huggingface/lighteval#1332](https://github.com/huggingface/lighteval/pull/1332). Left an [independent reproduction comment](https://github.com/huggingface/lighteval/pull/1332) on the fix PR while building this adapter, since it hit the exact same crash from a completely different angle (reading real per-document details for EvalPort conversion, not the original reporter's quantization-eval use case) and confirms all four affected call sites, not just the one in the original traceback.

This adapter's own code never imports or calls `xxhash` — the `test` extra pins `xxhash<4.0` purely so this package's own tests can drive a real `Pipeline.evaluate()` run without hitting that unrelated crash while huggingface/lighteval#1332 is still open.

## Two more real things this module gets right because they were verified, not assumed

1. **`pipeline.get_results()` does not return per-sample data.** It returns the aggregate summary dict (`config_general`/`results`/`versions`/`config_tasks`/`summary_tasks`/`summary_general`). The real per-document records — one `DetailsLogger.Detail(doc, model_response, metric)` per evaluated sample — come from the separate `pipeline.get_details()` method, returning `{task_name: [Detail, ...]}`. This module reads from `get_details()`, not `get_results()`.
2. **Even a classic multiple-choice task is scored generatively in this version.** `hellaswag` looks like a `Doc.choices`/`Doc.gold_index` loglikelihood task, but its real installed `LightevalTaskConfig` uses `Metrics.exact_match` (`metric_name="em"`, backed by `ExactMatches(strip_strings=True)` — confirmed by reading `lighteval/metrics/metrics.py` directly), which scores `model_response.text` (the model's generated string) against the gold choice text, not `model_response.logprobs`/`model_response.argmax_logits_eq_gold`. Because `Metrics.exact_match`'s real `metric_name` genuinely is `"em"` backed by literal exact-match semantics — not just named similarly — mapping it to EvalPort's native `exact_match` grader type is an honest mapping, not a guess.

## Usage

### Suite side — real per-document details to an EvalPort suite, and back

```python
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.dummy.dummy_model import DummyModel, DummyModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval_openeval_adapter import to_openeval, from_openeval

evaluation_tracker = EvaluationTracker(output_dir="./out", save_details=False, push_to_hub=False)
pipeline = Pipeline(
    tasks="hellaswag|0",
    pipeline_parameters=PipelineParameters(launcher_type=ParallelismManager.ACCELERATE),
    evaluation_tracker=evaluation_tracker,
    model_config=...,  # your real model config
)
pipeline.evaluate()

details = pipeline.get_details()["hellaswag|0"]
suite = to_openeval("hellaswag|0", details, suite_id="my_hellaswag_run")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

# ...and back: recovers query/choices/doc-id for every test case this module produced
recovered = from_openeval(suite)
for r in recovered:
    print(r["doc_id"], r["query"][:50])
```

### Results side — real per-document scores to an EvalPort ResultSet

```python
from lighteval_openeval_adapter import result_to_openeval

result_set = result_to_openeval(
    "hellaswag|0", details,
    suite_id="my_hellaswag_run", run_id="run-2026-08-20",
    started_at="2026-08-20T00:00:00Z",
    aggregate=pipeline.get_results()["results"]["hellaswag:0"],  # real corpus-level aggregate, preserved verbatim
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every score in the produced `ResultSet` is the real per-document number `lighteval` itself computed via one of its actual metric functions — nothing here is derived from, or interpolated out of, the aggregate. The real aggregate, when supplied, is preserved verbatim under `result_set["metadata"]["lighteval"]["aggregate"]`.

## Grader mapping

| `lighteval` metric | EvalPort grader `type` | Direction |
|---|---|---|
| `em` (`Metrics.exact_match`, backed by `ExactMatches(strip_strings=True)`) | `exact_match` | both |
| any other metric (`extractive_match`, `f1`, `chrf++`, `edit_distance`, a task-specific metric, ...) | `custom` (`params.handler = "lighteval:<metric_name>"`) | both |

`em` is the one `lighteval` metric name with a direct, zero-required-param EvalPort grader equivalent — per `spec/schemas/grader.json`, `exact_match` has no `allOf` branch, meaning no required params at all. Every other metric has no honest EvalPort-native equivalent this module could map to without fabricating required params, so it maps to `custom` with the real metric name and handler preserved — the same convention every other adapter in this ecosystem uses for a grader type it can't natively represent.

## What round-trips losslessly, and what doesn't

**Suite side round-trips cleanly** for anything this module wrote: query, choices, gold_index, instruction, and task-specific `doc.specific` data are all preserved under `metadata["lighteval"]` and recovered by `from_openeval()`. A `Suite` (or individual `TestCase`) not produced by this module's `to_openeval()` — i.e. missing the `metadata["lighteval"]` namespace — is cleanly skipped by `from_openeval()` rather than guessed at, the same convention every adapter in this ecosystem uses for data it didn't originate.

**Results side is honestly partial in one specific way:** `from_openeval()` recovers enough to identify the original document and re-drive an equivalent evaluation against it (query, choices, gold_index, doc_id) — it does not reconstruct a live `lighteval.tasks.lighteval_task.LightevalTask` object, which needs a full task config (`prompt_function`, `solver`/`scorer` for inspect_ai-backed tasks, HF dataset repo/subset/splits) that isn't portable data. That's consistent with how this ecosystem's other framework-object adapters (e.g. [lm-eval-harness](../lm-eval-harness-openeval-adapter)'s `Task`, [Vertex AI](../vertexai-openeval-adapter)'s `PairwiseMetric`) handle objects that need more live framework state than a portable data format can carry.

## Running the tests

```bash
cd adapters/lighteval-openeval-adapter
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest tests/ -v
```

19 tests, all running real `Pipeline.evaluate()` calls against `lighteval` 0.13.0's own built-in `dummy` model (a genuine evaluation loop using no real model weights, but real tasks/documents pulled live from the Hugging Face Hub) — `hellaswag` (multiple-choice, generative-scored via the real `exact_match`/`em` metric) and `gsm8k` (generation, `inspect_ai`-solver-backed, `extractive_match` metric) — plus the real `openeval.validate.validate_suite()`/`validate_result_set()`. Not mocks. Requires `xxhash<4.0` to work around huggingface/lighteval#1330 (see above) until huggingface/lighteval#1332 merges. Network access is required the first time a given task's dataset is used, since `lighteval` fetches it live from the Hub.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
