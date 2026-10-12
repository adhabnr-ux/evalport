# flashrag-openeval-adapter

Convert [RUC-NLPIR/FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) datasets, items, and evaluator metric scores to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

This started as [RUC-NLPIR/FlashRAG#246](https://github.com/RUC-NLPIR/FlashRAG/issues/246), a proposal to add an adapter either inside FlashRAG or as a standalone package in EvalPort's own `adapters/` directory. The maintainer (`ignorejjj`) said FlashRAG doesn't currently see enough demand to justify adding and maintaining the integration itself, and explicitly invited building it independently ("you're welcome to build an independent adapter in your own repository") — option 1 from that issue, which is exactly what this package is. Zero changes to FlashRAG; built only against its public `Item`/`Dataset`/`Evaluator` shapes from the outside.

The issue also flagged a quirk in `LLMJudge.calculate_metric` (it rescaled a 0–10 rating as `score / 10 + 1`, giving `2.0` for a top rating rather than `1.0`). The maintainer corrected that FlashRAG's current `main` branch no longer has the `+1` — confirmed by reading `flashrag/evaluator/metrics.py` directly: it was fixed in commit [`d98e892`](https://github.com/RUC-NLPIR/FlashRAG/commit/d98e892beb8f5b86574eabb85c3d5c7cd7c5c024) ("fix: correct LLM judge score parsing", 2026-09-29), a week after the issue was filed against an older commit and before the maintainer's reply. This adapter clamps every score into EvalPort's required `[0, 1]` range regardless — both as a backstop against that now-fixed bug recurring and because any other metric could in principle go out of range — and records the original value under `metadata.flashrag_raw_score` whenever a clamp actually changes something, rather than silently masking it.

## Install

```bash
pip install "flashrag-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/flashrag-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support.

## Usage

### Export a FlashRAG dataset to an EvalPort suite

```python
from flashrag.dataset.dataset import Dataset
from flashrag_openeval_adapter import to_openeval_suite

dataset = Dataset(config={"dataset_name": "nq_test"}, dataset_path="nq_test.jsonl")

suite = to_openeval_suite(dataset, metric_names=["em", "f1"], suite_id="nq_test_run")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

Each metric name becomes one `custom` grader (`gr_flashrag_<metric_name>`, `params.handler = "flashrag:<metric_name>"`) attached to every test case, since FlashRAG metrics apply uniformly across a dataset rather than per sample — `em`/`acc`/`retrieval_recall_top{k}`/etc. have no direct EvalPort standard-grader equivalent, so this follows the same "type openness" convention `ares-openeval-adapter` and `ragrank-openeval-adapter` use. Multi-choice items (`item.choices` non-empty) are resolved to literal answer text exactly the way `flashrag.evaluator.metrics.BaseMetric.get_dataset_answer` does it, not left as a bare choice index.

### Export FlashRAG evaluation output to an EvalPort result set

```python
from flashrag.evaluator.evaluator import Evaluator
from flashrag_openeval_adapter import to_openeval_resultset

dataset.update_output("pred", generated_answers)  # from your RAG pipeline

evaluator = Evaluator(config)  # config["metrics"] == ["em", "f1"]
aggregate = evaluator.evaluate(dataset)  # mutates dataset in place

result_set = to_openeval_resultset(
    dataset, metric_names=["em", "f1"], suite_id="nq_test_run", run_id="run_001", aggregate=aggregate,
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Call this **after** `Evaluator.evaluate(dataset)` — that call is what populates `item.output["metric_score"][metric_name]` on each item, which is what `to_openeval_resultset` reads.

`grader_results[j].passed` is `score >= passed_threshold` (default `1.0`). FlashRAG mixes genuinely binary metrics (`em`, `acc`, `retrieval_recall_top{k}`) with continuous ones (`f1`, `rouge-*`, `bleu`, `retrieval_precision_top{k}`, `llm_judge`), and EvalPort's schema requires a bool regardless — pass a lower threshold (e.g. `0.5`) for continuous metrics if a partial score should still count as a pass. The top-level `result.passed` is `True` only when every configured metric passed for that item. A metric that never got a score for some item (e.g. `Evaluator.evaluate` caught and logged an exception for that metric rather than raising) produces `score: null, passed: false` rather than being silently dropped.

### Import an EvalPort suite as a FlashRAG dataset

```python
from flashrag.dataset.dataset import Dataset
from flashrag_openeval_adapter import from_openeval

item_dicts = from_openeval(suite)  # list[dict]: id/question/golden_answers/...
dataset = Dataset(config={"dataset_name": "from_evalport"}, data=item_dicts)
```

## Grounded in

- `flashrag.dataset.dataset.Item` / `Dataset` — `flashrag/dataset/dataset.py`
- `flashrag.evaluator.evaluator.Evaluator` — `flashrag/evaluator/evaluator.py`
- `flashrag.evaluator.metrics.BaseMetric` and its subclasses (`ExactMatch`, `F1_Score`, `Retrieval_Recall`/`Retrieval_Precision`, `LLMJudge`, ...) — `flashrag/evaluator/metrics.py`

all as of [RUC-NLPIR/FlashRAG@main](https://github.com/RUC-NLPIR/FlashRAG) at the time this was written, including commit `d98e892` for the `LLMJudge` fix referenced above.

## Tests

```bash
cd adapters/flashrag-openeval-adapter
pip install -e ".[test]"
pytest -v
```

Tests run against the real `flashrag.dataset.dataset.Item`/`Dataset` and `flashrag.evaluator.evaluator.Evaluator` classes (including a genuine `Evaluator.evaluate()` run through FlashRAG's actual `em`/`f1` metric code) and validate every produced document against EvalPort's real `openeval.validate.validate_suite()`/`validate_result_set()` — no mocks of either side.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
