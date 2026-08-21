# opencompass-openeval-adapter

Convert [OpenCompass](https://github.com/open-compass/opencompass) `CustomDataset` rows and real evaluator scoring output to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why `CustomDataset`, and not one of the 100+ built-in benchmarks?

OpenCompass ships 100+ curated benchmark loaders (`gsm8k.py`, `mmlu.py`, `humaneval.py`, ...) under `opencompass/datasets/`, each with its own bespoke row schema -- there's no single shape shared across all of them to build a generic adapter against. But OpenCompass also exposes one genuinely portable surface for exactly this purpose: [`CustomDataset`](https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/custom.py), the "bring your own eval data" path it documents for users who aren't running one of the curated benchmarks -- a flat list of dict rows loaded from `.jsonl`/`.csv`, either multiple-choice (single-uppercase-letter option columns `A`, `B`, `C`, ... plus an `output_column` naming the correct letter) or free-text QA (`output_column` naming the reference answer). That maps directly onto EvalPort's own portable `TestCase` (input + expected_output + graders), which is exactly the interchange problem EvalPort exists to solve.

## Install

```bash
pip install "opencompass-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/opencompass-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Export `CustomDataset` rows to an EvalPort suite

```python
from opencompass.datasets.custom import CustomDataset
from opencompass_openeval_adapter import to_openeval
from openeval.validate import validate_suite

# real OpenCompass loading path: a .jsonl file of MCQ rows
rows = CustomDataset.load(path="my_mcq_dataset.jsonl")

suite = to_openeval(rows, options=["A", "B", "C", "D"], suite_id="my_mcq_dataset")
assert validate_suite(suite).valid
```

Free-text QA datasets work the same way, minus `options`:

```python
rows = CustomDataset.load(path="my_qa_dataset.jsonl")
suite = to_openeval(rows, suite_id="my_qa_dataset")
```

`rows` can be the real `datasets.Dataset` object `CustomDataset.load()` returns, or any plain `list[dict]` -- e.g. rows you loaded or authored some other way, without going through OpenCompass's loader at all.

### Import an EvalPort suite as `CustomDataset`-ready rows

```python
from opencompass_openeval_adapter import from_openeval
import json

rows = from_openeval(suite)
with open("rows.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

# ...and OpenCompass's own real loader reads it straight back:
reloaded = CustomDataset.load(path="rows.jsonl")
```

### Score real predictions and export an EvalPort ResultSet

```python
from opencompass_openeval_adapter import result_to_openeval
from openeval.validate import validate_result_set

predictions = [my_model(row) for row in rows]  # real model output, however you produced it

result_set = result_to_openeval(
    rows, predictions,
    options=["A", "B", "C", "D"],       # omit for a QA dataset
    suite_id=suite["id"], run_id="run_001", started_at="2026-08-21T00:00:00Z",
)
assert validate_result_set(result_set).valid
```

This calls OpenCompass's own real evaluators directly -- `opencompass.datasets.custom.OptionSimAccEvaluator` for MCQ, `opencompass.openicl.icl_evaluator.AccEvaluator` for QA -- so the scores in the `ResultSet` are exactly what a real OpenCompass run would produce, not a reimplementation.

### The full loop

```python
suite = to_openeval(rows, options=["A", "B", "C", "D"], suite_id="my_dataset")
rows2 = from_openeval(suite)
predictions = [my_model(row) for row in rows2]
result_set = result_to_openeval(
    rows2, predictions, options=["A", "B", "C", "D"],
    suite_id=suite["id"], run_id="run_001", started_at="2026-08-21T00:00:00Z",
)
# result_set["results"][i]["test_case_id"] == suite["test_cases"][i]["id"], preserved end to end
```

## Grader mapping: `custom` for MCQ, genuinely `exact_match` for QA

OpenCompass's real `OptionSimAccEvaluator` (used for MCQ) does *fuzzy* option parsing before comparing to the reference letter: exact letter match first, then a regex-based "first option mentioned" extraction, then substring matching against each option's own text, then a Levenshtein-distance fallback (`OptionSimAccEvaluator.match_any_label`, read directly from `opencompass/datasets/custom.py`). That is not what EvalPort's `exact_match` grader type means -- a literal comparison of `actual_output` to `expected_output` -- so mapping it there would overclaim precision this adapter doesn't have. It maps to `custom` instead, with `params.handler = "opencompass:OptionSimAccEvaluator"` and the real `options` list preserved, the same convention every adapter in this ecosystem uses for scoring logic with no exact EvalPort-native equivalent.

OpenCompass's real `AccEvaluator` (used for QA), by contrast, genuinely does exact string equality: it maps every distinct prediction/reference string to an integer id (`AccEvaluator._preprocess`, read directly from `opencompass/openicl/icl_evaluator/icl_hf_evaluator.py`) and scores those ids with HuggingFace `evaluate`'s `accuracy` metric -- mathematically identical to plain `str(pred) == str(ref)` per item, since distinct strings never collide onto the same id and equal strings always do. So `exact_match` is the honest, not approximate, mapping for the QA path.

## Recovering a per-item breakdown OpenCompass's own evaluator doesn't return

`OptionSimAccEvaluator.score()` returns a real per-item `details` dict natively (`{"<row index>": {"pred", "parsed", "refr", "correct"}}`) -- confirmed, by reading `opencompass/tasks/openicl_eval.py` (the code that actually writes OpenCompass's own on-disk result files), to be dumped *verbatim* into those files when a run sets `dump_details=True`. So for MCQ, `result_to_openeval()` is reading OpenCompass's own real per-item result shape, not inventing one.

`AccEvaluator.score()` does not return per-item details at all -- only the aggregate accuracy. EvalPort's `ResultSet` schema requires one `Result` per test case (`results` is `minItems: 1`), so `result_to_openeval()` computes `str(pred) == str(ref)` per row itself for the QA path. This isn't a guess at what `AccEvaluator` does internally -- per the grader-mapping section above, it *is* what `AccEvaluator` does internally, applied one row at a time instead of in aggregate. This package's test suite cross-checks it directly: `sum(per-item correct) / n` is asserted to equal `AccEvaluator`'s own real `accuracy` output (as a fraction), for every QA case exercised -- not just that the output shapes match.

## What this adapter deliberately does not attempt

OpenCompass's own default prompt-rendering template (`make_mcq_gen_config`/`make_qa_gen_config` in `custom.py` build a specific `HUMAN`/`BOT` turn structure from a row) is not reproduced here -- prompt formatting is a runner-side concern, not a data-interchange concern. `TestCase.input` is the underlying question/context/options text (flattened as `"key: value"` lines, one per column, in the row's own column order), from which any prompt template -- including OpenCompass's own -- can be rendered. Every other adapter in this ecosystem draws the same line (see e.g. [dspy-openeval-adapter](../dspy-openeval-adapter)'s explanation of why the metric callable itself isn't serialized).

The 100+ curated benchmark loaders (`gsm8k.py`, `mmlu.py`, `humaneval.py`, ...) are also out of scope: each has its own bespoke, framework-specific row schema with no shared structure to generalize an adapter over from the outside. `CustomDataset` is the one surface OpenCompass itself designed to be generic, and is what this adapter targets.

## What round-trips losslessly, and what doesn't

OpenCompass → EvalPort → OpenCompass (via this adapter both ways): lossless -- every original row (every column, MCQ or QA) is preserved verbatim under `test_case.metadata.opencompass.row` and restored exactly by `from_openeval()`, confirmed by test (`TestFromOpenevalRoundTrip`, including a full loop through a real `.jsonl` file and OpenCompass's own real `CustomDataset.load()` again).

OpenCompass → EvalPort → some other tool: the flattened input text and expected output are readable by any EvalPort consumer, but a different tool has no way to know which flattened lines were MCQ option choices versus free-form context -- the same tradeoff every adapter here takes for structure that doesn't have a native EvalPort field.

## Testing

28 tests, all passing locally against the real, installed `opencompass` package (0.5.3, the current PyPI release) and the real `openeval.validate.validate_suite()`/`validate_result_set()` -- not mocks. This includes: the real `opencompass.datasets.custom.CustomDataset.load()` loader against real `.jsonl` files on disk; the real `opencompass.datasets.custom.OptionSimAccEvaluator` and `opencompass.openicl.icl_evaluator.AccEvaluator` scoring real predictions; a full MCQ loop through a `.jsonl` file, this adapter, and back into OpenCompass's own loader; and the aggregate-accuracy cross-check described above for the QA path.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
