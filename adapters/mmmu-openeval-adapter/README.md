# mmmu-openeval-adapter

Convert [MMMU](https://github.com/MMMU-Benchmark/MMMU) ("A Massive
Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert
AGI") dataset samples and graded evaluation runs to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

Filed and built following
[MMMU-Benchmark/MMMU#87](https://github.com/MMMU-Benchmark/MMMU/issues/87),
where the MMMU team asked for this as a standalone adapter living in the
EvalPort repo rather than inside MMMU itself.

## Why a standalone package?

MMMU's evaluation surface (`mmmu/main_eval_only.py`,
`mmmu/utils/{data_utils,eval_utils}.py`) is research code, not an installable
package — there's nothing to add an optional export hook to without
depending on the whole repo layout. This package works against plain dicts
shaped like MMMU's own sample and eval-result objects (see the source
docstring for the exact shapes, read directly from MMMU's code rather than
guessed), so it converts real MMMU output without needing MMMU itself as a
dependency.

## Install

```
pip install "mmmu-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/mmmu-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support.

## Usage

### Dataset samples -> EvalPort Suite

```python
from mmmu_openeval_adapter import to_openeval, from_openeval

# One dict per sample, in the shape MMMU's own
# utils.data_utils.process_single_sample() produces:
# {"id", "question", "options", "answer", "question_type", ...}
samples = [
    {
        "id": "validation_Accounting_1",
        "question": "What is the value of X in the balance sheet?",
        "options": "['10', '20', '30', '40']",
        "answer": "B",
        "question_type": "multiple-choice",
    },
    {
        "id": "validation_Math_7",
        "question": "What is the area of the shaded region?",
        "answer": "3.14",
        "question_type": "open",
    },
]

suite = to_openeval(samples, suite_id="mmmu_validation")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

tasks = from_openeval(suite)  # recover MMMU-shaped sample dicts
```

Multiple-choice test cases are graded with the standard `exact_match`
grader; open questions get a `custom` grader identifying MMMU's own
`eval_open()` semantics (number/string normalization + substring match),
since that logic has no honest standard-grader equivalent. Category
(`"Accounting"`) and domain (`"Business"`, via MMMU's own
`DOMAIN_CAT2SUB_CAT`) are derived from the id and recorded in
`metadata.mmmu`.

Images are not embedded (MMMU dataset samples carry `PIL.Image.Image`
objects, which aren't JSON-serializable or portable) — only a
`metadata.mmmu.has_image` flag is set. Join back on `TestCase.id` against
the original MMMU sample if you need the actual image.

### Graded run -> EvalPort ResultSet

This is the direction [#87](https://github.com/MMMU-Benchmark/MMMU/issues/87)
asked for: exporting `main_eval_only.py`'s own per-category
`evaluation_result` / `printable_results`, not just the dataset.

```python
from mmmu_openeval_adapter import result_to_openeval

# exampels_to_eval: built by main_eval_only.py right before evaluate()
exampels_to_eval = [
    {"id": "validation_Accounting_1", "question_type": "multiple-choice",
     "answer": "B", "parsed_pred": "B"},
    {"id": "validation_Math_7", "question_type": "open",
     "answer": "3.14", "parsed_pred": [3.14, "3.14"]},
]

# judge_dict: evaluate(exampels_to_eval)[0]
judge_dict = {"validation_Accounting_1": "Correct", "validation_Math_7": "Correct"}

# printable_results: the per-category/per-domain rollup built at the bottom
# of main_eval_only.py from DOMAIN_CAT2SUB_CAT + calculate_ins_level_acc()
printable_results = {
    "Overall-Business": {"num": 1, "acc": 1.0},
    "Accounting": {"num": 1, "acc": 1.0},
    "Overall-Science": {"num": 1, "acc": 1.0},
    "Math": {"num": 1, "acc": 1.0},
    "Overall": {"num": 2, "acc": 1.0},
}

result_set = result_to_openeval(
    exampels_to_eval,
    judge_dict,
    printable_results,
    suite_id="mmmu_validation",
    run_id="run_2026_09_24",
    started_at="2026-09-24T00:00:00Z",
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

`judge_dict`'s per-item `"Correct"`/`"Wrong"` verdicts become each
`Result.passed` (carried through verbatim, never re-derived — this adapter
does not reimplement MMMU's own `eval_multi_choice`/`eval_open` comparison).
`printable_results` is preserved verbatim as `ResultSet.summary`, so the
per-discipline breakdown that's central to what MMMU measures (broad
expert-level knowledge across Art & Design, Business, Science, Health &
Medicine, Humanities & Social Science, and Tech & Engineering) survives the
round trip untouched rather than collapsing to one aggregate accuracy
number.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 - see LICENSE.
