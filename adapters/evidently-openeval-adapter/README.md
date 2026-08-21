# evidently-openeval-adapter

Convert [Evidently](https://github.com/evidentlyai/evidently) evaluation DataFrames and `Dataset`s to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "evidently-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/evidently-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Export a test-case DataFrame to an EvalPort suite

Evidently's native evaluation surface is a `pandas.DataFrame`, one row per test case, evaluated by attaching `evidently.descriptors.*` objects.

```python
import pandas as pd
from evidently_openeval_adapter import to_openeval

df = pd.DataFrame({
    "question": ["What is the capital of France?", "What is the capital of Japan?"],
    "expected": ["Paris", "Tokyo"],
    "answer": ["Paris", "Tokyo"],
})

suite = to_openeval(
    df,
    input_columns=["question"],
    expected_column="expected",
    graders=["exact_match"],
    descriptor_types={"exact_match": "ExactMatch"},
)

from openeval.validate import validate_suite
assert validate_suite(suite).valid
```

`input_columns` names which column(s) become `TestCase.input` (EvalPort has no named-column concept, so multiple columns are flattened into `["column: value", ...]`, one string per column). `expected_column` names the one column that becomes `expected_output`. `graders` names the descriptor alias(es) this suite's test cases will later be scored against — pass the same `alias=...` strings you'll give each `evidently.descriptors.*` object. `descriptor_types` optionally maps each alias to the Evidently descriptor *class* name (e.g. `"ExactMatch"`) so the resulting grader gets typed correctly instead of defaulting to `custom` — see "Grader type inference" below. On a round trip through this adapter, nothing is lost: every original column value is additionally preserved under `test_case.metadata.evidently.columns`.

### Import an EvalPort suite as a DataFrame, ready to evaluate

```python
from evidently_openeval_adapter import from_openeval
from evidently import Dataset, DataDefinition
from evidently.descriptors import ExactMatch

df = from_openeval(suite)  # -> pandas.DataFrame, columns restored
dataset = Dataset.from_pandas(
    df,
    data_definition=DataDefinition(),
    descriptors=[ExactMatch(columns=["expected", "answer"], alias="exact_match")],
)
```

A suite built by this adapter round-trips its exact original columns. A hand-authored suite (or one from a different EvalPort-speaking tool) instead gets positionally-named columns (`input_1`, `input_2`, ...) unless you pass `input_columns=[...]` explicitly to name them yourself.

`from_openeval()` also always adds an `"id"` column (each row's `TestCase.id`), unless the original columns already used that name for real data. That's what lets `evaluation_result_to_openeval()` recover each row's test case id automatically after the frame has passed through `evidently.Dataset.from_pandas()` and back out — a `pandas.DataFrame` has no hidden slot to carry that bookkeeping invisibly, so here it's a real, visible, documented column instead (same reasoning as [haystack-openeval-adapter](../haystack-openeval-adapter)).

### Export an evaluated Dataset to an EvalPort ResultSet

```python
from evidently_openeval_adapter import evaluation_result_to_openeval

# dataset.as_dataframe() now has an "exact_match" column, one bool per row
result_set = evaluation_result_to_openeval(
    dataset,
    descriptor_columns=["exact_match"],
    suite_id=suite["id"],
    descriptor_types={"exact_match": "ExactMatch"},
    output_column="answer",
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every column named in `descriptor_columns` becomes its own grader result per row; a row's overall `passed` is the AND of every descriptor's `passed` for that row. `bool` values map directly (`score` 1.0/0.0, `passed` the bool itself). Numeric values are clamped into EvalPort's required `[0, 1]` range, with the unclamped raw value preserved in `grader_result.metadata.evidently.raw_score` whenever clamping changed it. Non-numeric values (e.g. an LLM-judge classification label) get `score: null` with the label preserved in `grader_result.reason`; pass an optional `pass_values={"column": {"acceptable_label", ...}}` to control `passed` for those, otherwise they default to `passed=False`.

### The full loop

```python
suite = to_openeval(df, input_columns=["question"], expected_column="expected",
                     graders=["exact_match"], descriptor_types={"exact_match": "ExactMatch"},
                     ids=["fr", "jp"])
df2 = from_openeval(suite)
dataset = Dataset.from_pandas(df2, data_definition=DataDefinition(),
                               descriptors=[ExactMatch(columns=["expected", "answer"], alias="exact_match")])
result_set = evaluation_result_to_openeval(dataset, descriptor_columns=["exact_match"], suite_id=suite["id"])
# result_set["results"][i]["test_case_id"] == suite["test_cases"][i]["id"], preserved end to end
```

## Grader type inference

Unlike `haystack-openeval-adapter`, which can infer a grader type from a fixed, framework-defined metric name, Evidently descriptors have no fixed output column name — it's whatever `alias` the caller chose. So type inference here runs off an explicit `descriptor_types` mapping the *caller* supplies (`{alias: EvidentlyDescriptorClassName}`), not automatic name-matching. Only `"ExactMatch"` is mapped to anything other than EvalPort's `"custom"` grader type, for the same reason `haystack-openeval-adapter` only auto-infers `exact_match`: it's the one descriptor with no required `params`, so nothing is fabricated. `"llm_judge"` requires `params.model`/`params.prompt`, `"semantic_similarity"` requires `params.threshold`, `"regex"` requires `params.pattern` — none of which this module has an honest value for without your actual descriptor configuration.

## What round-trips losslessly, and what doesn't

Evidently → EvalPort → Evidently (via this adapter both ways): lossless — every column's value for every row survives exactly, restored from `metadata.evidently.columns`.

Evidently → EvalPort → some other tool: the flattened `"column: value"` input strings and the single `expected_output` value are readable by any EvalPort consumer, but a different tool has no way to know which column was the actual model input versus retrieved context versus free-form metadata — the same tradeoff every adapter here takes for structure that doesn't have a native EvalPort field.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
