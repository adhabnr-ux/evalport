# haystack-openeval-adapter

Convert [Haystack](https://github.com/deepset-ai/haystack) (deepset) evaluation input columns and `EvaluationRunResult` to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install haystack-openeval-adapter
```

## Usage

### Export input columns to an EvalPort suite

Haystack's evaluation surface is columnar: a plain `dict[str, list[Any]]` of named columns, the same shape both `haystack.components.evaluators.*` and `haystack.evaluation.EvaluationRunResult` consume.

```python
from haystack_openeval_adapter import to_openeval

inputs = {
    "questions": ["What is the capital of France?", "What is the capital of Japan?"],
    "ground_truth_answers": ["Paris", "Tokyo"],
    "predicted_answers": ["Paris", "Tokyo"],
}

suite = to_openeval(
    inputs,
    input_keys=["questions"],
    expected_key="ground_truth_answers",
    graders=["answer_exact_match"],
)

from openeval.validate import validate_suite
assert validate_suite(suite).valid
```

`input_keys` names which column(s) become `TestCase.input` (EvalPort has no named-column concept, so multiple keys are flattened into `["key: value", ...]`, one string per key). `expected_key` names the one column that becomes `expected_output`. `graders` names the evaluator metric(s) this suite's test cases will later be scored against — pass the same names `EvaluationRunResult.results` will use as keys (e.g. `"answer_exact_match"`, `"faithfulness"`, `"sas_evaluator"`). On a round trip through this adapter, nothing is lost: every original column value is additionally preserved under `test_case.metadata.haystack.columns`.

### Import an EvalPort suite as input columns, ready to evaluate

```python
from haystack_openeval_adapter import from_openeval
from haystack.components.evaluators import AnswerExactMatchEvaluator

cols = from_openeval(suite)  # -> dict[str, list[Any]], columns restored
result = AnswerExactMatchEvaluator().run(
    ground_truth_answers=cols["ground_truth_answers"],
    predicted_answers=cols["predicted_answers"],
)
```

A suite built by this adapter round-trips its exact original columns. A hand-authored suite (or one from a different EvalPort-speaking tool) instead gets positionally-named columns (`input_1`, `input_2`, ...) unless you pass `input_keys=[...]` explicitly to name them yourself.

`from_openeval()` also always adds an `"id"` column (each row's `TestCase.id`), unless the original columns already used that name for real data. That's not a side effect to work around — it's what lets `evaluation_result_to_openeval()` recover each row's test case id automatically after the columns have passed through a real Haystack evaluator run, with no extra wiring. (A plain `dict[str, list]` has no hidden slot to carry that bookkeeping invisibly the way `dspy-openeval-adapter` hides it on a `dspy.Example` instance attribute — so here it's a real, visible, documented column instead.)

### Export an EvaluationRunResult to an EvalPort ResultSet

```python
from haystack.evaluation import EvaluationRunResult
from haystack_openeval_adapter import evaluation_result_to_openeval

run_result = EvaluationRunResult(
    run_name="my_rag_pipeline",
    inputs=inputs,
    results={
        "answer_exact_match": {"score": 1.0, "individual_scores": [1, 1]},
    },
)
result_set = evaluation_result_to_openeval(run_result, suite_id=suite["id"], output_column="predicted_answers")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every metric in `run_result.results` becomes its own grader result per row; a row's overall `passed` is the AND of every metric's `passed` for that row. Scores are clamped into EvalPort's required `[0, 1]` range, with the unclamped raw value preserved in `grader_result.metadata.haystack.raw_score` whenever clamping changed it, and the metric's own aggregate score preserved in `grader_result.metadata.haystack.aggregate_score`.

### The full loop

```python
suite = to_openeval(inputs, input_keys=["questions"], expected_key="ground_truth_answers",
                     graders=["answer_exact_match"], ids=["fr", "jp"])
cols = from_openeval(suite)
eval_result = AnswerExactMatchEvaluator().run(
    ground_truth_answers=cols["ground_truth_answers"], predicted_answers=cols["predicted_answers"]
)
run_result = EvaluationRunResult("run", inputs=cols, results={"answer_exact_match": eval_result})
result_set = evaluation_result_to_openeval(run_result, suite_id=suite["id"])
# result_set["results"][i]["test_case_id"] == suite["test_cases"][i]["id"], preserved end to end
```

## Grader type inference

Only `"answer_exact_match"` is automatically typed as EvalPort's `"exact_match"` grader — that type has no required `params`, so the mapping can't misrepresent anything. Every other evaluator name (`"faithfulness"`, `"context_relevance"`, `"sas_evaluator"`, `"document_map"`/`"document_mrr"`/`"document_ndcg"`/`"document_recall"`, `"llm_evaluator"`, or any custom name) maps to EvalPort's `"custom"` grader type with `params.handler` set to the metric name.

This is deliberate: EvalPort's `"llm_judge"` type requires `params.model` and `params.prompt`, and `"semantic_similarity"` requires `params.threshold` — none of which this module has an honest value for without your actual Haystack evaluator configuration (a live `chat_generator`, a real prompt template, a chosen similarity threshold). Fabricating placeholder values for required grader params would produce a suite that looks correctly typed but silently misrepresents what will actually run. If you want one of these typed correctly, build the `Grader` dict yourself with your real configuration and pass it directly as a `Suite.graders` entry — see [spec/SPEC.md](https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md).

## What round-trips losslessly, and what doesn't

Haystack → EvalPort → Haystack (via this adapter both ways): lossless — every input column's value for every row survives exactly, restored from `metadata.haystack.columns`.

Haystack → EvalPort → some other tool: the flattened `"key: value"` input strings and the single `expected_output` value are readable by any EvalPort consumer, but a different tool has no way to know which Haystack column was the actual pipeline input versus retrieved context versus free-form metadata — the same tradeoff every adapter here takes for structure that doesn't have a native EvalPort field.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
