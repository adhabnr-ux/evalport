# uptrain-openeval-adapter

Convert [UpTrain](https://github.com/uptrain-ai/uptrain) evaluation datasets and results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install uptrain-openeval-adapter
```

## Usage

### Dataset → EvalPort suite, and back

```python
from uptrain_openeval_adapter import to_openeval, from_openeval

data = [
    {
        "id": "0",
        "question": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "context": "France is a country in Europe. Its capital is Paris.",
        "ground_truth": "Paris",
    },
]

suite = to_openeval(data, suite_id="geo_eval")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

# ...and back: load an EvalPort suite as UpTrain-shaped rows
rows = from_openeval(suite)

from uptrain import EvalLLM, Evals
eval_llm = EvalLLM(openai_api_key="sk-...")
results = eval_llm.evaluate(data=rows, checks=[Evals.CONTEXT_RELEVANCE, Evals.RESPONSE_RELEVANCE])
```

UpTrain rows use its default `DataSchema` field names — `question`, `response`, `context`, `ground_truth` — which map directly onto EvalPort's `TestCase.input`/`expected_output`/`context`. UpTrain's `response` is the answer *already generated elsewhere* that UpTrain is scoring, not a task input, so it has no home on an EvalPort `TestCase` (the same situation the [Ragas adapter](../ragas-openeval-adapter) handles for Ragas's own `answer` field) — `to_openeval()` keeps the full original row, `response` included, under the test case's `metadata["uptrain"]["row"]`, and `from_openeval()` reconstructs it verbatim when present, so a round trip through this adapter never drops a field UpTrain would need to re-run.

### Evaluation results → EvalPort ResultSet

```python
from uptrain_openeval_adapter import results_to_openeval

# `results` is exactly what EvalLLM.evaluate() returns: each row from `data`
# with score_<check>/explanation_<check> keys added per check.
result_set = results_to_openeval(results, suite_id="geo_eval", run_id="run-1")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every `score_<check>` key UpTrain adds becomes one EvalPort `GraderResult` (`score_confidence_<check>`, which UpTrain adds separately to record the judge's confidence in its own score, is folded into that grader result's `metadata.confidence` rather than treated as a second grader). Scores are clamped into EvalPort's required `[0, 1]` range and pass at `>= 0.5`; the matching `explanation_<check>` becomes the grader result's `reason`. Each result's `actual_output` comes from the row's `response` field — the answer UpTrain was scoring.

## What round-trips losslessly, and what doesn't

Exporting a dataset to EvalPort and back into UpTrain (`to_openeval` → `from_openeval` → `EvalLLM.evaluate(data=...)`) is lossless: the full original row — including `response`, `id`, and any extra fields UpTrain's schema supports (`conversation`, `sub_questions`, etc., if present) — is preserved verbatim under `metadata["uptrain"]["row"]` and reconstructed exactly on the way back. What doesn't survive a round trip through a *different* tool is that UpTrain-specific row shape — a receiving tool that isn't UpTrain only sees the generic `input`/`expected_output`/`context`/`metadata` fields, the same tradeoff every adapter in this repository makes (see the [Weave adapter](../weave-openeval-adapter) for the same pattern with Weave-specific fields).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
