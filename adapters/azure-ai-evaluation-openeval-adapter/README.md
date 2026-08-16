# azure-ai-evaluation-openeval-adapter

Convert [Azure AI Evaluation SDK](https://pypi.org/project/azure-ai-evaluation/) (`azure-ai-evaluation`, part of the Azure AI Foundry / azure-sdk-for-python family) data rows, evaluators, and `EvaluationResult`s to/from [EvalPort](https://github.com/adhabnr-ux/evalport) (Apache 2.0), the open interchange format for portable LLM evaluation datasets -- test cases, graders, suites, and results as plain JSON, shared across DeepEval, Promptfoo, Inspect AI, AutoGen, CrewAI, Ragas, LangSmith, Braintrust, MLflow, and 15+ other frameworks.

## Install

```bash
pip install azure-ai-evaluation-openeval-adapter
```

## Usage

### Export `evaluate()`'s own inputs to an EvalPort Suite

```python
from azure.ai.evaluation import F1ScoreEvaluator
from azure_ai_evaluation_openeval_adapter import to_openeval
from openeval.validate import validate_suite

suite = to_openeval(
    data="my_eval_data.jsonl",           # same shape evaluate(data=...) accepts
    evaluators={"f1": F1ScoreEvaluator()},  # same shape evaluate(evaluators=...) accepts
    suite_id="my_eval_suite",
)
assert validate_suite(suite).valid
```

`to_openeval()` accepts exactly what `azure.ai.evaluation.evaluate()` itself accepts for `data` (a `.jsonl` path or an already-loaded list of row dicts) and `evaluators` (a dict of evaluator instances -- built-in or custom). Every evaluator becomes one EvalPort grader, referenced by every test case, mirroring how `evaluate()` runs every evaluator against every row. An optional `evaluator_config` (the same column-mapping dict `evaluate()` accepts) is preserved verbatim in the suite's metadata for a lossless round trip.

### Load an EvalPort suite as `evaluate()` input rows

```python
from azure_ai_evaluation_openeval_adapter import from_openeval
from azure.ai.evaluation import evaluate, F1ScoreEvaluator
import json

rows = from_openeval(suite)
with open("rows.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

result = evaluate(data="rows.jsonl", evaluators={"f1": F1ScoreEvaluator()})
```

(The real, installed `azure-ai-evaluation` requires `evaluate(data=...)` to be a path or path-like object, not a raw list -- write `from_openeval()`'s output to a `.jsonl` file first, as shown above.)

### Export a completed `EvaluationResult` to an EvalPort ResultSet

```python
from azure_ai_evaluation_openeval_adapter import evaluation_result_to_openeval
from openeval.validate import validate_result_set

result = evaluate(data="rows.jsonl", evaluators={"f1": F1ScoreEvaluator()})
result_set = evaluation_result_to_openeval(result, suite_id="my_eval_suite")
assert validate_result_set(result_set).valid
```

## Grader mapping: every evaluator maps to `custom`, deliberately

Every `azure-ai-evaluation` evaluator -- `F1ScoreEvaluator`/`BleuScoreEvaluator`/`GleuScoreEvaluator`/`MeteorScoreEvaluator`/`RougeScoreEvaluator` (local NLP metrics), `RelevanceEvaluator`/`CoherenceEvaluator`/`GroundednessEvaluator`/etc. (AI-assisted, need a live `model_config`), the content-safety evaluators (`ViolenceEvaluator`, etc., need a live Azure AI Foundry project), and any custom class/function you pass in -- maps to EvalPort's `custom` grader type, with `params.handler` set to the evaluator's real class or function name.

This is deliberate, not a shortcut. EvalPort's `semantic_similarity` grader type requires `params.threshold` and implies an embedding/model-based method (`params.model`, `params.provider`); `llm_judge` requires a real `params.prompt`. The local NLP metrics (BLEU, ROUGE, METEOR, GLEU, F1) are lexical/n-gram overlap metrics, not embeddings -- mapping them to `semantic_similarity` would misrepresent how they compute a score. The AI-assisted and safety evaluators genuinely need live credentials (`model_config`, `azure_ai_project`) that this adapter doesn't have and can't safely fabricate a `prompt`/`threshold` for. `custom` with the evaluator's real class name is the only mapping that's honest across all three categories.

## Result parsing: recovering per-metric score/passed/reason from `evaluate()`'s flat row shape

`evaluate()`'s real `EvaluationResult` (confirmed directly against the installed package, not assumed from docs) returns `rows` as flat dicts like:

```python
{
    "inputs.query": "...", "inputs.response": "...", "inputs.ground_truth": "...",
    "outputs.f1.f1_score": 1.0, "outputs.f1.f1_score_score": 1.0,
    "outputs.f1.f1_score_passed": True, "outputs.f1.f1_score_result": "pass",
    "outputs.f1.f1_score_reason": None, "outputs.f1.f1_score_threshold": 0.5,
}
```

`evaluation_result_to_openeval()` groups `outputs.<evaluator_key>.*` columns by evaluator, recovers each metric's "base name" from the `<base>`/`<base>_score`/`<base>_passed`/`<base>_result`/`<base>_reason` suffix convention (real behavior of the SDK, including the `f1_score`/`f1_score_score` duplication you can see above), and builds one `GraderResult` per evaluator per row with the real score (clamped to EvalPort's required `[0, 1]`), `passed`, and `reason`. The full raw field group is preserved under `grader_result.metadata.azure_ai_evaluation` so nothing is lost even when the base-name heuristic doesn't apply cleanly (e.g. a plain custom evaluator returning just `{"score": True}`).

## What round-trips losslessly, and what doesn't

`to_openeval()` preserves the complete original row (every column, not just `query`/`response`/`ground_truth`) under `test_case.metadata.azure_ai_evaluation.row`, so `from_openeval()` restores it byte-for-byte on a round trip through EvalPort. For a suite built elsewhere (no prior round trip through this adapter), `from_openeval()` falls back to a heuristic mapping: `input` -> `query`, `expected_output` -> `ground_truth`.

What doesn't survive a round trip through a *different* EvalPort-speaking tool: an AI-assisted or safety evaluator's live configuration (`model_config`, `azure_ai_project`) can't be reconstructed from the outside -- those graders are captured as `custom` with the evaluator's real class name (for a faithful record of *what* ran), not as a runnable object.

## Testing

21 tests, all passing locally against the real `azure-ai-evaluation` package and the real `openeval.validate.validate_suite()`/`validate_result_set()` -- not mocks, including a full suite -> rows -> real `evaluate()` call -> ResultSet round trip. `BleuScoreEvaluator`/`GleuScoreEvaluator`/`MeteorScoreEvaluator`/`RougeScoreEvaluator` need NLTK's `punkt_tab`/`wordnet` corpora downloaded at runtime; where that wasn't available in the sandbox this was built in, tests use `F1ScoreEvaluator` (also a local NLP metric, no such dependency) plus plain custom evaluators instead -- grader-definition coverage for BLEU is still exercised (`to_openeval()` builds a correct grader for it) without requiring an actual scored run.

## Spec

<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>

## License

Apache-2.0 -- see [LICENSE](LICENSE)
