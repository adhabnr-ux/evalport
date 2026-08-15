# llamaindex-openeval-adapter

Convert [LlamaIndex](https://github.com/run-llama/llama_index) (`llama_index.core.evaluation`) evaluators, evaluation inputs, and `BatchEvalRunner` results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install llamaindex-openeval-adapter
```

This installs `evalport-sdk` as its only hard dependency. `llama-index-core` itself is **not** installed automatically -- add it separately (`pip install llama-index-core`) alongside whichever LLM/embedding integrations your evaluators need (e.g. `llama-index-llms-openai`).

## Usage

### Evaluation inputs → EvalPort suite, and back

```python
from llama_index.core.evaluation import FaithfulnessEvaluator, SemanticSimilarityEvaluator
from llamaindex_openeval_adapter import to_openeval, from_openeval

eval_suite = to_openeval(
    queries=["What is the capital of France?"],
    evaluators={
        "faithful": FaithfulnessEvaluator(),
        "sim": SemanticSimilarityEvaluator(similarity_threshold=0.85),
    },
    references=["Paris is the capital of France."],
    contexts_list=[["Paris is the capital of France, and its largest city."]],
    suite_id="geo_facts",
)

from openeval.validate import validate_suite
assert validate_suite(eval_suite).valid

# ...and back: rebuild everything needed to run a real BatchEvalRunner.
rebuilt = from_openeval(eval_suite)
from llama_index.core.evaluation import BatchEvalRunner

runner = BatchEvalRunner(evaluators=rebuilt["evaluators"], workers=4)
my_app_responses = [my_app(q) for q in rebuilt["queries"]]  # run your own system under test
results = await runner.aevaluate_response_strs(
    queries=rebuilt["queries"],
    response_strs=my_app_responses,
    contexts_list=rebuilt["contexts_list"],
    reference=rebuilt["references"],
)
```

`to_openeval()` takes exactly the shape `BatchEvalRunner(evaluators).aevaluate_response_strs(queries=..., contexts_list=..., reference=...)` itself takes -- everything needed to *define* an evaluation run, deliberately excluding the responses being graded, since those don't exist yet at suite-definition time (they come from whatever system you're testing, and are captured afterward -- see below). Each entry in the `evaluators` dict becomes one EvalPort grader, applied to every test case, matching how `BatchEvalRunner` itself runs every evaluator against every query uniformly.

### BatchEvalRunner results → EvalPort ResultSet

```python
from llamaindex_openeval_adapter import batch_eval_result_to_openeval

result_set = batch_eval_result_to_openeval(
    results,                          # the Dict[str, List[EvaluationResult]] BatchEvalRunner returned
    test_case_ids=rebuilt["ids"],     # align results back to test cases by id
    evaluators=rebuilt["evaluators"], # same mapping the runner was built with
    response_strs=my_app_responses,
    suite_id="geo_facts",
    run_id="run-2026-08-15",
    started_at="2026-08-15T00:00:00Z",
)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

`evaluators` is required here, not optional: an `EvaluationResult` carries no record of which evaluator class produced it, so `batch_eval_result_to_openeval()` needs the same `{name: BaseEvaluator}` mapping the run used in order to know how to normalize each grader's score and which EvalPort grader `type` it corresponds to (see below).

## Grader mapping

| LlamaIndex evaluator | EvalPort grader `type` | Direction |
|---|---|---|
| `SemanticSimilarityEvaluator` | `semantic_similarity` (`params.threshold` = `similarity_threshold`) | both |
| `FaithfulnessEvaluator` / `ResponseEvaluator` | `llm_judge` | both |
| `RelevancyEvaluator` / `QueryResponseEvaluator` | `llm_judge` | both |
| `AnswerRelevancyEvaluator` | `llm_judge` | both |
| `ContextRelevancyEvaluator` | `llm_judge` | both |
| `CorrectnessEvaluator` | `llm_judge` (score rescaled, see below) | both |
| `GuidelineEvaluator` | `llm_judge` (its own `guidelines` text becomes the rubric) | both |
| `PairwiseComparisonEvaluator` | `custom` (`params.handler` names the class) | export only |
| any other/unrecognized evaluator class | `custom` | export only |
| `code`, `human`, `"model graded"`, unrecognized `custom` handlers | -- | clean-skipped on import |

`SemanticSimilarityEvaluator` is the one deterministic (embedding-based, not LLM-judged) evaluator in `llama_index.core.evaluation`; everything else in the public API is an LLM-as-judge evaluator of a single response, which is why they all map onto EvalPort's single `llm_judge` grader type -- distinguished from each other by the rubric text baked into `params.prompt` and by the full evaluator config preserved verbatim under `grader.metadata.llama_index` (used to reconstruct the *exact* evaluator class and its config on import, not just "some llm_judge").

`PairwiseComparisonEvaluator` judges two candidate responses against each other rather than one response against a query/context/reference -- there's no EvalPort grader shape for a two-response comparison, so (matching the `giskard-openeval-adapter`'s handling of its own opaque check kinds) it's exported as `"custom"` with its full config preserved, and deliberately **not** reconstructed on import even when `grader.metadata.llama_index.class` says `PairwiseComparisonEvaluator` -- a real bug this adapter's own test suite caught: the first version reconstructed it anyway, which would have crashed the first time anyone tried to run it as a single-response evaluator.

## What round-trips losslessly, and what doesn't

Evaluator configuration that maps onto a grader's own fields (a similarity threshold, a correctness score threshold, `GuidelineEvaluator`'s guidelines text) round-trips exactly -- `from_openeval()` reconstructs the exact evaluator class with the exact config, not a generic stand-in, whenever the grader carries this adapter's own `metadata.llama_index` (`openeval.*` is the only reserved metadata prefix; everything else preserved this way follows the same convention every adapter in this repo uses).

Two spots are honestly approximate:

- **`llm_judge` prompts and the `model` field.** Most llama_index evaluators (`FaithfulnessEvaluator`, `RelevancyEvaluator`, etc.) don't expose a single literal prompt string -- `FaithfulnessEvaluator` in particular runs its check by building a query engine over the context and querying it, not by filling in one template. `to_openeval()` synthesizes an honest, human-readable description of what each evaluator actually checks (see `_LLM_JUDGE_RUBRICS` in the source) rather than fabricating a fake extracted prompt, and always includes the `{output}`/`{input}` (and, for `CorrectnessEvaluator`, `{expected}`) tokens EvalPort's real validator requires `llm_judge` prompts to contain. Similarly, llama_index evaluators resolve their LLM from `Settings.llm` (or a per-evaluator override) at call time rather than storing a model id on the evaluator itself, so `params.model` is filled with the placeholder `"llamaindex-configured-llm"` -- the same gap, and the same honest handling, documented in the [Giskard adapter](../giskard-openeval-adapter#what-round-trips-losslessly-and-what-doesnt) for `giskard-checks`' own `LLMJudge`.
- **`CorrectnessEvaluator`'s score scale.** It scores on a 1-5 scale (default `score_threshold=4.0`), not EvalPort's required `[0, 1]` range. `batch_eval_result_to_openeval()` rescales it as `(score - 1) / 4`; the untouched original 1-5 score is always preserved under `grader_result.metadata.llama_index.raw_score` so nothing is lost, just normalized for spec compliance.

A hand-authored grader with no `metadata.llama_index` at all (e.g. one written directly as JSON, or produced by a different adapter) still gets a working evaluator back on import -- `semantic_similarity` reconstructs a plain `SemanticSimilarityEvaluator` from `params.threshold`, and `llm_judge` reconstructs a `GuidelineEvaluator` using `params.prompt` as its guidelines text, the most generic "judge against a text rubric" evaluator llama_index has.

## Running the tests

```bash
cd adapters/llamaindex-openeval-adapter
python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest tests/ -v
```

36 tests, all running against the real `llama_index.core.evaluation` classes (`FaithfulnessEvaluator`, `CorrectnessEvaluator`, `SemanticSimilarityEvaluator`, `GuidelineEvaluator`, `RelevancyEvaluator`, `AnswerRelevancyEvaluator`, `PairwiseComparisonEvaluator`, `BatchEvalRunner`, `EvaluationResult`) and the real `openeval.validate.validate_suite()`/`validate_result_set()` -- not mocks of this adapter's dependencies. Constructing an evaluator or actually running `BatchEvalRunner` needs an LLM/embedding model; the tests configure llama_index's own first-party offline test doubles (`MockLLM`, `MockEmbedding`) via the real `Settings` singleton -- the same mechanism llama_index's own test suite and its users reach for to run evaluators without an API key, not a mock this adapter invented. One test (`TestEndToEndWithRealBatchEvalRunner`) drives the entire pipeline end to end: `to_openeval()` → `from_openeval()` → a real, live `await BatchEvalRunner(...).aevaluate_response_strs(...)` call → `batch_eval_result_to_openeval()` → `validate_result_set()`.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 -- see [LICENSE](LICENSE).
