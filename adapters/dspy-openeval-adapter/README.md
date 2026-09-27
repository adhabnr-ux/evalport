# dspy-openeval-adapter

Convert [DSPy](https://github.com/stanfordnlp/dspy) devsets and `dspy.Evaluate` results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install "dspy-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/dspy-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## Usage

### Export a devset to an EvalPort suite

```python
import dspy
from dspy_openeval_adapter import to_openeval

devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
]

suite = to_openeval(devset, input_keys=["question"], expected_key="answer")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("my_suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

`input_keys` names which `Example` field(s) become `TestCase.input` (EvalPort has no named-field concept, so multiple keys are flattened into `["key: value", ...]` — one string per key). `expected_key` names the one field that becomes `expected_output`. On a round trip through this adapter, nothing is lost: every original field is additionally preserved under `test_case.metadata.dspy.fields`.

### Import an EvalPort suite as a devset, ready to run

```python
from dspy_openeval_adapter import from_openeval

devset = from_openeval(suite)  # -> list[dspy.Example], input keys already marked

evaluator = dspy.Evaluate(devset=devset, metric=my_metric, display_progress=True)
result = evaluator(my_program)
```

A suite built by this adapter round-trips its exact original `Example` fields. A hand-authored suite (or one from a different EvalPort-speaking tool) is laid out like this:

| EvalPort `TestCase` field | `dspy.Example` | Input or label |
|---|---|---|
| `input` (string, or each array entry) | `input_1`, `input_2`, ... (or your `input_keys=[...]`) | input |
| `expected_output` | `expected_output` (or your `expected_key=`) | label |
| `context`, `retrieval_context` | fields of the same name | input (a RAG program reads them) |
| `expected_tools` | field of the same name; `[]` stays `[]` ("no tool should be called") | label |
| `metadata`, `tags`, `tools_called` | kept on the example, outside `inputs()`/`labels()` | — |

Rename the `context` / `retrieval_context` / `expected_tools` fields to match your signature with `field_names={"context": "passages"}`. Each example remembers this layout, so `to_openeval(devset)` (no `input_keys` needed) writes every value back to its native field: a string `input` comes back as the same string, not `["input_1: ..."]`. The layout is stored as a private attribute, which `Example.copy()`/`with_inputs()` don't carry over; after those, pass `input_keys` and get the usual flattening.

### Export evaluation results to an EvalPort ResultSet

```python
from dspy_openeval_adapter import evaluation_result_to_openeval

result = evaluator(my_program)  # a dspy.EvaluationResult
result_set = evaluation_result_to_openeval(result, suite_id=suite["id"], metric=my_metric)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

A DSPy metric is arbitrary Python — it may return `bool` (the common case), a plain number (rarely outside `[0, 1]`), or a `dspy.Prediction(score=..., feedback=...)` for GEPA-style feedback-augmented metrics. All three are handled: bools map directly, `Prediction.score`/`feedback` map onto the grader result's `score`/`reason`, and any other numeric score is clamped into EvalPort's required `[0, 1]` range with the original value preserved in `grader_result.metadata.dspy.raw_score` whenever clamping changed it.

### The full loop

```python
suite = to_openeval(devset, input_keys=["question"], expected_key="answer", ids=["fr", "jp"])
devset2 = from_openeval(suite)
result = dspy.Evaluate(devset=devset2, metric=my_metric, display_progress=False)(my_program)
result_set = evaluation_result_to_openeval(result, suite_id=suite["id"])
# result_set["results"][i]["test_case_id"] == suite["test_cases"][i]["id"], preserved end to end
```

## Why the metric itself isn't a real grader

EvalPort has no way to serialize an arbitrary Python callable, and a DSPy metric function is exactly that — there's no portable, re-executable representation to give it. `to_openeval()` therefore emits one placeholder `custom` grader per suite that documents "the caller must supply a metric at run time," rather than fabricating a fake grader implementation. This is the same honest limitation every adapter in this ecosystem takes for framework-specific logic with no native EvalPort shape (see [vertexai-openeval-adapter](../vertexai-openeval-adapter)'s `CustomMetric` handling, or [crewai-openeval-adapter](../crewai-openeval-adapter)'s tool-selection grader, for the same pattern).

## Running a suite's own graders in DSPy (opt-in): `graders_to_dspy_metrics()`

The other direction does have a faithful mapping for one grader type. `graders_to_dspy_metrics(suite)` returns `{grader_id: metric}`, each a DSPy metric `metric(example, pred, trace=None) -> bool` whose `__name__` is the grader id:

```python
from dspy_openeval_adapter import graders_to_dspy_metrics

metric = graders_to_dspy_metrics(suite, output_key="answer")["gr_exact_match"]
result = dspy.Evaluate(devset=from_openeval(suite), metric=metric)(my_program)
result_set = evaluation_result_to_openeval(result, suite_id=suite["id"], metric=metric)
# grader_results[].grader_id == "gr_exact_match", matching the suite
```

`expected_key` (default `"expected_output"`) names the Example field with the reference; `output_key` names the Prediction field with the output (optional when the prediction has one field).

| EvalPort grader | DSPy metric | Faithful? |
|---|---|---|
| `exact_match` (`ignore_case`, `trim_whitespace`) | generated metric function | yes: trims when `trim_whitespace` (default true), lowercases when `ignore_case` (default false), then `==`, the same steps as the reference runner's `gradeExactMatch` (`cli/src/run/graders/tier1.ts`) |
| `semantic_similarity` | — | no: the score depends on the embedding model the runner picks, which the grader doesn't pin down |
| `llm_judge` / `model graded` | — | no: running the grader's prompt needs a judge LM, and `dspy.evaluate`'s LM metrics (`SemanticF1`, `CompleteAndGrounded`) use their own prompts |
| `contains`, `regex`, `json_schema`, `json_path`, `code`, `human`, `custom` | — | not mapped: no `dspy.evaluate` built-in metric computes exactly these |

Unsupported types raise `ValueError` with the reason, or are left out with `skip_unsupported=True`.

Why not DSPy's own `answer_exact_match`: it lowercases and removes punctuation and articles whatever the params say, so it passes `"45."` against `"45"` and `"$70,000"` against `"70000"`, which EvalPort's `exact_match` fails ([`examples/interop`](../../examples/interop) demo 3 shows this on GSM8K).

Details: Python's `str.strip()`/`str.lower()` stand in for JavaScript's `trim()`/`toLowerCase()` (their whitespace sets differ on exactly six code points: U+FEFF is trimmed by JS only, U+001C–U+001F and U+0085 by Python only); a missing reference compares as `""`, as in the reference runner; params the spec doesn't define (e.g. `strip`, used by `benchmarks/gsm8k`) raise a `UserWarning` and are ignored, as the reference runner ignores them. `exact_match()` (the comparison itself) is exported too.

## What round-trips losslessly, and what doesn't

DSPy → EvalPort → DSPy (via this adapter both ways): lossless — every `Example` field and its input/label marking survives exactly, restored from `metadata.dspy.fields`.

EvalPort → DSPy → EvalPort (a suite from another tool, through `from_openeval()` then `to_openeval()`): every `TestCase` data field comes back byte-identical — `id`, `input` in its original string/array form, `expected_output`, `context`, `retrieval_context`, `tools_called`, `expected_tools` (including `[]`), `tags`, `metadata` — with no `metadata.dspy` added. If you add fields to an example or change its input marking in between, `metadata.dspy` is added (next to the original metadata) so those changes survive too. Not carried: `graders` (placeholder, see above; use `graders_to_dspy_metrics()` to run `exact_match` graders) and suite-level `name`, `description`, `config`, `metadata`. [`examples/interop`](../../examples/interop) demo 1 checks this against real benchmark suites.

DSPy → EvalPort → some other tool: the flattened `"key: value"` input strings and the single `expected_output` value are readable by any EvalPort consumer, but a different tool has no way to know which field was a DSPy signature's actual input versus free-form context — the same tradeoff every adapter here takes for structure that doesn't have a native EvalPort field.

## Changelog

### 0.2.0

Fixes found by the cross-framework demos in `examples/interop/`:

- A foreign suite's string `input` now round-trips as the same string. Before, it became field `input_1` and came back as `["input_1: <text>"]`.
- `context` and `retrieval_context` (inputs) and `expected_tools` (a label; `[]` kept) become Example fields and are written back. Before, `from_openeval()` dropped them.
- `metadata`, `tags` and `tools_called` of a foreign test case come back unchanged. Before, `metadata` was replaced by `metadata.dspy`.
- `to_openeval()`: `input_keys` is optional for examples from `from_openeval()`, and when `ids` is omitted those examples keep their original test case ids (before: `dspy_tc_<n>`).
- **Behavior change:** foreign-suite examples gain `context` / `retrieval_context` / `expected_tools` fields, so their `inputs()` / `labels()` include them. Rename them with `field_names=`, or drop them with `example.without(...)`, if your program or metric doesn't expect them.
- New opt-in `graders_to_dspy_metrics()` and `exact_match()` (see above).

### 0.1.0

Initial release.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
