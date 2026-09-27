# Cross-framework interop demos

Three runnable scripts that move eval data between real evaluation frameworks through
EvalPort and check, field by field, what survives. They use the real framework packages
(DeepEval, DSPy, Haystack), not mocks. They run offline and deterministically: no model is
called, and each script blocks sockets and DNS itself and prints how many connections were
attempted (always 0). CI runs them in the `interop-examples` job.

Each script exits non-zero if a ResultSet or suite fails validation, if a score or verdict
changes in transit, or if **any field changes that the script doesn't list as a known loss**.
Known losses are printed with the reason, not hidden. "Lossless" in this README means
byte-identical after the round trip.

| # | Script | Frameworks / adapters | What it shows |
|---|---|---|---|
| 1 | [`1_dataset_portability.py`](1_dataset_portability.py) | DeepEval `LLMTestCase`, DSPy `dspy.Example`, via `deepeval-openeval-adapter` and `dspy-openeval-adapter` | Suite → framework objects → suite, diffed field by field |
| 2 | [`2_results_portability.py`](2_results_portability.py) | DeepEval `evaluate()` → `haystack.evaluation.EvaluationRunResult`, via `deepeval-openeval-adapter` and `haystack-openeval-adapter` | Framework results → ResultSet → another framework → ResultSet; per-test-case scores agree |
| 3 | [`3_cross_framework_comparison.py`](3_cross_framework_comparison.py) | DeepEval, DSPy and Haystack, each with its own adapter | Three ResultSets for one suite, joined with `ResultSet.group` and compared per test case |

Tested versions (pinned in [`requirements.txt`](requirements.txt)): `deepeval==4.2.6`,
`dspy==3.4.0`, `haystack-ai==3.2.0`, Python 3.11. None of them pulls in torch. A fresh
venv installs in about 30–45 s, and the whole test file runs in about 18 s.

## Run it

From the repository root:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e sdk/python \
  -e adapters/deepeval-openeval-adapter \
  -e adapters/dspy-openeval-adapter \
  -e adapters/haystack-openeval-adapter \
  -r examples/interop/requirements.txt

python examples/interop/1_dataset_portability.py
python examples/interop/2_results_portability.py
python examples/interop/3_cross_framework_comparison.py

python -m pytest -p no:deepeval examples/interop/test_interop.py -v   # what CI runs
```

**Inputs.** The suites are real: a 10-case slice of [`benchmarks/gsm8k`](../../benchmarks/gsm8k/)
and [`benchmarks/truthfulqa`](../../benchmarks/truthfulqa/), plus
[`examples/rag-eval-suite.json`](../rag-eval-suite.json) and
[`examples/agent-tools.json`](../agent-tools.json), which add `context` and `expected_tools`
coverage. The model outputs scored in demos 2 and 3,
[`fixtures/gsm8k_outputs.json`](fixtures/gsm8k_outputs.json), are **hand-written and
synthetic**; no model produced them. They include deliberate formatting variants
(`"$70,000"`, `" 540\n"`, `"45."`) and one wrong answer.

## Demo 1: dataset portability

For each source suite, the script runs EvalPort → `from_openeval()` → real framework objects
(`LLMTestCase(**kwargs)`; `dspy.Example`) → `to_openeval()` → EvalPort. It validates the
result and diffs every Suite and TestCase field against the original. Real output, table
section (the script also prints one before/after example and the reason for each lossy field):

```
  field               via LLMTestCase         via dspy.Example
  suite.id            lossless (4 suites)     lossless (4 suites)
  suite.version       lossy 4/4 suites        lossy 4/4 suites
  suite.name          lossy 4/4 suites        lossy 4/4 suites
  suite.description   lossy 1/1 suites        lossy 1/1 suites
  suite.graders       lossy 4/4 suites        lossy 4/4 suites
  suite.config        lossy 2/2 suites        lossy 2/2 suites
  suite.metadata      lossy 4/4 suites        lossy 4/4 suites
  tc.id               lossless (24 cases)     lossless (24 cases)
  tc.input            lossless (24 cases)     lossy 24/24 cases
  tc.expected_output  lossless (23 cases)     lossless (23 cases)
  tc.context          lossless (1 cases)      lossy 1/1 cases
  tc.expected_tools   lossy 1/3 cases         lossy 3/3 cases
  tc.graders          lossy 24/24 cases       lossy 24/24 cases
  tc.metadata         lossy 24/24 cases       lossy 24/24 cases
...
network connections attempted: 0
deepeval: round-trips suite.id, tc.id, tc.input, tc.expected_output, tc.context
dspy: round-trips suite.id, tc.id, tc.expected_output
RESULT: PASS -- every round-tripped suite validates and every changed field is a documented lossy field (see 'why' above).
```

The test-case data (ids, prompts, references, context and non-empty tool lists) survives
DeepEval unchanged. Grader definitions and metadata don't survive either framework.

## Demo 2: results portability

1. Runs a real `deepeval.evaluate()` with DeepEval's deterministic `ExactMatchMetric` and
   gets back native `EvaluationResult`, `TestResult` and `MetricData` objects.
2. Converts them with `test_results_to_openeval()` into a ResultSet, validates it, then
   serializes it to JSON and parses it back.
3. Imports the ResultSet into Haystack's `EvaluationRunResult` and reads Haystack's own
   `aggregated_report()`.
4. Converts that with `evaluation_result_to_openeval()` into a second ResultSet, validates it,
   and compares all four representations per test case.

```
[1] deepeval 4.2.6: evaluate() -> EvaluationResult with 10 TestResult (metric 'Exact Match', threshold 1.0)
[2] test_results_to_openeval() -> ResultSet: 10 results, valid=True, 4393 bytes of JSON
[3] resultset_to_haystack() -> EvaluationRunResult; Haystack's own aggregated_report(): {'exact_match': 0.7}
[4] evaluation_result_to_openeval() -> ResultSet: 10 results, valid=True

  test case expected  actual_output DeepEval ResultSet Haystack ResultSet2  agree
  gsm8k_0         18  "18"               1.0       1.0      1.0        1.0  yes
  gsm8k_2      70000  "$70,000"          0.0       0.0      0.0        0.0  yes
  gsm8k_3        540  " 540\n"           1.0       1.0      1.0        1.0  yes
  gsm8k_6        260  "280"              0.0       0.0      0.0        0.0  yes
  ... (10/10 rows agree)
  pass rate: DeepEval ResultSet 0.70 | Haystack ResultSet 0.70 | Haystack aggregated_report 0.70

  Fields that did not survive DeepEval -> EvalPort -> Haystack -> EvalPort:
    lossy: completed_at  (1 value(s)) -- EvaluationRunResult has no timestamps
    lossy: metadata  (1 value(s)) -- ResultSet-level metadata ({'openeval': {'source': 'deepeval'}}) has no slot
    lossy: results[].grader_results[].metadata  (40 value(s)) -- threshold, strict_mode, evaluation_model, metric_name are replaced by Haystack's aggregate_score
    lossy: results[].grader_results[].reason  (10 value(s)) -- EvaluationRunResult stores only numeric individual_scores; MetricData.reason is dropped
    lossy: results[].metadata  (30 value(s)) -- per-result metadata (deepeval index/conversational/multimodal) has no slot
    lossy: runner  (2 value(s)) -- EvaluationRunResult has no runner/tool-version field
    lossy: summary.avg_score  (1 value(s)) -- the Haystack adapter's summary omits avg_score (optional field)
    lossy: summary.skipped  (1 value(s)) -- the Haystack adapter's summary omits skipped (optional field)

network connections attempted: 0
RESULT: PASS -- all 10 per-test-case scores, pass/fail verdicts and outputs agree across DeepEval, both ResultSets and Haystack; the only losses are listed above.
```

Step 3 uses `resultset_to_haystack()`, a 15-line bridge defined in the script. **No adapter in
this repo imports a ResultSet into a framework's results type** (see Findings).

## Demo 3: cross-framework comparison with `ResultSet.group`

The same 10 GSM8K cases and the same outputs are scored by each framework's own exact-match
metric through its real evaluation entry point:

- DeepEval: `evaluate()` with `ExactMatchMetric`
- DSPy: `dspy.Evaluate()` with `answer_exact_match`
- Haystack: `AnswerExactMatchEvaluator().run()`, wrapped in `EvaluationRunResult`

Each framework's result goes through its adapter into a ResultSet file with
`group: {group_id, role, label, sequence}`. A consumer that knows nothing about the three
frameworks loads the files, joins them on `group_id` + `test_case_id`, and builds the
comparison table. The script also checks that every score in each ResultSet equals the score
that framework produced natively.

```
  wrote gsm8k-deepeval.resultset.json      role=deepeval  valid=True  (DeepEval 4.2.6 ExactMatchMetric)
  wrote gsm8k-dspy.resultset.json          role=dspy      valid=True  (DSPy 3.4.0 answer_exact_match)
  wrote gsm8k-haystack.resultset.json      role=haystack  valid=True  (Haystack 3.2.0 AnswerExactMatchEvaluator)

  group_id=gsm8k-exact-match-3-frameworks: 3 members joined on test_case_id

  test case expected  output       deepeval      dspy  haystack  verdict
  gsm8k_0         18  "18"             pass      pass      pass  all pass
  gsm8k_1          3  "3"              pass      pass      pass  all pass
  gsm8k_2      70000  "$70,000"        fail      pass      fail  DISAGREE
  gsm8k_3        540  " 540\n"         pass      pass      fail  DISAGREE
  gsm8k_4         20  "20"             pass      pass      pass  all pass
  gsm8k_5         64  "64"             pass      pass      pass  all pass
  gsm8k_6        260  "280"            fail      fail      fail  all fail
  gsm8k_7        160  "160"            pass      pass      pass  all pass
  gsm8k_8         45  "45."            fail      pass      fail  DISAGREE
  gsm8k_9        460  "460"            pass      pass      pass  all pass
  pass rate                                    0.70      0.90      0.60

  3/10 test cases get different verdicts from different frameworks on identical outputs:
    gsm8k_2: "$70,000" vs "70000" -- passed by dspy only
    gsm8k_3: " 540\n" vs "540" -- passed by deepeval, dspy only
    gsm8k_8: "45." vs "45" -- passed by dspy only
  Why: DeepEval strips whitespace then compares; Haystack compares raw strings; DSPy lowercases and drops punctuation/articles first. The suite's own grader (exact_match {"ignore_case": true, "strip": true}) does not travel into any of them (demo 1), so each applies its own semantics.

network connections attempted: 0
RESULT: PASS -- 3 valid ResultSets joined via group.group_id; every score in them equals the framework's native score.
```

All three frameworks call their metric "exact match", yet they give pass rates of 0.60, 0.70
and 0.90 on identical outputs. The data moved without loss; the metric semantics differ.
Joining sibling ResultSets per test case is what makes that difference visible.

## Lossy fields for the adapters used here

The demos surfaced every entry below, and each script lists them in its `KNOWN_LOSSY` table.
Entries marked **undocumented** were not in the adapter's own README or docstring when these
demos were written.

| Adapter / direction | Field | What happens |
|---|---|---|
| deepeval: suite → `LLMTestCase` → suite | `graders` (suite and test case) | replaced by one placeholder `custom` grader, `gr_deepeval_metrics`, because DeepEval picks metrics at `evaluate()` time (documented) |
| | `TestCase.metadata` | dropped: `from_openeval()` never sets `LLMTestCase.metadata`, so e.g. TruthfulQA's `truthfulqa_all_correct` answer list is lost (**undocumented**) |
| | `expected_tools: []` | an empty list, meaning "no tool should be called", is dropped because `to_openeval()` skips falsy lists; the assertion becomes "no expectation". Non-empty lists round-trip (**undocumented**) |
| | `tools_called` / `expected_tools` | `from_openeval()` returns name strings, which `LLMTestCase` 4.2.6 rejects with `TypeError`; callers must wrap them in `ToolCall(name=...)` (documented as "construct `ToolCall` yourself") |
| | suite `name`, `description`, `config`, `metadata`, `version` | not representable on a list of `LLMTestCase`; the adapter stamps its own name and version |
| dspy: suite → `dspy.Example` → suite | `input` | a string input becomes the `input_1` field and comes back as `["input_1: <text>"]` (documented in the module docstring) |
| | `context`, `expected_tools` | dropped: `from_openeval()` maps only `input` and `expected_output` onto a foreign suite's Example fields |
| | `graders`, `metadata`, suite-level fields | placeholder `dspy_metric` grader; metadata replaced by `metadata.dspy` |
| deepeval → ResultSet → haystack → ResultSet | `grader_results[].reason` | dropped: `EvaluationRunResult` stores only numeric scores |
| | `grader_results[].metadata` | `threshold`, `strict_mode`, `evaluation_model` and `metric_name` are dropped |
| | `results[].metadata`, `runner`, `completed_at`, ResultSet `metadata`, `summary.avg_score`/`skipped` | no slot in `EvaluationRunResult`, or omitted by the Haystack adapter |
| | `run_id` | kept only if the caller passes `run_id=run.run_name`; the adapter doesn't read `run_name` itself |

## Findings: what isn't round-trippable

- **ResultSet import is one-way for every mainstream framework.** Every framework adapter's
  `from_openeval()` (DeepEval, DSPy, Haystack, Ragas, MLflow, Langfuse, Phoenix, Opik, Weave,
  Braintrust, LangSmith, TruLens and others) takes a Suite. The only ResultSet importers
  (luml, nasde, clawbench, nuguard, pyserini, daimax) target niche tools, and some of those
  can't be installed from PyPI. Demo 2 uses its own 15-line bridge for that reason.
- **Graders never travel into a framework.** Neither DeepEval nor DSPy has a slot for a grader
  definition, so the suite's `exact_match {ignore_case, strip}` is replaced by a placeholder.
  Demo 3 shows the consequence: three frameworks apply three different "exact match" rules.
- **The Ragas and MLflow adapters convert results to Suites, not ResultSets.** Scores go under
  `TestCase.metadata.*_scores`, so they couldn't be used for a ResultSet round trip here.
- **Import order:** importing `dspy` before `deepeval` in one process makes `deepeval` fail
  with a circular-import error in `openai._models`. That comes from dspy's lazy `openai`
  proxy, not EvalPort. The demos import deepeval first.
