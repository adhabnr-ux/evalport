# arthur-bench-openeval-adapter

Convert [Arthur Bench](https://github.com/arthur-ai/bench) (`arthur-bench`) `TestSuite`s, `TestRun`s, and scoring results to/from [EvalPort](https://github.com/adhabnr-ux/evalport) (Apache 2.0), the open interchange format for portable LLM evaluation datasets -- test cases, graders, suites, and results as plain JSON, shared across DeepEval, Promptfoo, Inspect AI, AutoGen, CrewAI, Ragas, LangSmith, Braintrust, MLflow, and 20+ other frameworks.

## Install

```bash
pip install arthur-bench-openeval-adapter
```

## Usage

### Export a TestSuite to an EvalPort Suite

```python
from arthur_bench.run.testsuite import TestSuite
from arthur_bench_openeval_adapter import to_openeval
from openeval.validate import validate_suite

suite = TestSuite(
    name="capitals-suite",
    scoring_method="exact_match",
    input_text_list=["What is the capital of Japan?", "What is 2+2?"],
    reference_output_list=["Tokyo", "4"],
)

result = to_openeval(suite, suite_id="capitals-suite")
assert validate_suite(result).valid
```

### Load an EvalPort suite back as a TestSuite

```python
from arthur_bench_openeval_adapter import from_openeval

restored = from_openeval(result)  # built-in scorers resolve automatically
run = restored.run(run_name="my-run", candidate_output_list=["Tokyo", "4"], save=False)
```

A suite whose grader wraps a **custom** (non-built-in) `Scorer` subclass can't be resolved by name alone -- Arthur Bench itself refuses to reconstruct a custom scorer from a string (see the `UserValueError` it raises), and this adapter can't either. Pass the original scorer instance explicitly:

```python
restored = from_openeval(result, scorer=MyCustomScorer())
```

### Export a completed TestRun to an EvalPort ResultSet

```python
from arthur_bench_openeval_adapter import run_to_openeval
from openeval.validate import validate_result_set

run = suite.run(run_name="my-run", candidate_output_list=["Tokyo", "4"], save=False)
result_set = run_to_openeval(run, suite=suite, suite_id="capitals-suite")
assert validate_result_set(result_set).valid
```

`TestRun`/`TestCaseOutput` don't carry the scorer's identity themselves (just bare scores and categories), so `run_to_openeval()` needs either the original `suite` (typical usage: `suite = TestSuite(...); run = suite.run(...)`) or `scorer_name="exact_match"` directly to know which scorer produced the results and apply the right pass/fail logic below. Without either, results still convert (nothing is dropped) but under a generic `"scorer"` grader id and a numeric-only pass heuristic.

## Grader mapping: every scorer maps to `custom` -- including `exact_match`

Every Arthur Bench `Scorer` -- built-in (`exact_match`, `bertscore`, `readability`, `specificity`, `word_count_match`, `hedging_language`, `qa_correctness`, `summary_quality`, `hallucination`, `python_unit_testing`) and custom -- maps to EvalPort's `custom` grader type, with `params.handler` set to the scorer's real `name()` and `params.config` set to its `to_dict()` (Arthur Bench's own generic, JSON-serializable scorer-config representation, used unchanged so `from_openeval()` can reconstruct a matching scorer via `Scorer.from_dict()`).

For `exact_match` specifically this is **not** the obvious choice -- EvalPort has its own `exact_match` grader type, and Arthur Bench's `ExactMatch` scorer really is a literal string-equality check. But reading the installed package's source (`arthur_bench/scoring/exact_match.py`) turned up a real, confirmed quirk: `ExactMatch(case_sensitive=True)` -- the default used when `scoring_method="exact_match"` is passed as a plain string -- actually lowercases both sides before comparing (case-**in**sensitive); `case_sensitive=False` is what performs a true case-sensitive compare. That's inverted from what the parameter name implies. Mapping this to EvalPort's own `exact_match` grader type risks a case-differing output silently scoring differently if the suite is later re-run by a different, spec-conformant `exact_match` implementation. `custom`, with the real `case_sensitive` value captured in `params.config`, avoids that risk entirely and generalizes to every other scorer -- several of which (`hallucination`, `qa_correctness`, `summary_quality`) need a live Arthur-hosted API or an LLM judge this adapter has no honest way to fabricate credentials for.

## `passed`: derived from the real category, not just a numeric threshold

Several of Arthur Bench's built-in categorical scorers have **more than two** categories -- confirmed by reading each scorer's source, not assumed. `qa_correctness` returns `incorrect`/`correct`/`invalid`; `summary_quality` returns `reference`/`candidate`/`equal`/`invalid`. A blanket `score >= 0.5` threshold isn't a safe stand-in for these the way it is for a strictly binary scorer like `exact_match` or `python_unit_testing`. `run_to_openeval()` instead uses each result's real `category.name` against a small, verified table (`exact_match` -> `match`; `hallucination` -> `no hallucination`; `python_unit_testing` -> `pass`; `qa_correctness` -> `correct`; `summary_quality` -> `candidate`/`equal`) built by reading the installed package's category definitions.

For **continuous** (non-categorical) scorers without a category to key off of, `passed` falls back to `score >= 0.5` -- except `hedging_language`, which this adapter inverts (`score < 0.5`): its own docstring states "higher values corresponding to higher likelihood of hedging language being present," meaning higher is *worse*, not better, unlike every other continuous scorer checked here (`specificity`, `bertscore`, `word_count_match`, all confirmed higher-is-better).

## What round-trips losslessly, and what doesn't

`to_openeval()` preserves each test case's original `input`/`reference_output` under `test_case.metadata.arthur_bench`, so `from_openeval()` restores them exactly on a round trip through EvalPort, and preserves the suite's `description` the same way. For a suite built elsewhere (no prior round trip through this adapter), `from_openeval()` falls back to a heuristic mapping: `input` -> the TestSuite's input text, `expected_output` -> the reference output.

What doesn't survive a round trip through a *different* EvalPort-speaking tool: a custom `Scorer` subclass can't be reconstructed from the outside without the original class (`from_openeval(..., scorer=...)` is required for those), and any scorer needing a live Arthur-hosted API or LLM judge is captured as `custom` with its real name and config (a faithful record of *what* ran), not as a runnable object.

## Testing

18 tests, all passing locally against the real installed `arthur-bench` package and the real `openeval.validate.validate_suite()`/`validate_result_set()` validators -- not mocks, including a full suite -> restored suite -> real `TestSuite.run()` call -> ResultSet round trip. Tests use `LocalBenchClient(root_dir=<tmp_path>)` for isolation, since Arthur Bench's local client writes suite/run files to disk immediately on `TestSuite` construction (confirmed by reading `arthur_bench/client/local/client.py`) and looks up existing suites by name. `exact_match` and `word_count_match` are exercised with real, live scoring calls (both run fully offline); `readability`, `bertscore`, and `hedging_language` need NLTK corpora or transformer model downloads this sandbox's SSRF-protection proxy blocked at build time, so their grader-definition mapping is covered by `to_openeval()` without an actual scored run -- documented here rather than silently skipped.

## Spec

<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>

## License

Apache-2.0 -- see [LICENSE](LICENSE)
