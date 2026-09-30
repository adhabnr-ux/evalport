# Examples

Sample EvalPort documents and runnable examples. Everything here is exercised in CI.

## Sample documents

- [`basic-suite.json`](basic-suite.json) and [`results.json`](results.json): the smallest valid Suite and ResultSet. The exporters below use them as their default inputs.
- [`multi-turn.json`](multi-turn.json), [`rag-eval-suite.json`](rag-eval-suite.json), [`safety.json`](safety.json), [`agent-tools.json`](agent-tools.json): suites for other shapes of evaluation.
- [`migrations/`](migrations/): the same suite as written for DeepEval and Promptfoo, converted to EvalPort.

## Runnable examples

| Directory | What it does | Status |
|---|---|---|
| [`interop/`](interop/) | Moves eval data between the real DeepEval, DSPy and Haystack packages through EvalPort and checks, field by field, what survives. | Maintained here, CI-tested. |
| [`croissant/`](croissant/) | One-way export of a Suite + ResultSet to MLCommons Croissant Tasks JSON-LD, validated with SHACL. | Maintained here, CI-tested. Not reviewed by MLCommons. |
| [`every-eval-ever/`](every-eval-ever/) | One-way export of a Suite + ResultSet to one [Every Eval Ever](https://github.com/evaleval/every_eval_ever) 0.3.0 aggregate record, validated against the unmodified upstream schema and, on Python 3.12+, EEE's own validator. | Unsolicited prototype, CI-tested. Not reviewed or accepted by the Every Eval Ever maintainers. |

The two exporters are lossy by construction. Each README has a "What is lost" section, and each states how `score: null` ("not verified", SPEC Validation Rule 6) is handled, because neither target format has a null score.
