# trulens-openeval-adapter

Convert TruLens (https://github.com/truera/trulens, published today as the
`trulens` / `trulens-core` packages by Snowflake) records and feedback
results to and from EvalPort (https://github.com/adhabnr-ux/evalport), the
open interchange format for portable LLM evaluation datasets.

## Why this maps to a ResultSet, not an EvalSuite

Most adapters in this repo convert an unexecuted suite of test cases (an
`EvalSuite`) into EvalPort's format, then run it. TruLens works the other way
around: it's an observability layer that instruments an already-built app,
so what you get back from a TruLens session is always already-executed
`Record`s plus their `FeedbackResult` scores. There's no TruLens-native
concept of a not-yet-run test suite to convert.

So `to_openeval()` here produces an EvalPort `ResultSet` (results that
already ran), and `from_openeval()` goes the other direction: it turns an
EvalPort `EvalSuite`'s test cases into a TruLens ground-truth golden set,
ready to hand to `trulens.feedback.GroundTruthAgreement` so an existing
EvalPort suite can drive TruLens's own agreement-based feedback functions.

## Install

```
pip install "trulens-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/trulens-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support (the same install path documented for every
other adapter in this repo).

## Usage

```python
from trulens.core.session import TruSession
from trulens_openeval_adapter import to_openeval

session = TruSession()
# However you already have your Records and FeedbackResults on hand --
# e.g. from `with tru_app as recording: ...` then `recording.records`,
# and each record's own `.feedback_results_as_completed` iterator.
records = [...]           # trulens.core.schema.record.Record instances
feedback_results = [...]  # trulens.core.schema.feedback.FeedbackResult instances

result_set = to_openeval(records, feedback_results, suite_id="my_rag_app", run_id="2026-09-19-eval")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

import json
with open("my_results.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

Feedback scores map one-to-one onto `GraderResult`s (`grader_id` = the
feedback function's `name`), scored against a configurable `pass_threshold`
(default `0.5`) since TruLens feedback functions don't carry a universal
pass/fail cutoff of their own:

```python
result_set = to_openeval(records, feedback_results, pass_threshold=0.7)
```

A record whose `main_error` is set (the wrapped app itself raised) becomes a
`Result.error` of type `"runner_error"` with `passed: false`, rather than
being scored — the same distinction EvalPort's `error` field already draws
between "no usable result was produced" and "produced and scored, but
failed."

Going the other direction — turning an existing EvalPort suite into ground
truth for a TruLens feedback function:

```python
from trulens.feedback import GroundTruthAgreement
from trulens_openeval_adapter import from_openeval

golden_set = from_openeval(my_evalport_suite)
# -> [{"query": "...", "expected_response": "..."}, ...]
ground_truth = GroundTruthAgreement(golden_set, provider=...)
```

## Field mapping

| TruLens | EvalPort |
|---|---|
| `Record.record_id` | `Result.test_case_id` |
| `Record.main_output` | `Result.actual_output` (JSON-serialized if not already a string) |
| `Record.main_error` | `Result.error` (`type: "runner_error"`) |
| `Record.main_input`, `.app_id`, `.tags` | `Result.metadata` |
| `FeedbackResult.name` | `GraderResult.grader_id` |
| `FeedbackResult.result` | `GraderResult.score` (preserved in `metadata.raw_result` instead of clipped, if outside EvalPort's `[0,1]` range) |
| `FeedbackResult.error` | `GraderResult.reason`, `score: null`, `passed: false` |
| `FeedbackResult.status` | `GraderResult.metadata.status` |

## Verification

Field names were checked against the real, installed `trulens-core==2.14.0`
source (`trulens.core.schema.record.Record`,
`trulens.core.schema.feedback.FeedbackResult`, and
`trulens.feedback.GroundTruthAgreement.agreement_measure`'s own docstring
example for the golden-set shape), not assumed from older `trulens_eval`-era
documentation. `tests/test_adapter.py` includes one test
(`test_field_names_match_real_trulens_classes`) that constructs the actual
TruLens classes rather than the lightweight stand-ins used elsewhere in the
suite, gated behind the `test-full` extra (`pip install ".[test-full]"`) so
the adapter itself stays free of a hard TruLens dependency.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 — see LICENSE.
