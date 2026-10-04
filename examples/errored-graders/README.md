# When one grader of several errors: DeepEval and Inspect AI → EvalPort

> **Status: unsolicited example.** This is not an official DeepEval or Inspect AI
> integration, has not been proposed to or reviewed by either project, and is not an
> adapter. It runs the real [DeepEval](https://github.com/confident-ai/deepeval) (4.2.8) and
> [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) (0.3.276) evaluation code to
> see what each reports when a grader raises, and shows what EvalPort can and cannot say about
> the result.
>
> **The errors are constructed.** There is no LLM, network or API key. The "model" outputs are
> canned, and the second grader (`flaky`) is plain Python that raises on purpose, standing in
> for an LLM judge that could not be reached. Four rows were built to cover the cases; their
> proportions say nothing about how often real judges fail. What is real is the framework: its
> own code decides what a row looks like after a grader raised.

## The situation

An EvalPort `Result` can be graded by several graders. Validation Rule 6 gives a grader that
produced no score a name: `score: null`, `passed: false`, "not verified". It also says what
a Result whose graders are *all* null is: `passed: false`, "unscored". It says nothing about
the case in between, where some graders scored and one did not. That is the case here.

| Row | `exact` | `flaky` | What happened |
|---|---|---|---|
| q1 | passed | passed | both ran |
| q2 | passed | **raised** | one grader never produced a score; the other passed |
| q3 | failed | **raised** | one verified failure; one grader never produced a score |
| q4 | **raised** | **raised** | nothing produced a score |

## What the two frameworks do (real runs, pinned versions)

**DeepEval 4.2.8**, `ErrorConfig(ignore_errors=True)`:

- an errored metric gets `score=None`, `success=False` and its error text; the metrics that ran
  keep their scores;
- the row's `success` is `False` for q2, q3 and q4 (fail-closed). Only q1 passes, so the pass rate is 0.25
  and DeepEval's console summary says "Passed: 1, Failed: 3";
- without `ignore_errors` (its default), the exception escapes `evaluate()` and no results exist.

**Inspect AI 0.3.276**, `fail_on_error=False`:

- the exception is recorded on the **sample** (`sample.error`), not on a scorer, and the scores
  that were produced stay in the log;
- there is no row-level pass flag. The run status is `success`, `completed_samples` is 1 of 4, and
  each scorer's accuracy is computed over only the samples that scorer scored (`exact` 0.667
  over three samples, `flaky` 1.0 over one);
- with the default `fail_on_error=True`, the run status is `error` and there are no aggregate scores;
- **scorer order matters.** With `[exact, flaky]`, q2 keeps `exact`'s passing score. With
  `[flaky, exact]`, q2 and q3 have **no score from either scorer**, so q3's failure is not in the
  log at all. The log cannot say whether the later scorer was never attempted or was attempted
  and discarded; only the absence is observed.

**Judgeval 1.3.3: read, not run.** From its source (`judgeval/offline_tests/offline_test_runner.py`
and `types.py`): a scorer result carries an `error` field, and the row-level `success` is whatever
a user-supplied `pass_condition_fn(data, scorers_data)` returns; with no such function it is `None`
and `OfflineTestResult.passed` is `None`. So the framework leaves the decision to the caller. I
did not run it, and nothing here claims how its hosted judges behave.

## Writing it as EvalPort

`errored_graders_to_evalport.py` writes a Suite and a ResultSet per framework (`sample_output/`).
An errored grader is `score: null`, `passed: false`; the DeepEval error text is its `reason`
(Inspect's error is on the sample, so it becomes `Result.error`, type `runner_error`). The only
choice that matters is how a row with a mix is aggregated into `Result.passed`:

- **`spec-default`**: the AND of the non-null graders. This is what SPEC.md says (Rule 6, and the
  Aggregation Extension's default `all`: null-scored results are "excluded from the aggregation
  entirely").
- **`fail-closed`**: false unless every grader produced a passing score. The Python SDK validator accepts
  it (the TypeScript one was not run). SPEC.md's prose does not describe it, and no `openeval.aggregation`
  strategy means this.

Both validate. They differ in exactly one row:

| Row | graders | DeepEval's own row flag | `spec-default` | `fail-closed` | What Rule 6 lets a reader conclude (`spec-default`) | What `Result.verdict` (#49) would say |
|---|---|---|---|---|---|---|
| q1 | 1, 1 | passed | passed | passed | passed | passed |
| q2 | 1, null | **failed** | **passed** | failed | passed | **unverified** |
| q3 | 0, null | failed | failed | failed | verified failing | failed |
| q4 | null, null | failed | failed | failed | not verified | unverified |

| Summary over the four rows | `spec-default` | `fail-closed` |
|---|---|---|
| passed / failed / skipped (all-null) | 2 / 1 / 1 | 1 / 2 / 1 |
| `pass_rate` | **0.5** | **0.25** |

Same data, both valid, a factor of two apart. For DeepEval the `fail-closed` writing is the
framework's own row flag (asserted in the tests); the `spec-default` writing is what the spec's text
produces, and it turns "one grader never ran" into a pass.

## What this does and does not show about Discussion #49 (`Result.verdict`)

`--proposed-verdict` adds the `verdict` field proposed in
[Discussion #49](https://github.com/adhabnr-ux/evalport/discussions/49) (draft PR #83). **It is not in
the spec**; the flag tests the proposal against real behavior. The verdict here is this example's
reading of a row, not a spec rule: `failed` if any grader that ran failed, `unverified` if none
failed but one produced no score, `passed` if all ran and passed.

For:

- **q2 is the case Rule 6 cannot express.** Rule 6 names "all graders null". Row q2 is neither
  "passed" nor "verified failing", and no value in the spec's current vocabulary says so. This is the
  "scored but not established" case the RFC's evidence was missing. (The Langroid example in
  `examples/langroid/` had no such row; there every unverified row was all-null.)
- Two real frameworks, two behaviors, with the spec's default matching neither on q2.

Against, or at least not shown:

- The errors are constructed (see the note at the top), and one grader that sometimes cannot run
  is not the same as measured judge-failure rates.
- A `verdict` field is not the only way to say it. A metadata convention, a second
  `aggregation_status` value for "some graders unscored", also lets a producer mark q2 without a
  schema change. `--mark-partial` tests that alternative; see the next section.
- Inspect AI's logs do not say which scorer raised, so a converter cannot always attribute the error
  to a grader; it can only see which graders have no score.

## The alternative: a `"partial"` marker in metadata (`--mark-partial`)

Discussion #49 names its own competing option: no new field, just Rule 6 per grader plus a
`metadata.openeval.*` key. The spec already has `metadata.openeval.aggregation_status: "unscored"`
for an all-null row. `--mark-partial` extends it with `"partial"` on a row where some, but not all,
graders are null. **`"partial"` is not in the spec either.** `convention_class()` is what a consumer
that knows the convention can conclude from `passed` + `aggregation_status` alone.

| row | graders | `spec-default` + marker | convention reads | `fail-closed` + marker | convention reads | in fact |
|---|---|---|---|---|---|---|
| q1 | 1, 1 | `passed: true` | passed | `passed: true` | passed | passed |
| q2 | 1, null | `passed: true`, `partial` | **unverified** | `passed: false`, `partial` | **failed** | unverified |
| q3 | 0, null | `passed: false`, `partial` | failed | `passed: false`, `partial` | failed | failed |
| q4 | null, null | `passed: false`, `unscored` | unverified | `passed: false`, `unscored` | unverified | unverified |

What the marker shows (all asserted in `TestPartialMarkerConvention`):

- **It works, on these rows, with no schema or validator change.** With the spec's default
  aggregation the convention recovers exactly the classification `verdict` would give on all four
  rows. Both SDK validators on `main` accept the marked documents (the Python one in the tests, the
  TypeScript one run by hand on the same files), and the marker changes nothing but `metadata`.
- **Its reading depends on an aggregation policy the document does not declare.** Under
  `fail-closed`, q2 is `passed: false` + `partial`, which the convention has to read as a verified
  failure. Nothing in the document says which policy produced `passed` (`fail-closed` is not an
  `openeval.aggregation` strategy), so a consumer cannot know which reading applies. A
  producer-asserted `verdict` does not have this problem; it is the one thing the field buys that the
  convention cannot.
- **The unverified row keeps `passed: true`.** Every consumer that reads only `passed` (all of them,
  today) counts q2 as a pass. #49 forbids exactly that combination, which is why its validator rejects
  the `spec-default` writing (point 1 above). The marker and the field are not exclusive: a producer can
  write both, and the opt-in #83 tests check that the marked `fail-closed` document is still accepted.
- **Not shown here:** a row where a grader is null because nothing was sought (AgentEval's
  `NotApplicable`, e.g. no ground truth to compare against) would also be marked `partial` + `passed:
  true` and read as unverified, although the framework calls it a pass. The convention would need
  the per-grader reason too, and so would `verdict`. See the AgentEval adapter's q5 for that row.

What the PR #83 validator does with these documents (opt-in tests, below), which the RFC text does not address:

1. **`unverified` requires `passed: false`, but the spec default makes q2 `passed: true`.** The
   `spec-default` documents with `--proposed-verdict` are rejected (`VERDICT_PASSED_MISMATCH` on q2).
   A producer that adds `verdict` must also abandon the spec's default aggregation for such rows, and
   the RFC does not say so.
2. **`failed` is rejected on a row with `error`** (`VERDICT_ERROR_CONFLICT`). Inspect's q3 has a verified
   failing grader and a sample error; this example calls it `failed`, and the proposal refuses that.
   The producer must write `unverified` (losing an established failure) or omit `verdict`. This is the
   concrete case for the RFC's own open question "should `error` plus `failed` be allowed?".

## What is lost

Only the final answer text, per-grader score and pass, DeepEval's error text, and Inspect's sample
error are written. Messages, traces, model calls and costs are not. One grader type
(`exact_match`) stands in for both graders in the Suite; the difference between them is that one
raises.

## Run it

```bash
pip install -e ../../sdk/python -r requirements.txt        # from this directory
python errored_graders_to_evalport.py --out-dir out         # out/<framework>/suite.json, results.json
python errored_graders_to_evalport.py --out-dir out --policy fail-closed
python errored_graders_to_evalport.py --out-dir out --proposed-verdict --policy fail-closed
python errored_graders_to_evalport.py --out-dir out --mark-partial   # the metadata-convention alternative
python observe.py                                           # just the raw observations
python -m pytest -p no:cacheprovider .                      # 75 tests; 82 with the RFC #49 branch
```

`--deterministic` fixes `started_at` and omits `completed_at`; the checked-in
[`sample_output/`](sample_output/) is that output and a test fails if it goes stale. To run the opt-in
`verdict` checks:

```bash
git fetch origin pull/83/head:rfc49 && git worktree add ../rfc49-checkout rfc49
EVALPORT_RFC49_CHECKOUT=$PWD/../rfc49-checkout python -m pytest -p no:cacheprovider .
```

The framework behaviors above were observed with the pinned versions; the first test class fails
if a newer release changes them, which is the signal to re-check this page.

The same four-row shape was later run through AgentEval (.NET) with its real `CompositeEval`; see
[`adapters/agenteval-dotnet-openeval-adapter/`](../../adapters/agenteval-dotnet-openeval-adapter/#what-agenteval-does-when-one-of-two-graders-cannot-measure-a-case).
