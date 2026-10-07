# How a Result's outcome is represented: what `main` says and what the open RFCs would add

> **Not normative.** This page is a reading aid for two open RFCs, written so their
> interaction can be checked in one place. Where it describes a proposal it says so. If it
> disagrees with [`spec/SPEC.md`](../../spec/SPEC.md), the spec wins and this page is wrong.
> Nothing here has landed except the rows marked "on `main`".

A `Result` has one required boolean, `passed`. Evaluation runs end in more situations than
a boolean can name. This page lists six, says how each is written today, and what
[Discussion #49](https://github.com/adhabnr-ux/evalport/discussions/49) (`Result.verdict`)
and [Discussion #47](https://github.com/adhabnr-ux/evalport/discussions/47)
(`Result.constraint_violations`) would add. Both are open; neither is on `main`.

## The six situations

| Situation | On `main` today | If #49 lands (`verdict`) | If #47 lands (`constraint_violations`) | Result-level denominator |
|---|---|---|---|---|
| **Passed**: graders scored it and it met the bar. | `passed: true`; `score` values in [0, 1]. | `verdict: "passed"` (optional). Requires `passed: true`. | no change | Counted. |
| **Failed (verified)**: graders ran and it did not meet the bar. | `passed: false`; at least one non-null `score` (Rule 6: "verified failing"). | `verdict: "failed"` (optional). Requires `passed: false`. | no change | Counted. |
| **Not verified**: no grader established an outcome (a pending `human` review, an unsupported grader type, a runner error before scoring, a stalled agent). | Every `GraderResult` has `score: null` and `passed: false` (Rule 6); the `Result` is `passed: false`, and `metadata.openeval.aggregation_status: "unscored"` is RECOMMENDED. Null scores are left out of aggregation, not counted as failures. | `verdict: "unverified"`: the producer's explicit statement that the row stays in result-level denominators and is reported as unverified. Requires `passed: false`. | no change | Rule 6 says consumers MUST NOT treat a null as a scored failure and may exclude it from the aggregate denominator or report it separately. #49 would add: an `unverified` row MUST NOT be dropped from `summary.total`. |
| **Partly verified**: some graders produced a score and at least one did not (a judge outage on one of two graders). | The grader that did not run is `score: null`, `passed: false`; the others keep their scores. Under the default aggregation (null-scored graders excluded) `Result.passed` is the AND of the *scored* graders, so the row is `passed: true` when every scored grader passed. Nothing on the row says one grader never ran. | `verdict: "unverified"` if no scored grader failed. It requires `passed: false`, which differs from that default (`VERDICT_PASSED_MISMATCH` on the #83 branch). If a scored grader failed: `"failed"`, but not if `error` is also present (`VERDICT_ERROR_CONFLICT`). | no change | Counted: a scored grader exists. Whether it counts as a pass is the open aggregation question below. |
| **Error**: the harness produced no usable result (`timeout`, `provider_error`, `runner_error`). | `Result.error` object; `passed: false`. The spec text on `main` does not say whether such a row is in a denominator; the RFC texts describe consumers as dropping error rows. | `verdict` must be absent or `"unverified"` when `error` is present (`VERDICT_ERROR_CONFLICT`). Setting `"unverified"` on an error row says it stays in the denominator. | **`error` wins** over `constraint_violations`: a row with both is read as errored. | Consumer's choice today; with `verdict: "unverified"`, counted. |
| **Hard constraint violated**: the output may score well, but broke a rule that must fail the row outright (AOBench's RBAC hard fail is the motivating case). | No first-class representation. | Recommended `verdict: "failed"`: the violation is itself an established judgment. | `constraint_violations[]` with `invalidates_result: true` forces `passed: false` (`CONSTRAINT_INVALIDATES_PASS`) and keeps the row in denominators. A non-invalidating entry changes nothing. | Counted. |

Where a cell says "would", the behavior is in a draft reference PR
([#83](https://github.com/adhabnr-ux/evalport/pull/83) for `verdict`,
[#48](https://github.com/adhabnr-ux/evalport/pull/48) for `constraint_violations`), not on
`main`. The draft PRs are for review and are not for merge.

## Rules that fix the combinations

From Rule 6 on `main`, and the two reference branches:

- `passed` keeps its meaning in every proposal: a consumer that reads only `passed` stays
  correct. Neither RFC changes how `passed` is computed from graders.
- For a Result with some null-scored graders, the default aggregation and #49 disagree on `passed`: the default makes it `true` if every scored grader passed, and `verdict: "unverified"` requires `false`. Neither text says which wins (see the questions below).
- `verdict: "passed"` requires `passed: true`; `"failed"` and `"unverified"` require
  `passed: false` (#49, `VERDICT_PASSED_MISMATCH`).
- With `error` present, `verdict` is absent or `"unverified"` (#49, `VERDICT_ERROR_CONFLICT`).
  This is the strict starting point on purpose: relaxing it later is backward compatible,
  tightening it would not be.
- An invalidating constraint violation forces `passed: false` (#47), so it already forbids
  `verdict: "passed"` through the first rule. No new rule is needed for that pair.
- `error` wins over `constraint_violations` (#47): the row is read as errored, not as
  disqualified-but-scored. If such a row also carries `verdict: "unverified"` (#49), it
  stays in the denominator as unverified.
- `verdict` is epistemic: it says whether the outcome was established, not whether a retry
  is safe. A `"failed"` row caused by a hard constraint did happen (an overreach is not
  undone by failing the row), which is why #49 does not make `"failed"` a retry guarantee.
  #49 does say consumers SHOULD NOT automatically re-run an `unverified` row without
  reconciling external state first.

## What real data exists for each situation

| Situation | Evidence in this repository | What it is not |
|---|---|---|
| Not verified, no `error` | [`examples/langroid/`](../../examples/langroid/): Langroid's `STALLED`, `MAX_TURNS` and `KILL` statuses (real runtime, mocked LLM). | Not a measurement of how often real agents stall. |
| Not verified, with `error` | Same example: `TIMEOUT` (`error.type: "timeout"`). Also #49's conformance fixtures from a real `agent-watch` stall line. | |
| Partly verified (some graders scored, one did not) | [`examples/errored-graders/`](../../examples/errored-graders/): real DeepEval 4.2.8 and Inspect AI 0.3.276 runs with a grader that raises. DeepEval marks the row failed; Inspect AI keeps the scores that were produced and records the error on the sample. Judgeval 1.3.3 was read, not run. [`adapters/agenteval-dotnet-openeval-adapter/`](../../adapters/agenteval-dotnet-openeval-adapter/): real AgentEval 0.43.0-beta `CompositeEval` runs with a judge whose canned reply has no JSON. AgentEval's verdict depends on whether the errored component was declared `Required`: required gives an `error` label (no verdict, `passed: false`), optional gives `pass` with a note saying 1 of 2 components was measured. A required judge that was skipped holds the pass back as `warn` with the composite marked not measured (0.43, changed after this package's report). A judge whose transport throws takes the whole case down instead (no result). | The errors are constructed (no LLM, canned outputs), so the proportions mean nothing about real judges. |
| Hard constraint | Discussion #47's worked example. **Waiting** on a fixture generated from a real AOBench run, which AOBench's maintainer offered to contribute. | No real-output fixture yet. |
| Verified failure vs. not verified, side by side | `examples/langroid/`: the same eight runs written both ways. The pass rate is identical (0.25); the failure count is 1 vs. 6 and `avg_score` is 0.667 vs. 0.25. | Constructed runs, so the proportions mean nothing. |

One finding runs against the proposal and is kept here on purpose. In the Langroid data
every unverified row already has all grader scores `null`, so Rule 6 yields the same
classification as `verdict` would (the example's tests assert this). Nothing in that data
needs the field. What `verdict` would add is the case where a grader did run and score a
row whose outcome was still not established, and the case of a row with no grader results
at all. The #49 reference fixtures include one with `grader_results: []`; this repository
has no real-run example of the scored-but-unverified case yet, which is the evidence the
RFC is still missing.

A second finding points the other way. In the `examples/errored-graders/` runs, a row where one grader scored a pass and another produced no score is exactly that case: under the spec's default aggregation it is `passed: true`, under DeepEval's own behavior it is `false` (pass rate 0.5 vs 0.25 on the same four rows, both documents valid), and Rule 6 has no name for it. It is still the author's own constructed example, so it does not meet the landing bar proposed on the Discussion, and a metadata convention could mark the row too. It also shows that the underlying question is aggregation, which exists with or without a `verdict` field.

The convention was then tested on the same rows (`--mark-partial` in the same example: `metadata.openeval.aggregation_status: "partial"` on a row with some null graders, extending the spec's existing `"unscored"`). With the spec's default aggregation it recovers exactly the classification `verdict` would give on all four rows, validates on `main` with both SDKs, and needs no schema change. Two limits, both asserted in the tests: its reading depends on which aggregation produced `passed`, and the document does not say which (under the fail-closed writing the same marked row reads as a verified failure), whereas `verdict` is asserted by the producer; and the unverified row keeps `passed: true`, so a consumer that reads only `passed` still counts it as a pass, which is the combination #49 forbids. The two are not exclusive; a producer can write both. See the example's README for the table.

A third framework splits the difference. AgentEval (`adapters/agenteval-dotnet-openeval-adapter/`, real runs) makes the same mixed row `pass` when the unmeasured component was optional and `error` (no verdict) when it was required, declared per component on the eval definition, and writes how much of the verdict was measured into the result either way. Its result-level `error` label is a verdict value EvalPort does not have: on the EvalPort side it can only be `passed: false`, which a reader must take as "verified failing" when any grader scored. The package writes that label under `metadata.agenteval`, which is the convention alternative to #49 in practice. Still the author's own example, built against the published package after the maintainer named the extension points.

That example then changed the framework. The package's report in AgentEvalHQ/AgentEval#203 (a required component that returned `skipped` let a composite pass) became AgentEval 0.43.0-beta's "no verdict reads PASS when something failed or was not measured" (AgentEvalHQ/AgentEval#279, merged 2026-10-06, released the same day; the CHANGELOG cites #203). On the same seven cases: a required judge that did not run now holds the pass back as `warn` with `measurement: notMeasured`, a composite-level "not established" that the package maps to `verdict: "unverified"`; a measured failure now decides the verdict even when a required judge errored (`fail`, not `error`), which answers #49's open question 3 the other way from #83's `error` + `failed` rejection, for a component-level error; and an optional component's error no longer decides anything. The maintainer also reviewed the package and, in the same thread, put his framework's `NotMeasured` state next to this RFC's `unverified`: "`NotMeasured` is the one closer to your `unverified`, and it belongs in the denominator your `Result.verdict` RFC describes." That is an outside maintainer's reading of the semantics, recorded here as such; it is not a statement that AgentEval needs the field, and the fixture is still the author's.

## Open questions this page does not settle

#49's reference text lists five (the first three bear on the table above); one more came out of running real frameworks (6), and 3 got a concrete case:

1. Should `needs-input` map to `"unverified"`, to `"failed"`, or to a fourth value? The
   maintainer's latest comment on the Discussion leans toward leaving per-participant
   states like this in `metadata` under a producer-chosen key.
2. Should `"failed"` promise retry safety, or stay purely epistemic as written?
3. Is the `error` plus `"failed"` rule right, or should it be allowed for runs that provably
   never started, and for rows where a grader that ran established the failure? Inspect AI's
   sample error plus a failing scorer is such a row; the #83 branch rejects `failed` on it.
4. Should `verdict` be all-or-none within a `ResultSet`, rather than a SHOULD?
5. Does a rate-based grader need an unverified-trials count next to `trials` / `successes`
   ([Discussion #67](https://github.com/adhabnr-ux/evalport/discussions/67))?
6. For a Result where some graders scored and one produced none, should `Result.passed` default to the AND of the scored graders (the spec's text) or to `false` (DeepEval's behavior)? #49's rule that `unverified` requires `passed: false` assumes the latter without saying so. AgentEval's answer is "whichever the suite declared for that grader" (`Required` per component), which neither the spec's aggregation extension nor #49 can express.

And one about process: the maintainer's proposal on the Discussion is that #49 lands only
with an independent implementer's fixture or statement of need; without one, extend the
comment period and then close the RFC as a Rule 6 plus `metadata` convention.

Comments belong on the Discussions, not here.
