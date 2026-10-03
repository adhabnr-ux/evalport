# How a Result's outcome is represented: what `main` says and what the open RFCs would add

> **Not normative.** This page is a reading aid for two open RFCs, written so their
> interaction can be checked in one place. Where it describes a proposal it says so. If it
> disagrees with [`spec/SPEC.md`](../../spec/SPEC.md), the spec wins and this page is wrong.
> Nothing here has landed except the rows marked "on `main`".

A `Result` has one required boolean, `passed`. Evaluation runs end in more situations than
a boolean can name. This page lists five, says how each is written today, and what
[Discussion #49](https://github.com/adhabnr-ux/evalport/discussions/49) (`Result.verdict`)
and [Discussion #47](https://github.com/adhabnr-ux/evalport/discussions/47)
(`Result.constraint_violations`) would add. Both are open; neither is on `main`.

## The five situations

| Situation | On `main` today | If #49 lands (`verdict`) | If #47 lands (`constraint_violations`) | Result-level denominator |
|---|---|---|---|---|
| **Passed**: graders scored it and it met the bar. | `passed: true`; `score` values in [0, 1]. | `verdict: "passed"` (optional). Requires `passed: true`. | no change | Counted. |
| **Failed (verified)**: graders ran and it did not meet the bar. | `passed: false`; at least one non-null `score` (Rule 6: "verified failing"). | `verdict: "failed"` (optional). Requires `passed: false`. | no change | Counted. |
| **Not verified**: no grader established an outcome (a pending `human` review, an unsupported grader type, a runner error before scoring, a stalled agent). | Every `GraderResult` has `score: null` and `passed: false` (Rule 6); the `Result` is `passed: false`, and `metadata.openeval.aggregation_status: "unscored"` is RECOMMENDED. Null scores are left out of aggregation, not counted as failures. | `verdict: "unverified"`: the producer's explicit statement that the row stays in result-level denominators and is reported as unverified. Requires `passed: false`. | no change | Rule 6 says consumers MUST NOT treat a null as a scored failure and may exclude it from the aggregate denominator or report it separately. #49 would add: an `unverified` row MUST NOT be dropped from `summary.total`. |
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

## Open questions this page does not settle

#49's reference text lists five (the first three bear on the table above):

1. Should `needs-input` map to `"unverified"`, to `"failed"`, or to a fourth value? The
   maintainer's latest comment on the Discussion leans toward leaving per-participant
   states like this in `metadata` under a producer-chosen key.
2. Should `"failed"` promise retry safety, or stay purely epistemic as written?
3. Is the `error` plus `"failed"` rule right, or should it be allowed for runs that provably
   never started?
4. Should `verdict` be all-or-none within a `ResultSet`, rather than a SHOULD?
5. Does a rate-based grader need an unverified-trials count next to `trials` / `successes`
   ([Discussion #67](https://github.com/adhabnr-ux/evalport/discussions/67))?

And one about process: the maintainer's proposal on the Discussion is that #49 lands only
with an independent implementer's fixture or statement of need; without one, extend the
comment period and then close the RFC as a Rule 6 plus `metadata` convention.

Comments belong on the Discussions, not here.
