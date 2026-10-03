# Langroid Task terminal states → EvalPort

> **Status: unsolicited example.** This is not an official Langroid integration, has not
> been proposed to or reviewed by the Langroid maintainers, and is not a Langroid
> adapter. It runs the real [Langroid](https://github.com/langroid/langroid) `Task`
> machinery (version 0.68.4) to see which terminal states a Task actually ends in, and
> shows one way to write each of them as an EvalPort ResultSet.
>
> **The LLM is mocked.** Langroid's own `MockLM` answers with fixed strings; no model,
> network or API key is involved. The eight runs are *constructed* to end in different
> states. Their proportions say nothing about how often a real agent stalls or times out.

## The observation

`Task.run()` returns `None` when a task ends as `STALLED` or `MAX_TURNS`. Langroid's
`Task.result()` does this on purpose (its source comment says the result is not known and
Langroid does not want to guess it), and `result()` is documented as overridable. The
`StatusCode` that says *why* the task ended is dropped with it. The other non-`DONE`
endings return something, but not an answer: `TIMEOUT` and `KILL` return whatever message
was pending.

A harness that grades what `run()` returned therefore records "the task never produced
anything" as an empty string that fails `exact_match`: an infrastructure outcome becomes a
wrong answer, and the resulting document is perfectly valid, so no schema validator can flag it.

`langroid_to_evalport.py` subclasses `Task` (`RecordingTask`, which only remembers the
status `result()` was called with), runs eight scenarios, and writes the two documents.
Real run, `langroid==0.68.4`:

| Scenario | Langroid status | `run()` returned | Written as |
|---|---|---|---|
| `done_correct` | `DONE` | `'4'` | passed, score 1 |
| `done_wrong` | `DONE` | `'5'` | failed, score 0 |
| `fixed_turns` | `FIXED_TURNS` | `'4'` | passed, score 1 |
| `stalled` | `STALLED` | `None` | not verified, score `null` |
| `max_turns` | `MAX_TURNS` | `None` | not verified, score `null` |
| `timeout` | `TIMEOUT` | `'still thinking'` | not verified, score `null`, `error.type: "timeout"` |
| `killed` | `KILL` | `'partial work'` | not verified, score `null` |
| `agent_raises` | (none; `run()` raised) | raised `RuntimeError` | not verified, score `null`, `error.type: "runner_error"` |

"Not verified" is [Validation Rule 6](../../SPEC.md): `score: null`, `passed: false`, with
`metadata.openeval.aggregation_status: "unscored"` on the Result. The original status is kept
in `metadata.langroid.status`, and a partial output (`timeout`, `killed`) is kept as
`actual_output` but is not graded.

## Two ways to write the same eight runs

`--policy naive` writes what grading `run()`'s return value would write (`None` → `""`,
everything scored). Both documents validate. Both have a pass rate of 0.25 (2 of 8).

| | `status-aware` (default) | `naive` |
|---|---|---|
| verified passes / verified failures / not verified | 2 / 1 / 5 | 2 / 6 / 0 |
| `summary.avg_score` | 0.6667 (over the 3 scored rows) | 0.25 (over all 8) |
| a stall vs. a wrong answer | different (`null` vs `0`) | identical (`passed: false`, score 0) |

The pass rate does not move; the failure count and the accuracy do. The tests assert these
numbers against the real runs.

## Design choices (judgments, not facts about Langroid)

- **Which statuses are graded.** `DONE` and `FIXED_TURNS` are graded: in the first the task
  said it finished, and in the second the caller asked for exactly N turns, so the last
  message is the result by construction. Everything else is "not verified". Any status this
  example did not observe (including every one in "Not produced" below), an exception, or a
  judged status without content also falls into "not verified": the mapping fails closed.
- **Partial output is not an answer.** A task killed or timed out may have said "4" by
  coincidence; that is not a finished run, so it is not graded. The text is kept for people.
- **`timeout` and exceptions also get `Result.error`.** Langroid's `TIMEOUT` maps onto the
  existing `error.type: "timeout"`; an exception maps onto `runner_error`. `STALLED`,
  `MAX_TURNS` and `KILL` get **no** `error`: Langroid reports them as ordinary statuses,
  not as exceptions or failures of the runner. For those three, Rule 6 is the only place
  the signal can live.
- **`summary`.** The spec does not say how `summary` counts a Result whose graders are all
  unscored. This example counts them as `skipped` so `total = passed + failed + skipped`,
  keeps them in the `pass_rate` denominator, and leaves `null` scores out of `avg_score` and
  `by_grader`. That is a choice, stated here so nobody reads `skipped` as a spec rule.

## Not produced

These `StatusCode` values were **not** observed, so nothing here claims how they behave:
`OK`, `ERROR`, `MAX_COST`, `MAX_TOKENS`, `NO_ANSWER`, `USER_QUIT`, `INF_LOOP`. With
`MockLM`, `run(max_tokens=1)` and `run(max_cost=...)` fell back to `STALLED` instead of
`MAX_TOKENS`/`MAX_COST`, so those limits cannot be exercised without a real model.
`INF_LOOP` is, per the source, raised as an `InfiniteLoopException`; the attempts here
ended as `STALLED` instead. `USER_QUIT` needs interactive mode.

## Relation to Discussion #49 (`Result.verdict`)

`--proposed-verdict` adds the `verdict` field proposed in
[Discussion #49](https://github.com/adhabnr-ux/evalport/discussions/49) (draft PR #83).
**It is not in the spec** and the flag exists to test the proposal against real data.

What this data shows, both ways:

- In these eight rows, every unverified row already has all grader scores `null`, so
  Rule 6 alone yields the same classification as `verdict` (asserted in the tests). Nothing
  here *needs* the field.
- Three of the five non-answers (`STALLED`, `MAX_TURNS`, `KILL`) have no `error` to carry
  the signal, which is why `verdict` could not be folded into `error`. This example shows
  Rule 6 doing that job, not that `verdict` is required.
- The `TIMEOUT` row has both `error` (type `timeout`) and an unverified outcome. Today a
  consumer may drop a bare `error` row from its denominators; Discussion #49 proposes that
  setting `verdict: "unverified"` states the row must stay in them.
- The tests check the `verdict` variant against the PR #83 validator, when you point
  `EVALPORT_RFC49_CHECKOUT` at a checkout of that branch (skipped otherwise, and in CI,
  because the branch is a proposal). That includes the two proposed rejections,
  `VERDICT_PASSED_MISMATCH` and `VERDICT_ERROR_CONFLICT`.

## What is lost

The Langroid message history, tool calls and costs are not written; `actual_output` is the
text of the last message only. One `Task` per test case, one grader, no per-turn data. The
status vocabulary is Langroid's and lives in `metadata.langroid`; EvalPort does not
standardize terminal states.

## Run it

```bash
pip install -e ../../sdk/python -r requirements.txt     # from this directory
python langroid_to_evalport.py --out-dir out            # writes out/suite.json, out/results.json
python langroid_to_evalport.py --out-dir out --policy naive
python langroid_to_evalport.py --out-dir out --proposed-verdict
python -m pytest -p no:cacheprovider .                  # 47 tests; 52 with the RFC #49 branch
```

`--deterministic` fixes `started_at` and drops timestamps and durations; the checked-in
[`sample_output/`](sample_output/) is that output, and a test fails if it goes stale.
To run the opt-in `verdict` checks:

```bash
git fetch origin pull/83/head:rfc49 && git worktree add ../rfc49-checkout rfc49
EVALPORT_RFC49_CHECKOUT=$PWD/../rfc49-checkout python -m pytest -p no:cacheprovider .
```
