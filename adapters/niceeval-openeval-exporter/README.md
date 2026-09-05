# niceeval-openeval-exporter

Export [NiceEval](https://github.com/NiceEval/NiceEval)'s Inspection query protocol results (`run.summary`, optionally enriched with `attempt.get`) to [EvalPort](https://github.com/adhabnr-ux/evalport) `ResultSet` JSON, the open interchange format for portable LLM evaluation results.

## Why a standalone package?

A proposal for an EvalPort export was opened on [NiceEval/NiceEval#196](https://github.com/NiceEval/NiceEval/issues/196). Maintainer [CorrectRoadH](https://github.com/CorrectRoadH) closed it as `not_planned`, explaining (translated from the original Chinese) that NiceEval won't take on the EvalPort dependency itself but had no objection to someone building it independently against the public Inspection query protocol. This package is that independent build, following the same "standalone adapter" playbook as the rest of this repo (see [luml-openeval-adapter](../luml-openeval-adapter), [mlflow-openeval-adapter](../mlflow-openeval-adapter), and others).

## Why "exporter", not "adapter"

NiceEval already uses "Adapter" for the layer that drives a system under test (`defineAgent` / `defineSandboxAgent`). Calling this package an EvalPort "adapter" would collide with that vocabulary, so — matching the original issue's own proposal — this is an **exporter**: it only reads already-computed NiceEval Inspection results and produces EvalPort `ResultSet` documents. It never touches NiceEval eval *definitions*.

## One direction only

There is no `from_openeval()`. NiceEval's assertion vocabulary — `pattern()`, `includes()`, `jsonMatch()`, `closedQA()`, `toolMatch()`, `eventMatch()`, arbitrary `satisfies()` predicates, and their `and()`/`or()`/`not()` combinators (all real, defined in `packages/niceeval/src/assertions/match.ts`) — cannot be reconstructed from an EvalPort `TestCase`/`Grader` pair without either dropping real structure or inventing NiceEval code that was never authored. Same reasoning as [agenteval-openeval-adapter](../agenteval-openeval-adapter) in this repo.

## The most important thing to understand before using this package

The original issue sketched mapping individual NiceEval matchers to individual EvalPort graders, each with its own pass/fail — `pattern()` → `regex`, `includes()` → `contains`, `closedQA()` → `llm_judge`, and so on, one `GraderResult` per assertion.

**That mapping is not implementable against NiceEval's real, public Inspection query protocol.** This was verified by reading the actual protocol schema (`packages/niceeval/src/inspection/results.ts`, in full), not assumed from the issue's sketch. The `attempt.get` operation's `assertions` field (`AssertionIndexSchema`) exposes, per assertion, only:

```ts
{ entryId: string, display: { label?: string, key?: string, groupPath: string[] } }
```

There is no pass/fail, no matched/mismatched/unavailable state, and no matcher-type field anywhere in that schema — `display.key` is an author-supplied opaque label, not a matcher-type identifier, and this package never assumes otherwise. The *only* pass/fail information the protocol exposes is a single, already-folded whole-attempt `verdict` (`"passed" | "failed" | "errored" | "skipped" | null`, the output of `foldVerdict()` in `packages/niceeval/src/eval/record/verdict.ts`) and a whole-attempt `score` (`InspectionScoredValue`, the output of `buildScorePayload()` in `packages/niceeval/src/eval/record/score.ts`).

Given that, fabricating a separate per-assertion `GraderResult` — each carrying a pass/fail this package cannot actually observe — would mean inventing test results. So this exporter builds **exactly one** `GraderResult` per `Result`, carrying the real, whole-attempt verdict and score, and preserves each assertion's real identity (`entryId` / `label` / `key` / `groupPath`) as descriptive metadata only, never as a fabricated separate grade.

## Install

```bash
pip install "niceeval-openeval-exporter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/niceeval-openeval-exporter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working, the same install path documented for [luml-openeval-adapter](../luml-openeval-adapter)).

## Why Python, for a TypeScript-native protocol?

NiceEval's Inspection query protocol is plain, Effect-Schema-encoded JSON end to end. The real `niceeval query run --request <file|-> [--record <file>]` CLI command (`packages/niceeval/src/inspection/cli/contribution.ts`) writes exactly that JSON to stdout — a single, versioned document, not a TypeScript object. A Python package can consume it directly with zero NiceEval runtime dependency, and — critically — this lets every output get validated against the real `openeval.validate.validate_result_set()` in this package's own test suite, the same rigor used for every other adapter in this repo. A TypeScript exporter would have had to either reimplement that validation against EvalPort's raw JSON Schema or skip it.

## Usage

```python
import json
from niceeval_openeval_exporter import to_openeval, to_openeval_json
from openeval.validate import validate_result_set

# Option 1: you already have the parsed run.summary document (or the full
# CLI envelope wrapping it -- extra envelope keys are ignored).
result_set = to_openeval(run_summary)
assert validate_result_set(result_set).valid

# Option 2: raw JSON text, e.g. captured from
#   niceeval query run --request run-summary-request.json
result_set = to_openeval_json(open("run-summary.json").read())

with open("results.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

Enrich with real assertion identities (labels/groupPaths, never fabricated pass/fail) by also passing `attempt.get` documents, joined internally by NiceEval's own `locator` field (the only field both a `run.summary` member and an `attempt.get` result carry — `AttemptDocument` itself has no `attemptOrdinal`, so this join cannot be reconstructed from `slotId`/`evalId`/`attemptOrdinal` alone):

```python
result_set = to_openeval(run_summary, attempts=[attempt_doc_1, attempt_doc_2, ...])
```

### What each field means

| NiceEval (`run.summary` member) | EvalPort `Result` | Notes |
|---|---|---|
| `evalId` (path-derived, stable — `defineEval()`/`defineScoreEval()` reject a caller-supplied `id`, per `packages/niceeval/src/define.ts`) | `test_case_id` | Prefixed with `slotId::` when a run has more than one distinct slot, to keep ids unique. |
| `attemptOrdinal` | `attempt` | NiceEval's ordinal is zero-based (`packages/niceeval/src/record/model/definition.ts`'s `AttemptOrdinalSchema`); EvalPort requires `attempt >= 1` (verified against the real validator), so this exporter stores `attemptOrdinal + 1`. |
| `verdict` (`passed`/`failed`) | `passed` + one `GraderResult` | See below. |
| `verdict == "errored"` or `outcome` in `errored`/`cancelled`/`interrupted` | `passed: false`, `error: {type: "runner_error" \| "assertion_error", message}`, `grader_results: []` | Never a graded `0.0` — "never scored" and "scored zero" are different facts, matching this repo's established convention (see `agenteval-openeval-adapter`). |
| `verdict == "skipped"` | `passed: false`, `error: {type: "skipped", ...}`, `grader_results: []` | EvalPort's `Result` has no "skipped" concept; this is a deliberate, documented choice so a consumer can still tell a skip apart from a real failure by inspecting `error.type`. |
| member never dispatched (`state` in `not-dispatched`/`interrupted`/`missing`, `verdict: null`) | `passed: false`, `error: {type: "not_evaluated", ...}`, `grader_results: []` | Included rather than silently dropped, so `ResultSet.results` accounts for every member NiceEval reported. |
| `score` (`InspectionScoredValue`) | `GraderResult.score` | `earned / possible` when `state` is `"complete"` or `"unavailable"` and `possible > 0`; `null` for `"not-scored"` or a missing/zero denominator. Raw `earned`/`possible`/`unavailable` preserved under `grader_result.metadata.niceeval.score_raw`. |
| `assertions.entries[]` from a joined `attempt.get` | `result.metadata.niceeval.score_raw.assertions[]` (via `grader_results[0].metadata.niceeval.assertions`) | `entry_id`/`label`/`key`/`group_path` only — never a fabricated pass/fail, see above. |

`ResultSet.run_id` / `.suite_id` / `.started_at` / `.completed_at` default to real values read from `run_summary["runs"]` (a NiceEval `RunDocument` has real `runId`, `experimentId`, `startedAt`, `completedAt` fields — verified against `packages/niceeval/src/record/model/definition.ts` — converted from `UtcMillis`, confirmed to be plain Unix-epoch milliseconds per `packages/niceeval/src/record/codec/identifiers.ts`). Unlike some other adapters in this repo (e.g. `luml-openeval-adapter`, where no such field exists anywhere), no fabricated clock reading is ever needed here as long as at least one run is present; override any of them explicitly if you'd rather supply your own.

## Credit

Built in direct response to the proposal and closure on [NiceEval/NiceEval#196](https://github.com/NiceEval/NiceEval/issues/196), and to my own follow-up comment there committing to build it independently. No NiceEval maintainer reviewed or shaped this specific mapping beyond that closing comment — it was designed and verified independently against NiceEval's real, public Inspection query protocol source, exactly as promised.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
