# agenteval-openeval-adapter

Export [AgentEval](https://github.com/lokesh75-kank/agenteval) determinism-sampling reports to [EvalPort](https://github.com/adhabnr-ux/evalport)'s `ResultSet` interchange format — the open standard for portable LLM evaluation results.

This follows from [agenteval#13](https://github.com/lokesh75-kank/agenteval/issues/13), where maintainer [@lokesh75-kank](https://github.com/lokesh75-kank) confirmed the mapping against AgentEval's actual types and said: *"You're welcome to build the adapter against the public `agenteval-core` API; it's MIT and the `SuiteReport`/`ScenarioRunSummary` shapes are stable within 0.3.x."*

## Why a `ResultSet`, not an `EvalSuite`

AgentEval's defining mechanic is **determinism sampling**: `runner.ts` runs every scenario `runs` times and reports `passingRuns / totalRuns` as `ScenarioRunSummary.determinism` — the flakiness signal the tool exists to produce. That's evidence *from an execution*, which is what EvalPort's `ResultSet` document represents (`Result` objects joined by `test_case_id` + `run_id`, with `attempt`/`isolation` for exactly this "repeated trials of the same case" shape — added to the spec for this reason, see [Discussion #22](https://github.com/adhabnr-ux/evalport/discussions/22)). It is not a portable test-case definition (`EvalSuite`) another runner could execute, because AgentEval's assertion vocabulary (`tool_called`, `every_claim_has_citation`, `citations_resolve`, `quote_matches_source`, `refusal`, `recall_at_k`, ...) has no equivalent among EvalPort's 11 well-known grader types.

This package is intentionally **one-directional** (`to_openeval` only, export). Going the other way — turning an EvalPort `TestCase` into an AgentEval `Scenario` — would silently drop that same assertion vocabulary and produce a `Scenario` with no meaningful `asserts`. AgentEval's own README frames the tool as "the reliability and audit-evidence layer," and a `from_openeval` that manufactured empty assertions would misrepresent that.

## Install

```bash
pip install "agenteval-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/agenteval-openeval-adapter"
```

Not yet published to PyPI — installs directly from source via pip's `git+`/`#subdirectory=` support, same as this repo's other adapters.

**A note on `evalport-sdk` versions:** `Result.attempt` and `ResultSet.isolation` — the fields this adapter's whole mapping depends on — were added to the spec in `1.0.0-rc.5` ([Discussion #22](https://github.com/adhabnr-ux/evalport/discussions/22)) and to `evalport-sdk` in `1.3.0`. As of this writing, PyPI still serves `evalport-sdk==1.0.0` (pre-dating that change) — see `sdk/python/pyproject.toml`'s own note on this. Until a `1.3.0+` release is published, install `evalport-sdk` from source too:

```bash
pip install "evalport-sdk @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=sdk/python"
```

## Usage

```python
import json
from agenteval_openeval_adapter import to_openeval

# `report` is AgentEval's SuiteReport, exactly as `agenteval run --json`
# (src/report/json.ts) writes it to disk -- no reshaping needed.
with open("agenteval-report.json") as f:
    report = json.load(f)

result_set = to_openeval(report, run_id="ci-run-42")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

with open("result_set.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

## What the mapping preserves

| AgentEval (`src/core/types.ts`, `src/core/trace.ts`) | EvalPort `ResultSet` |
|---|---|
| `ScenarioRunSummary.perRun[i]` (one of N runs of a scenario) | `Result` with `attempt: i+1`, sharing `test_case_id` |
| Fresh agent invocation per run (`runner.ts`'s `runOnce`) | `ResultSet.isolation: "fresh"` |
| `ScenarioResult.trace.finalText` / `.error` | `Result.actual_output` / `Result.error` |
| `AssertionResult` (per-assertion pass/fail) | `GraderResult` with `type: "agenteval_<kind>"` — open, non-well-known (see below) |
| `ScenarioResult.judge` (LLM-as-judge, self-consistency voting) | `GraderResult` with `grader_id: "judge"`, `type: "agenteval_llm_judge"`, `score` = vote fraction |
| `ScenarioRunSummary.determinism` | `ResultSet.summary["scenarios"][scenario_id]["determinism"]` — preserved explicitly, not left implicit in the attempts |

Two changes from the original issue's sketch, both from maintainer review on agenteval#13:

- **The judge outcome is included.** The original sketch's `grader_results` only covered `assertions`, dropping `perRun[].judge` entirely. `_judge_grader_result()` now maps it, using the self-consistency vote fraction (`passingVotes / votes`) as `score` — and `None`, not a fabricated `0.0`, when `votes` is `0` (AgentEval's own fail-closed case when a scenario declares a judge but no LLM client was supplied to the runner).
- **Determinism is explicit, not implicit.** Rather than making a consumer recompute `passingRuns / totalRuns` by counting passing `attempt`s, `to_openeval()` writes every scenario's `determinism`, `total_runs`, `passing_runs`, and overall `pass` straight into `ResultSet.summary`.

## What doesn't round-trip (being upfront about it)

AgentEval's assertion `kind`s become `GraderResult.type` values like `agenteval_every_claim_has_citation` — readable in any generic EvalPort viewer or report, but only a receiving system that already implements AgentEval's own grounding/assertion logic could re-execute the check. This is an export path for interop and reporting, not a way to make AgentEval scenarios executable by a different runner. That gap was named up front in the original proposal ([agenteval#13](https://github.com/lokesh75-kank/agenteval/issues/13)) and this adapter doesn't attempt to close it.

## Tests

```bash
pip install -e ".[test]"
pytest tests/
```

Tests build fixtures matching AgentEval's real `Scenario`/`ScenarioResult`/`ScenarioRunSummary`/`SuiteReport`/`AgentTrace` shapes (read directly from `src/core/types.ts` and `src/core/trace.ts`), including determinism sampling with mixed pass/fail runs, judge outcomes (including the zero-votes fail-closed case), assertion-kind mapping, run errors, and multi-scenario attempt-uniqueness — and validate every produced document against `openeval.validate.validate_result_set`.

## Credit

Mapping proposed and this adapter's initial version built by Sahi, following [agenteval#13](https://github.com/lokesh75-kank/agenteval/issues/13), where [@lokesh75-kank](https://github.com/lokesh75-kank) reviewed the mapping against AgentEval's real types and flagged the two fixes above. Not affiliated with AgentEval; built against its public `agenteval-core` API per the maintainer's go-ahead, the same standalone-package approach as this repo's other adapters (e.g. [crewai-openeval-adapter](../crewai-openeval-adapter)).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
